"""`ys`: add sources, fetch transcripts, inspect status, export."""

from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select

from ytscribe import ytdlp_runner as yt
from ytscribe.db.models import Segment, Source, Video, VideoSource
from ytscribe.db.session import get_session, init_db
from ytscribe.export import export_video, write_catalog
from ytscribe.logging_setup import setup_logging
from ytscribe.pipeline import POLICIES, Runner, enrich, run_batch, select_videos, upsert_source
from ytscribe.quality import quality_warnings
from ytscribe.settings import get_settings, load_config, load_env
from ytscribe.sources.resolve import event_from_title, expand_source_file, playlist_id, resolve

app = typer.Typer(help="Get transcripts for YouTube videos, playlists and channels.", no_args_is_help=True)


def _optional(name: str, module: str, attr: str, extra: str) -> None:
    """Register a sub-app whose dependencies are an optional extra; without them `ys <name>` explains how to install them."""
    try:
        app.add_typer(getattr(__import__(module, fromlist=[attr]), attr), name=name)
    except ImportError as e:
        missing = e.name  # `e` is deleted when the except block ends, so keep what the stub needs
        stub = typer.Typer(help=f"Needs the '{extra}' extra: uv sync --extra {extra}  ({missing} is missing)")

        @stub.callback(invoke_without_command=True)
        def _missing():
            console.print(f"`ys {name}` needs the '{extra}' extra: uv sync --extra {extra}  ({missing} is missing)")
            raise typer.Exit(1)

        app.add_typer(stub, name=name)


_optional("topics", "ytscribe.analysis.cli", "topics_app", "analysis")
_optional("llm", "ytscribe.analysis.llm.cli", "llm_app", "llm")
console = Console()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG on the console")):
    load_env()
    s = get_settings()
    log_file = setup_logging(s.logs_dir, "DEBUG" if verbose else s.log_level)
    logger.debug("ys started, log file {}", log_file)


@app.command()
def init():
    """Create the database."""
    init_db()
    console.print(f"database ready: {get_settings().db_url}")


@app.command()
def add(target: str = typer.Argument(..., help="YouTube URL (video/playlist/channel) or a configs/sources/*.yaml file")):
    """Enumerate a source and register its videos (idempotent; nothing is downloaded)."""
    init_db()
    resolved = (expand_source_file(Path(target)) if target.endswith((".yaml", ".yml")) else [resolve(target)])
    total_new = total = 0
    with get_session() as session:
        for r in resolved:
            _, new = upsert_source(session, r)
            total, total_new = total + len(r.videos), total_new + new
    console.print(f"{len(resolved)} source(s), {total} video entries, {total_new} new videos registered")


@app.command(name="list")
def list_(source: str = typer.Option(None, help="Filter by source title/url"),
          status: str = typer.Option(None, help="discovered | done | failed | unavailable")):
    """List registered videos."""
    with get_session() as session:
        q = select(Video)
        if status:
            q = q.where(Video.status == status)
        if source:
            q = q.join(VideoSource).join(Source).where(Source.title.ilike(f"%{source}%"))
        t = Table("id", "status", "origin", "dur", "title")
        for v in session.scalars(q.order_by(Video.upload_date, Video.id)).unique():
            t.add_row(v.id, v.status, v.origin or "", f"{(v.duration_s or 0) / 60:.0f}m", (v.title or "")[:70])
        console.print(t)


@app.command()
def fetch(
    source: str = typer.Option(None, help="Only videos of this source (title substring or URL)"),
    video: list[str] = typer.Option(None, "--video", help="Specific video id(s); reprocessed even if done"),
    limit: int = typer.Option(None, help="Process at most N videos"),
    policy: str = typer.Option(None, help="reuse | verify | always_asr (default from config)"),
    asr: str = typer.Option(None, help="whisper | mistral (default from config)"),
    retry_failed: bool = typer.Option(False, help="Also retry failed videos (up to 3 attempts)"),
    no_auto_update: bool = typer.Option(False, help="Do not upgrade yt-dlp on extractor failures"),
    config: str = typer.Option("default.yaml", help="Config file in configs/"),
):
    """Fetch transcripts: captions if available, otherwise ASR. Resumable: finished videos are skipped."""
    cfg, settings = load_config(config), get_settings()
    policy = policy or cfg.transcript_policy
    if policy not in POLICIES:
        raise typer.BadParameter(f"policy must be one of {POLICIES}")
    if asr:
        cfg.asr.backend = asr
    yt.configure(auto_update=cfg.ytdlp.auto_update and not no_auto_update)
    logger.info("fetch: policy={} asr={} yt-dlp={} auto_update={}", policy, cfg.asr.backend, yt.version(),
                cfg.ytdlp.auto_update and not no_auto_update)
    init_db()
    with get_session() as session:
        videos = select_videos(session, source, retry_failed, limit, ids=video or None)
        if not videos:
            console.print("nothing to fetch (use `ys add` first, or --retry-failed / --video)")
            return
        if policy != "reuse" or cfg.asr.backend == "mistral":
            logger.info("note: ASR may run, backend={}", cfg.asr.backend)
        stats = run_batch(Runner(cfg, settings, policy, session), videos)
        write_catalog(session, settings.data_dir)
    console.print(f"done: {stats['done']}  failed: {stats['failed']}  unavailable: {stats['unavailable']}  "
                  f"origins: {stats['origins']}")
    if stats["failed"]:
        console.print("[red]some videos failed[/red]: see `ys status` and the log in logs/")


@app.command()
def slides(
    video: list[str] = typer.Option(None, "--video", help="Video id(s); default: all done videos with a low-content transcript"),
    force: bool = typer.Option(False, help="Redo videos that already have slides"),
    config: str = typer.Option("default.yaml"),
):
    """Extract distinct frames + OCR text from slide-only videos and add them to the transcript (no LLM involved)."""
    cfg, settings = load_config(config), get_settings()
    yt.configure(auto_update=cfg.ytdlp.auto_update)
    init_db()
    with get_session() as session:
        q = select(Video).where(Video.status == "done")
        if video:
            q = q.where(Video.id.in_(video))
        cands = []
        for v in session.scalars(q.order_by(Video.id)).unique():
            if v.frames and not force:
                continue
            if video or quality_warnings(v.segments, v.duration_s):
                cands.append(v)
        logger.info("slides: {} videos to process", len(cands))
        runner = Runner(cfg, settings, "reuse", session)
        ok = failed = 0
        for i, v in enumerate(cands, 1):
            with logger.contextualize(video_id=v.id):
                log = logger.bind(stage="slides")
                log.info("[{}/{}] {} ({:.1f} min)", i, len(cands), v.title, (v.duration_s or 0) / 60)
                try:
                    runner.run_slides(v, log)
                    session.commit()
                    export_video(v, settings.data_dir)
                    ok += 1
                except Exception as e:
                    session.rollback()
                    failed += 1
                    log.exception("slides failed: {}", e)
        runner.close()
        write_catalog(session, settings.data_dir)
    console.print(f"slides: {ok} done, {failed} failed")


@app.command()
def status():
    """Counts by status and origin, plus failures."""
    with get_session() as session:
        t = Table("status", "videos", "minutes")
        for st, n, dur in session.execute(select(Video.status, func.count(), func.coalesce(func.sum(Video.duration_s), 0))
                                          .group_by(Video.status)):
            t.add_row(st, str(n), f"{dur / 60:.0f}")
        console.print(t)
        o = Table("origin", "videos")
        for origin, n in session.execute(select(Video.origin, func.count()).where(Video.status == "done")
                                         .group_by(Video.origin)):
            o.add_row(origin or "", str(n))
        console.print(o)
        total = session.scalar(select(func.count()).select_from(Video)) or 0
        failed = session.scalar(select(func.count()).select_from(Video).where(Video.status == "failed")) or 0
        if total and failed / total >= 0.1:
            console.print(f"[red]failure rate {failed}/{total} ({failed / total:.0%})[/red]")
        for v in session.scalars(select(Video).where(Video.status.in_(["failed", "unavailable"]))):
            console.print(f"[{'red' if v.status == 'failed' else 'yellow'}]{v.status}[/] {v.id} (attempts {v.attempts}) "
                          f"{(v.title or '')[:50]}: {(v.error or '')[:150]}")
        console.print(f"segments in DB: {session.scalar(select(func.count()).select_from(Segment))}")


@app.command(name="enrich")
def enrich_(all_: bool = typer.Option(False, "--all", help="Refetch metadata even for videos that already have it"),
            sleep: float = typer.Option(1.0, help="Seconds between requests"),
            config: str = typer.Option("default.yaml")):
    """Fetch metadata (upload date, description, tags, chapters, views...) for registered videos, no transcription.
    Also refreshes playlist ids/events on sources. Then rewrites transcript files and the catalog."""
    cfg, settings = load_config(config), get_settings()
    yt.configure(auto_update=cfg.ytdlp.auto_update)
    init_db()
    with get_session() as session:
        for src in session.scalars(select(Source)):
            src.external_id = src.external_id or playlist_id(src.url)
            if src.kind == "playlist":
                src.event_year, src.event_season = event_from_title(src.title)
        session.commit()
        q = select(Video) if all_ else select(Video).where(Video.metadata_fetched_at.is_(None))
        videos = [v for v in session.scalars(q.order_by(Video.id)).unique() if v.status != "unavailable"]
        logger.info("enrich: {} videos need metadata", len(videos))
        opts = {"retries": cfg.ytdlp.retries}
        stats = enrich(session, videos, opts, sleep)
        n = 0
        for v in session.scalars(select(Video).where(Video.status == "done")).unique():
            export_video(v, settings.data_dir)
            n += 1
        csv_path, _, rows = write_catalog(session, settings.data_dir)
    console.print(f"metadata: {stats}; re-exported {n} transcripts; catalog: {rows} rows -> {csv_path}")


@app.command()
def catalog():
    """Write data/catalog.csv and catalog.jsonl: one row per video, sorted by date, with playlist/event metadata."""
    with get_session() as session:
        csv_path, jsonl_path, n = write_catalog(session, get_settings().data_dir)
    console.print(f"{n} rows -> {csv_path} and {jsonl_path}")


@app.command()
def export(video: list[str] = typer.Option(None, "--video", help="Only these ids (default: all done)")):
    """Rewrite the transcript files (json/srt/txt/meta) from the database."""
    settings = get_settings()
    with get_session() as session:
        q = select(Video).where(Video.status == "done")
        if video:
            q = q.where(Video.id.in_(video))
        n = 0
        for v in session.scalars(q):
            export_video(v, settings.data_dir)
            n += 1
        write_catalog(session, settings.data_dir)
    console.print(f"exported {n} transcripts to {settings.data_dir / 'transcripts'} and refreshed the catalog")
