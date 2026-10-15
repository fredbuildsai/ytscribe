"""Per-video state machine: metadata -> captions (policy permitting) -> else audio -> ASR -> DB + files.

Each video is independent: a failure is recorded on the video and the batch carries on."""

import re
import shutil
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from ytscribe import captions
from ytscribe import ytdlp_runner as yt
from ytscribe.asr.backends.base import AsrBackend, AsrSegment
from ytscribe.asr.backends.registry import make_backend, origin_of
from ytscribe.audio import download_audio
from ytscribe.db.models import Frame, Segment, Source, Video, VideoSource
from ytscribe.export import export_video
from ytscribe.quality import quality_warnings
from ytscribe.settings import Config, Settings
from ytscribe.sources.resolve import Resolved, event_from_title, playlist_id

POLICIES = ("reuse", "verify", "always_asr")
MAX_ATTEMPTS = 3
FAILURE_ALERT_RATE = 0.3  # warn loudly when this share of a batch has failed (min 5 videos)


def upsert_source(session: Session, r: Resolved) -> tuple[Source, int]:
    """Store a resolved source and its videos. Returns (source, number of new videos)."""
    src = session.scalar(select(Source).where(Source.url == r.url))
    if not src:
        src = Source(kind=r.kind, url=r.url, title=r.title)
        session.add(src)
        session.flush()
    elif r.title:
        src.title = r.title
    src.external_id = r.external_id or src.external_id or playlist_id(r.url)
    if r.kind == "playlist":
        src.event_year, src.event_season = event_from_title(src.title)
    known = set(session.scalars(select(VideoSource.video_id).where(VideoSource.source_id == src.id)))
    new = 0
    for pos, ref in enumerate(r.videos):
        if not session.get(Video, ref.id):
            session.add(Video(id=ref.id, title=ref.title))
            new += 1
        if ref.id not in known:
            session.add(VideoSource(video_id=ref.id, source_id=src.id, position=pos))
    session.flush()
    logger.bind(stage="resolve").info("source '{}' ({}): {} videos, {} new", src.title, src.kind, len(r.videos), new)
    return src, new


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}


def word_overlap(reference: str, hypothesis: str) -> dict:
    """Content-word recall/precision of `hypothesis` against `reference` (same metric family as the ParlWatch M0 bake-off)."""
    ref, hyp = _content_words(reference), _content_words(hypothesis)
    both = ref & hyp
    return {"recall": round(len(both) / len(ref), 3) if ref else None,
            "precision": round(len(both) / len(hyp), 3) if hyp else None, "ref_words": len(ref), "hyp_words": len(hyp)}


def apply_info(video: Video, info: dict) -> None:
    """Copy the analysis-relevant yt-dlp metadata onto the video (used by fetch and by `ys enrich`)."""
    video.title = info.get("title") or video.title
    video.channel = info.get("channel") or info.get("uploader") or video.channel
    video.channel_id = info.get("channel_id") or video.channel_id
    video.upload_date = info.get("upload_date") or video.upload_date
    video.duration_s = info.get("duration") or video.duration_s
    video.description = info.get("description")
    video.tags = info.get("tags") or None
    video.categories = info.get("categories") or None
    video.chapters = [{"start_time": c.get("start_time"), "end_time": c.get("end_time"), "title": c.get("title")}
                      for c in info.get("chapters") or []] or None
    video.language = info.get("language")
    video.view_count, video.like_count = info.get("view_count"), info.get("like_count")
    video.comment_count = info.get("comment_count")
    video.thumbnail = info.get("thumbnail")
    video.metadata_fetched_at = datetime.now()


def enrich(session: Session, videos: list[Video], opts: dict, sleep_s: float = 1.0) -> dict:
    """Fetch metadata only (no transcription) for already-registered videos. Safe to re-run."""
    stats = {"enriched": 0, "failed": 0}
    for i, v in enumerate(videos, 1):
        with logger.contextualize(video_id=v.id):
            try:
                apply_info(v, yt.extract(v.url, {**opts, "skip_download": True}, stage="metadata", video_id=v.id))
                session.commit()
                stats["enriched"] += 1
                logger.bind(stage="enrich").info("[{}/{}] metadata stored: {} ({})", i, len(videos), v.title, v.upload_date)
            except Exception as e:
                session.rollback()
                stats["failed"] += 1
                logger.bind(stage="enrich").error("[{}/{}] metadata failed: {}", i, len(videos), e)
        time.sleep(sleep_s)
    logger.bind(stage="enrich").info("enrich summary: {}", stats)
    return stats


@dataclass
class Runner:
    cfg: Config
    settings: Settings
    policy: str
    session: Session
    backend: AsrBackend | None = None
    stats: dict = field(default_factory=lambda: {"done": 0, "failed": 0, "unavailable": 0, "origins": {}})
    _pool: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch"))
    _prefetched: dict[str, Future] = field(default_factory=dict)

    # --- audio prefetch: while video N is processed, fetch audio for N+1 if it will need ASR -------------------
    def prefetch(self, video_id: str) -> None:
        if video_id not in self._prefetched:
            self._prefetched[video_id] = self._pool.submit(self._prefetch_audio, video_id)

    def _prefetch_audio(self, video_id: str) -> Path | None:
        """Runs in the worker thread: touches only yt-dlp and disk, never the DB session. Failures are non-fatal."""
        with logger.contextualize(video_id=video_id, stage="prefetch"):
            try:
                if self.policy != "always_asr":
                    info = yt.extract(f"https://www.youtube.com/watch?v={video_id}",
                                      {**self._ytdlp_opts(), "skip_download": True})
                    if captions.pick_track(info, self.cfg.captions):
                        logger.info("has a caption track, no audio prefetch")
                        return None
                self._check_disk()
                path = download_audio(video_id, self.settings.data_dir / "audio", self._ytdlp_opts())
                logger.info("audio prefetched while the previous video was processing")
                return path
            except Exception as e:
                logger.warning("prefetch failed, the video will download its own audio: {!r}", e)
                return None

    def _take_audio(self, video_id: str) -> Path:
        fut = self._prefetched.pop(video_id, None)
        path = fut.result() if fut else None
        if path and path.is_file():
            logger.info("using prefetched audio {}", path.name)
            return path
        return download_audio(video_id, self.settings.data_dir / "audio", self._ytdlp_opts())

    def _discard_prefetched(self, video_id: str) -> None:
        """The video turned out not to need ASR: drop its speculative audio."""
        fut = self._prefetched.pop(video_id, None)
        path = fut.result() if fut else None
        if path and not self.cfg.keep_audio:
            logger.info("prefetched audio not needed (captions found), deleting")
            path.unlink(missing_ok=True)

    def close(self) -> None:
        for vid in list(self._prefetched):
            self._discard_prefetched(vid)
        self._pool.shutdown(wait=True)

    def _backend(self) -> AsrBackend:
        if self.backend is None:
            self.backend = make_backend(self.cfg.asr, self.settings.data_dir.parent / "models")
        return self.backend

    def _ytdlp_opts(self) -> dict:
        y, opts = self.cfg.ytdlp, {"retries": self.cfg.ytdlp.retries, "extractor_retries": self.cfg.ytdlp.retries}
        if y.cookies_from_browser:
            opts["cookiesfrombrowser"] = (y.cookies_from_browser,)
        return opts

    def _check_disk(self) -> None:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.settings.data_dir).free / 1e9
        if free < self.cfg.min_free_disk_gb:
            raise OSError(f"only {free:.1f} GB free, below min_free_disk_gb={self.cfg.min_free_disk_gb}")

    def _asr(self, video: Video, log) -> tuple[list[AsrSegment], str]:
        self._check_disk()
        backend = self._backend()
        audio = self._take_audio(video.id)
        t0 = time.monotonic()
        try:
            kwargs = {"checkpoint_dir": self.settings.data_dir / "asr_checkpoints" / video.id} \
                if self.cfg.asr.backend == "whisper" else {}
            segs = backend.transcribe(str(audio), self.cfg.language, **kwargs)
        finally:
            if not self.cfg.keep_audio:
                audio.unlink(missing_ok=True)
        dur = video.duration_s or (segs[-1].end_s if segs else 0)
        took = time.monotonic() - t0
        log.info("ASR done: {} segments in {:.0f}s ({:.1f}x realtime)", len(segs), took, dur / took if took else 0)
        if self.cfg.asr.backend == "whisper":
            shutil.rmtree(self.settings.data_dir / "asr_checkpoints" / video.id, ignore_errors=True)
        return segs, origin_of(backend)

    def run_slides(self, video: Video, log) -> dict:
        """Extract distinct frames + OCR text and store them as slide segments (origin 'slides:ocr:apple-vision').

        Hallucinated ASR (repeated lines) is dropped; other ASR text is kept alongside the slides."""
        from ytscribe.slides import extract_slides
        from ytscribe.video import download_video

        self._check_disk()
        scfg = self.cfg.slides
        vid_path = download_video(video.id, self.settings.data_dir / "video", self._ytdlp_opts(), scfg.max_height)
        frames_dir = self.settings.data_dir / "frames" / video.id
        shutil.rmtree(frames_dir, ignore_errors=True)
        try:
            slides = extract_slides(vid_path, video.id, video.duration_s, frames_dir, scfg)
        finally:
            if not scfg.keep_video:
                vid_path.unlink(missing_ok=True)
        origin = "slides:ocr:apple-vision"
        dropped_asr = 0
        if any(w.startswith("repetitive") for w in quality_warnings(video.segments, video.duration_s)):
            dropped_asr = len([x for x in video.segments if not x.origin.startswith("slides")])
            log.warning("dropping {} hallucinated ASR segments", dropped_asr)
        for seg in list(video.segments):
            if seg.origin.startswith("slides") or dropped_asr:  # old slides always replaced; ASR only if hallucinated
                video.segments.remove(seg)
        video.frames.clear()
        self.session.flush()
        for sl in slides:
            video.frames.append(Frame(idx=sl.idx, start_s=sl.start_s, end_s=sl.end_s, path=str(sl.path),
                                      ocr_text=sl.text, ocr_conf=sl.conf))
            if sl.text:
                video.segments.append(Segment(start_s=sl.start_s, end_s=sl.end_s, text=sl.text, origin=origin))
        kept_asr = {x.origin for x in video.segments if not x.origin.startswith("slides")}
        video.origin = "+".join(sorted(kept_asr) + [origin])
        info = {"frames": len(slides), "ocr_chars": sum(len(x.text) for x in slides), "dropped_asr": dropped_asr,
                "fps": scfg.fps, "same_thresh": scfg.same_thresh, "dup_thresh": scfg.dup_thresh,
                "min_run_s": scfg.min_run_s}
        video.meta = {**(video.meta or {}), "slides": info}
        self.session.flush()
        log.info("slides stored: {}", info)
        return info

    def _verify(self, video: Video, caps: captions.CaptionResult, log) -> dict:
        """ASR a prefix of the audio and compare with the captions over the same time span."""
        minutes = self.cfg.verify_minutes
        audio = self._take_audio(video.id)
        sample = audio.with_name(f"{video.id}.sample.flac")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-t", str(minutes * 60), "-i", str(audio), "-vn", "-ac",
                        "1", "-ar", "16000", str(sample)], check=True)
        try:
            asr = self._backend().transcribe(str(sample), self.cfg.language)
        finally:
            sample.unlink(missing_ok=True)
            if not self.cfg.keep_audio:
                audio.unlink(missing_ok=True)
        cutoff = minutes * 60
        ref = " ".join(s.text for s in caps.segments if s.start_s < cutoff)
        result = {"minutes": minutes, "asr": origin_of(self._backend()), **word_overlap(ref, " ".join(s.text for s in asr))}
        log.info("verify: captions vs ASR over first {} min: {}", minutes, result)
        return result

    def process(self, video: Video) -> str:
        # every log line below this point, including the ASR/ffmpeg ones, carries video_id and attempt
        with logger.contextualize(video_id=video.id, attempt=video.attempts + 1):
            try:
                return self._process(video)
            finally:
                self._discard_prefetched(video.id)

    def _process(self, video: Video) -> str:
        log = logger.bind(stage="pipeline")
        video.attempts += 1
        t0 = time.monotonic()
        try:
            log.info("start '{}' (policy={}, yt-dlp {})", video.title, self.policy, yt.version())
            self._check_disk()
            info = yt.extract(video.url, {**self._ytdlp_opts(), "skip_download": True}, stage="metadata",
                              video_id=video.id)
            apply_info(video, info)
            if info.get("is_live") or info.get("live_status") in ("is_upcoming", "is_live"):
                raise yt.VideoUnavailable(f"live_status={info.get('live_status')}")
            meta = {"ytdlp_version": yt.version(), "policy": self.policy, "language": self.cfg.language}

            segs, origin, caps = None, None, None
            if self.policy != "always_asr":
                try:
                    caps = captions.fetch_ytdlp(video.id, info, self.cfg.captions, self._ytdlp_opts())
                except yt.YtdlpFailure as e:  # e.g. HTTP 429 on the caption endpoint: try other routes, don't fail
                    log.warning("yt-dlp caption fetch failed ({}), trying fallbacks", str(e)[:120])
                    caps = None
                if not caps and self.cfg.captions.api_fallback:
                    log.info("yt-dlp gave no captions, trying youtube-transcript-api")
                    caps = captions.fetch_api(video.id, self.cfg.captions)
                if caps:
                    segs, origin = caps.segments, caps.origin
                    meta["captions_via"] = caps.via
                else:
                    log.info("no captions available -> falling back to ASR")
            else:
                log.info("policy always_asr: skipping captions")

            if segs is None:
                segs, origin = self._asr(video, log)
            elif self.policy == "verify":
                meta["verify"] = self._verify(video, caps, log)

            if not segs:
                raise RuntimeError("transcript is empty")
            warnings = quality_warnings(segs, video.duration_s)
            for w in warnings:
                log.warning("QUALITY: {}", w)
            if warnings:
                meta["warnings"] = warnings
            video.segments.clear()
            self.session.flush()
            video.segments.extend(Segment(start_s=s.start_s, end_s=s.end_s, text=s.text, origin=origin, words=s.words or None)
                                  for s in segs)
            meta["segments"], meta["words"] = len(segs), sum(len(x.text.split()) for x in segs)
            meta["caption_langs"] = {"manual": sorted(info.get("subtitles") or {}),
                                     "auto_count": len(info.get("automatic_captions") or {})}
            video.origin, video.status, video.error, video.meta = origin, "done", None, meta
            self.session.flush()
            if self.cfg.slides.auto and warnings:
                log.info("low-content transcript ({}), extracting slides from the video", "; ".join(warnings))
                try:
                    self.run_slides(video, log)
                except Exception as e:  # the speech/captions transcript stays; the slide step is additive
                    log.exception("slide extraction failed, keeping the existing transcript: {}", e)
                    video.meta = {**(video.meta or {}), "slides_error": str(e)[:300]}
            out = export_video(video, self.settings.data_dir)
            log.info("DONE origin={} segments={} chars={} in {:.0f}s -> {}", origin, len(segs),
                     sum(len(s.text) for s in segs), time.monotonic() - t0, out)
            self.stats["done"] += 1
            self.stats["origins"][origin] = self.stats["origins"].get(origin, 0) + 1
        except yt.VideoUnavailable as e:
            video.status, video.error = "unavailable", str(e)[:2000]
            log.warning("UNAVAILABLE: {}", e)
            self.stats["unavailable"] += 1
        except Exception as e:
            video.status, video.error = "failed", f"{type(e).__name__}: {e}"[:2000]
            log.exception("FAILED after {:.0f}s: {}", time.monotonic() - t0, e)
            self.stats["failed"] += 1
        self.session.commit()
        return video.status


def select_videos(session: Session, source: str | None, retry_failed: bool, limit: int | None,
                  ids: list[str] | None = None, force: bool = False) -> list[Video]:
    q = select(Video)
    if ids:
        q = q.where(Video.id.in_(ids))
    elif not force:
        wanted = ["discovered"] + (["failed"] if retry_failed else [])
        q = q.where(Video.status.in_(wanted))
        if retry_failed:
            q = q.where((Video.status != "failed") | (Video.attempts < MAX_ATTEMPTS))
    if source:
        q = q.join(VideoSource).join(Source).where((Source.url == source) | (Source.title.ilike(f"%{source}%")))
    q = q.order_by(Video.id)
    videos = list(session.scalars(q).unique())
    return videos[:limit] if limit else videos


def run_batch(runner: Runner, videos: list[Video]) -> dict:
    log = logger.bind(stage="batch")
    log.info("batch start: {} videos, policy={}, asr={}", len(videos), runner.policy, runner.cfg.asr.backend)
    sleep = runner.cfg.ytdlp.sleep_between_videos_s
    for i, v in enumerate(videos, 1):
        log.info("[{}/{}] {} {}", i, len(videos), v.id, v.title or "")
        if i < len(videos):
            runner.prefetch(videos[i].id)  # videos[i] is the NEXT video (0-based)
        runner.process(v)
        s = runner.stats
        n = s["done"] + s["failed"] + s["unavailable"]
        if n >= 5 and (s["failed"] / n) >= FAILURE_ALERT_RATE:
            log.error("HIGH FAILURE RATE: {}/{} videos failed so far, check the log (rate limiting? yt-dlp broken?)",
                      s["failed"], n)
        if i < len(videos):
            time.sleep(sleep)
    runner.close()
    log.info("batch summary: {}", runner.stats)
    return runner.stats


__all__ = ["POLICIES", "Runner", "apply_info", "enrich", "run_batch", "select_videos", "upsert_source", "word_overlap"]
