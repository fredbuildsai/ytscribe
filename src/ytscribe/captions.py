"""Existing-caption path: yt-dlp subtitle download (manual preferred over auto), VTT parsing via webvtt-py, and a
fallback through youtube-transcript-api. Only the auto-caption rollup cleanup is custom: YouTube auto VTT repeats the
previous line inside every cue, so naive parsing duplicates all text."""

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import webvtt
from loguru import logger

from ytscribe import ytdlp_runner as yt
from ytscribe.asr.backends.base import AsrSegment
from ytscribe.settings import CaptionsCfg

_TAGS = re.compile(r"<[^>]+>")


@dataclass
class CaptionResult:
    segments: list[AsrSegment]
    kind: str  # manual | auto
    lang: str
    via: str  # yt-dlp | transcript-api

    @property
    def origin(self) -> str:
        return f"captions:{self.kind}:{self.lang}"


def pick_track(info: dict, cfg: CaptionsCfg) -> tuple[str, str] | None:
    """(kind, lang) of the best available English caption track from a yt-dlp info dict, or None."""
    manual, auto = info.get("subtitles") or {}, info.get("automatic_captions") or {}
    order = [("manual", manual), ("auto", auto)]
    if not cfg.prefer_manual:
        order.reverse()
    base = cfg.languages[0].split("-")[0]
    for kind, tracks in order:
        for lang in cfg.languages:
            if tracks.get(lang):
                return kind, lang
        for lang in tracks:  # any other variant of the base language, e.g. 'en-CA'
            if lang.split("-")[0] == base:
                return kind, lang
    return None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _TAGS.sub("", text)).strip()


def parse_vtt(path: Path, auto: bool) -> list[AsrSegment]:
    """Cues -> segments. For auto captions, drop lines repeated from the previous cue (rollup) and empty cues."""
    segs: list[AsrSegment] = []
    last_line = ""
    for cue in webvtt.read(str(path)):
        lines = [_clean(line) for line in cue.raw_text.splitlines()]
        lines = [x for x in lines if x]
        if not lines:
            continue
        if auto:
            fresh = [x for x in lines if x != last_line and (not segs or x != segs[-1].text)]
            last_line = lines[-1]
            if not fresh:
                continue
            lines = fresh
        segs.append(AsrSegment(cue.start_in_seconds, cue.end_in_seconds, " ".join(lines)))
    if auto:  # rollup cues overlap in time: end each segment where the next begins
        for a, b in zip(segs, segs[1:], strict=False):
            a.end_s = min(a.end_s, b.start_s) if b.start_s > a.start_s else a.end_s
    return segs


def fetch_ytdlp(video_id: str, info: dict, cfg: CaptionsCfg, ytdlp_opts: dict) -> CaptionResult | None:
    track = pick_track(info, cfg)
    log = logger.bind(stage="captions", video_id=video_id)
    if not track:
        log.info("no English caption track (manual: {} langs, auto: {} langs)", len(info.get("subtitles") or {}),
                 len(info.get("automatic_captions") or {}))
        return None
    kind, lang = track
    log.info("caption track chosen: {} {}", kind, lang)
    with tempfile.TemporaryDirectory() as d:
        opts = {**ytdlp_opts, "skip_download": True, "subtitleslangs": [lang], "subtitlesformat": "vtt",
                "writesubtitles": kind == "manual", "writeautomaticsub": kind == "auto",
                "outtmpl": str(Path(d) / "%(id)s.%(ext)s")}
        yt.extract(f"https://www.youtube.com/watch?v={video_id}", opts, download=True, stage="captions",
                   video_id=video_id)
        files = sorted(Path(d).glob("*.vtt"))
        if not files:
            log.warning("yt-dlp reported track {} {} but wrote no .vtt", kind, lang)
            return None
        segs = parse_vtt(files[0], auto=kind == "auto")
    log.info("parsed {} caption segments ({} chars)", len(segs), sum(len(s.text) for s in segs))
    return CaptionResult(segs, kind, lang, "yt-dlp") if segs else None


def fetch_api(video_id: str, cfg: CaptionsCfg) -> CaptionResult | None:
    """youtube-transcript-api fallback (no download, same failure modes as YouTube's caption endpoints)."""
    from youtube_transcript_api import YouTubeTranscriptApi

    log = logger.bind(stage="captions", video_id=video_id)
    try:
        tl = YouTubeTranscriptApi().list(video_id)
        langs = cfg.languages
        try:
            t = tl.find_manually_created_transcript(langs) if cfg.prefer_manual else tl.find_transcript(langs)
        except Exception:
            t = tl.find_transcript(langs)
        fetched = t.fetch()
    except Exception as e:
        log.warning("youtube-transcript-api found nothing: {!r}", e)
        return None
    segs = [AsrSegment(s.start, s.start + s.duration, _clean(s.text)) for s in fetched if _clean(s.text)]
    kind = "auto" if t.is_generated else "manual"
    log.info("youtube-transcript-api: {} {} ({} segments)", kind, t.language_code, len(segs))
    return CaptionResult(segs, kind, t.language_code, "transcript-api") if segs else None
