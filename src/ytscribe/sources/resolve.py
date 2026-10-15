"""URL -> list of videos, via yt-dlp flat extraction (no downloads). Also expands a YAML source spec."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from ytscribe import ytdlp_runner as yt
from ytscribe.settings import load_yaml

FLAT = {"extract_flat": "in_playlist", "skip_download": True}


@dataclass
class VideoRef:
    id: str
    title: str | None = None


@dataclass
class Resolved:
    kind: str  # channel | playlist | video
    url: str
    title: str | None
    videos: list[VideoRef] = field(default_factory=list)
    external_id: str | None = None


def classify(url: str) -> str:
    if re.search(r"youtube\.com/(@|channel/|c/|user/)", url):
        return "channel"
    if "list=" in url or "/playlist" in url:
        return "playlist"
    return "video"


def playlist_id(url: str) -> str | None:
    m = re.search(r"[?&]list=([\w-]+)", url)
    return m.group(1) if m else None


def event_from_title(title: str | None) -> tuple[int | None, str | None]:
    """'2026 Spring Allotrope Connect Workshop' -> (2026, 'Spring'). Year-only titles give (year, None)."""
    m = re.search(r"\b(20\d\d)\b(?:\s+(Spring|Fall|Summer|Winter))?", title or "")
    return (int(m.group(1)), m.group(2)) if m else (None, None)


def _entries(info: dict) -> list[VideoRef]:
    """Flatten (possibly nested) flat-extraction entries to unique video refs, keeping order."""
    out, seen = [], set()

    def walk(entries):
        for e in entries or []:
            if not e:
                continue
            if e.get("_type") == "playlist" or e.get("entries"):
                walk(e.get("entries"))
            elif e.get("id") and e["id"] not in seen and len(e["id"]) == 11:
                seen.add(e["id"])
                out.append(VideoRef(e["id"], e.get("title")))

    walk(info.get("entries"))
    return out


def resolve(url: str, **ctx) -> Resolved:
    kind = classify(url)
    if kind == "channel" and not re.search(r"/(videos|playlists|streams|shorts)/?$", url):
        url = url.rstrip("/") + "/videos"
    logger.bind(stage="resolve", **ctx).info("resolving {} {}", kind, url)
    if kind == "video":
        info = yt.extract(url, {"skip_download": True}, stage="resolve")
        return Resolved("video", url, info.get("title"), [VideoRef(info["id"], info.get("title"))])
    info = yt.extract(url, FLAT, stage="resolve")
    videos = _entries(info)
    logger.bind(stage="resolve").info("{} '{}': {} videos", kind, info.get("title"), len(videos))
    return Resolved(kind, url, info.get("title"), videos, external_id=info.get("id") or playlist_id(url))


def list_channel_playlists(channel_url: str, title_regex: str | None = None) -> list[tuple[str, str]]:
    """[(title, playlist_url)] of a channel's playlists tab, optionally filtered by a title regex."""
    url = channel_url.rstrip("/") + "/playlists"
    info = yt.extract(url, FLAT, stage="resolve")
    rx = re.compile(title_regex, re.I) if title_regex else None
    found = []
    for e in info.get("entries") or []:
        title, purl = e.get("title") or "", e.get("url") or e.get("webpage_url")
        if purl and (not rx or rx.search(title)):
            found.append((title, purl))
        elif purl:
            logger.bind(stage="resolve").debug("playlist skipped by filter: {}", title)
    logger.bind(stage="resolve").info("{} matching playlists on {}", len(found), channel_url)
    return found


def expand_source_file(path: Path) -> list[Resolved]:
    """YAML spec -> resolved playlists (and optionally channel uploads). A playlist that fails is logged and skipped."""
    spec = load_yaml(path)
    out: list[Resolved] = []
    for title, purl in list_channel_playlists(spec["channel"], spec.get("playlist_title_regex")):
        try:
            r = resolve(purl)
            r.title = r.title or title
            out.append(r)
        except (yt.VideoUnavailable, yt.YtdlpFailure) as e:
            logger.bind(stage="resolve").error("playlist '{}' ({}) could not be enumerated: {}", title, purl, e)
    if spec.get("include_channel_videos"):
        out.append(resolve(spec["channel"]))
    return out
