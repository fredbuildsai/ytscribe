"""The only place that calls yt-dlp. YouTube breaks extractors often, so on an extractor-class failure this upgrades
yt-dlp in the running venv, reloads it and retries once (at most one upgrade per process).

Permanent failures (private, removed, members-only, age-gated...) are raised as `VideoUnavailable` without any upgrade.
"""

import importlib
import importlib.metadata
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import Any, TypeVar

from loguru import logger

from ytscribe.logging_setup import YtdlpLogger

T = TypeVar("T")

PERMANENT_MARKERS = (
    "private video", "video unavailable", "this video has been removed", "has been terminated", "members-only",
    "members only", "confirm your age", "age-restricted", "not available in your country", "copyright",
    "this video is no longer available", "account associated with this video", "live event will begin",
)


class VideoUnavailable(Exception):
    """The video itself cannot be fetched; upgrading yt-dlp will not help."""


class YtdlpFailure(Exception):
    """yt-dlp failed even after the self-heal attempt (if enabled)."""


_state = {"upgrade_attempted": False, "auto_update": True}
_lock = threading.Lock()  # prefetch runs yt-dlp in a second thread; the upgrade+reload must not interleave


def configure(auto_update: bool = True) -> None:
    _state["auto_update"] = auto_update


def version() -> str:
    try:
        return importlib.metadata.version("yt-dlp")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def is_permanent(message: str) -> bool:
    m = message.lower()
    return any(k in m for k in PERMANENT_MARKERS)


def _upgrade() -> bool:
    """pip/uv install -U yt-dlp into the interpreter's venv. True if the installed version changed."""
    before = version()
    cmds = [["uv", "pip", "install", "-U", "--python", sys.executable, "yt-dlp"],
            [sys.executable, "-m", "pip", "install", "-U", "yt-dlp"]]
    for cmd in cmds:
        logger.info("upgrading yt-dlp (current {}): {}", before, " ".join(cmd))
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.warning("upgrade command failed to run: {!r}", e)
            continue
        logger.debug("upgrade stdout: {}\nstderr: {}", r.stdout.strip(), r.stderr.strip())
        if r.returncode == 0:
            break
        logger.warning("upgrade exited {}", r.returncode)
    else:
        return False
    importlib.invalidate_caches()
    for name in [n for n in sys.modules if n == "yt_dlp" or n.startswith("yt_dlp.")]:
        del sys.modules[name]
    importlib.import_module("yt_dlp")
    after = version()
    logger.info("yt-dlp {} -> {}", before, after)
    return after != before


def _call(fn: Callable[[Any], T], opts: dict, ctx: dict) -> T:
    import yt_dlp

    full = {"quiet": True, "no_warnings": False, "logger": YtdlpLogger(**ctx), **opts}
    with yt_dlp.YoutubeDL(full) as ydl:
        return fn(ydl)


def run(fn: Callable[[Any], T], opts: dict | None = None, **ctx) -> T:
    """Run `fn(ydl)` with a configured YoutubeDL. `ctx` (video_id, stage...) is bound to the log lines."""
    log = logger.bind(**ctx)
    opts = opts or {}
    try:
        return _call(fn, opts, ctx)
    except Exception as e:  # yt_dlp.utils.DownloadError / ExtractorError, plus anything the extractor raises
        msg = str(e)
        if is_permanent(msg):
            log.warning("video unavailable: {}", msg)
            raise VideoUnavailable(msg) from e
        log.warning("yt-dlp {} failed: {}", version(), msg, exc_info=True)
        with _lock:
            do_upgrade = _state["auto_update"] and not _state["upgrade_attempted"]
            if do_upgrade:
                _state["upgrade_attempted"] = True
                changed = _upgrade()
        if not do_upgrade:
            raise YtdlpFailure(msg) from e
        if not changed:
            log.warning("no newer yt-dlp available (or upgrade failed); retrying once anyway")
        try:
            result = _call(fn, opts, ctx)
        except Exception as e2:
            log.error("retry after yt-dlp upgrade failed: {}", e2)
            if is_permanent(str(e2)):
                raise VideoUnavailable(str(e2)) from e2
            raise YtdlpFailure(str(e2)) from e2
        log.info("retry after yt-dlp upgrade succeeded (version {})", version())
        return result


def extract(url: str, opts: dict | None = None, download: bool = False, **ctx) -> dict:
    return run(lambda ydl: ydl.extract_info(url, download=download), opts, **ctx)
