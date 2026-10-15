"""loguru sinks: rich-ish console, daily DEBUG file, structured JSONL. Plus a bridge for yt-dlp's own logger.

Every record can carry `video_id`, `stage` and `attempt` via `logger.bind(...)`.
"""

import sys
from datetime import date
from pathlib import Path

from loguru import logger

_FMT = ("<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> "
        "<cyan>{extra[stage]}</cyan>:<magenta>{extra[video_id]}</magenta> {message}")
_configured = False


def setup_logging(logs_dir: Path, level: str = "INFO") -> Path:
    """Idempotent. Returns the path of today's text log."""
    global _configured
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"ytscribe-{date.today():%Y%m%d}.log"
    if _configured:
        return log_file
    logger.remove()
    logger.configure(extra={"stage": "-", "video_id": "-", "attempt": 0})
    logger.add(sys.stderr, level=level.upper(), format=_FMT, colorize=True)
    logger.add(log_file, level="DEBUG", rotation="20 MB", retention="30 days", enqueue=True,
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {extra[stage]}:{extra[video_id]}:a{extra[attempt]} | "
                      "{message}")
    logger.add(logs_dir / "ytscribe.jsonl", level="DEBUG", serialize=True, rotation="50 MB", retention="30 days",
               enqueue=True)
    _configured = True
    logger.debug("logging ready: file={} level={}", log_file, level)
    return log_file


class YtdlpLogger:
    """Passed to yt-dlp as `logger`: all its output lands in our sinks (verbose chatter at DEBUG)."""

    def __init__(self, **ctx):
        self._log = logger.bind(**ctx)

    def debug(self, msg: str) -> None:
        self._log.debug("yt-dlp: {}", msg)

    def info(self, msg: str) -> None:
        self._log.debug("yt-dlp: {}", msg)

    def warning(self, msg: str) -> None:
        self._log.warning("yt-dlp: {}", msg)

    def error(self, msg: str) -> None:
        self._log.error("yt-dlp: {}", msg)
