"""Video-only download (low resolution) for the slide extraction path."""

from pathlib import Path

from loguru import logger

from ytscribe import ytdlp_runner as yt


def download_video(video_id: str, out_dir: Path, ytdlp_opts: dict, max_height: int = 720) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logger.bind(stage="video", video_id=video_id)
    fmt = (f"bestvideo[height<={max_height}][ext=mp4]/bestvideo[height<={max_height}]/best[height<={max_height}]")
    opts = {**ytdlp_opts, "format": fmt, "continuedl": True, "noprogress": True,
            "outtmpl": str(out_dir / f"{video_id}.%(ext)s")}
    log.info("downloading video (<= {}p, no audio) to {}", max_height, out_dir)
    info = yt.extract(f"https://www.youtube.com/watch?v={video_id}", opts, download=True, stage="video",
                      video_id=video_id)
    path = Path((info.get("requested_downloads") or [{}])[0].get("filepath") or "")
    if not path.is_file():
        raise yt.YtdlpFailure(f"video download produced no file for {video_id}")
    log.info("video ready: {} ({:.1f} MB)", path.name, path.stat().st_size / 1e6)
    return path
