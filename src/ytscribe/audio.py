"""Audio-only download through yt-dlp (resumable) for the ASR path."""

from pathlib import Path

from loguru import logger

from ytscribe import ytdlp_runner as yt


def download_audio(video_id: str, out_dir: Path, ytdlp_opts: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logger.bind(stage="audio", video_id=video_id)
    opts = {**ytdlp_opts, "format": "bestaudio[ext=m4a]/bestaudio/best", "continuedl": True, "noprogress": True,
            "outtmpl": str(out_dir / f"{video_id}.%(ext)s")}
    log.info("downloading audio to {}", out_dir)
    info = yt.extract(f"https://www.youtube.com/watch?v={video_id}", opts, download=True, stage="audio",
                      video_id=video_id)
    path = Path((info.get("requested_downloads") or [{}])[0].get("filepath") or "")
    if not path.is_file():
        raise yt.YtdlpFailure(f"audio download produced no file for {video_id}")
    log.info("audio ready: {} ({:.1f} MB)", path.name, path.stat().st_size / 1e6)
    return path
