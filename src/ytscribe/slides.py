"""Slide-only videos: pull the distinct frames out of the video and OCR them, so a video with no speech still yields text.

Libraries do the heavy lifting: ffmpeg samples frames, `imagehash` (dHash) measures how different two frames are, and
Apple Vision (through `ocrmac`) reads the text locally. Only the change-detection policy below is ours:

1. Sample `fps` frames per second and hash each one.
2. Consecutive frames with hamming distance <= same_thresh are the *same slide* (compression / cursor noise).
3. A run of same-slide frames shorter than `min_run_s` is a fade or transition and is dropped.
4. The last frame of each surviving run is kept (an animation build is then shown complete).
5. Consecutive kept slides closer than dup_thresh are builds of one slide: keep the later one.
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ytscribe.settings import SlidesCfg


@dataclass
class Slide:
    idx: int
    start_s: float
    end_s: float
    path: Path
    text: str = ""
    conf: float | None = None


# --- change detection: pure functions over a list of hashes (anything supporting `a - b` -> int) ------------------

def stable_run_ends(hashes: list, same_thresh: int, min_run: int) -> list[int]:
    """Indices of the last frame of every run of 'same' frames that lasts at least `min_run` frames."""
    runs, start = [], 0
    for i in range(1, len(hashes) + 1):
        if i == len(hashes) or (hashes[i] - hashes[i - 1]) > same_thresh:
            if i - start >= min_run:
                runs.append(i - 1)
            start = i
    return runs


def merge_near_duplicates(kept: list[int], hashes: list, dup_thresh: int) -> list[int]:
    """Of consecutive kept frames that are near-identical (animation builds), keep only the later one."""
    out: list[int] = []
    for i in kept:
        if out and (hashes[i] - hashes[out[-1]]) <= dup_thresh:
            out[-1] = i
        else:
            out.append(i)
    return out


def select_slide_frames(hashes: list, cfg: SlidesCfg) -> list[int]:
    min_run = max(1, round(cfg.min_run_s * cfg.fps))
    return merge_near_duplicates(stable_run_ends(hashes, cfg.same_thresh, min_run), hashes, cfg.dup_thresh)


# --- ffmpeg / hashing / OCR --------------------------------------------------------------------------------------

def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        logger.bind(stage="slides").error("{} failed: {}", cmd[0], r.stderr[-500:])
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr.strip()[-300:]}")


def sample_hashes(video: Path, cfg: SlidesCfg) -> list:
    """One dHash per sampled frame (frame k is at t = k / fps)."""
    import imagehash
    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        _run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf", f"fps={cfg.fps},scale=640:-2",
              "-q:v", "4", f"{d}/f%06d.jpg"])
        return [imagehash.dhash(Image.open(f), hash_size=cfg.hash_size) for f in sorted(Path(d).glob("f*.jpg"))]


def save_frame(video: Path, t_s: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t_s:.2f}", "-i", str(video), "-frames:v", "1",
          "-q:v", "2", str(out)])


def ocr_image(path: Path, min_conf: float) -> tuple[str, float | None]:
    """Text of one frame in rough reading order (top to bottom, then left to right) and its mean confidence."""
    from ocrmac import ocrmac

    boxes = ocrmac.OCR(str(path), recognition_level="accurate", language_preference=["en-US"]).recognize()
    boxes = [b for b in boxes if b[1] >= min_conf]
    boxes.sort(key=lambda b: (-round(b[2][1] + b[2][3], 2), b[2][0]))  # Vision bbox origin is bottom-left
    text = "\n".join(b[0].strip() for b in boxes if b[0].strip())
    return text, (sum(b[1] for b in boxes) / len(boxes) if boxes else None)


def extract_slides(video: Path, video_id: str, duration_s: float | None, frames_dir: Path, cfg: SlidesCfg) -> list[Slide]:
    log = logger.bind(stage="slides", video_id=video_id)
    hashes = sample_hashes(video, cfg)
    if not hashes:
        raise RuntimeError("no frames could be sampled from the video")
    total = duration_s or len(hashes) / cfg.fps
    keep = select_slide_frames(hashes, cfg)
    log.info("sampled {} frames at {} fps -> {} distinct slides", len(hashes), cfg.fps, len(keep))
    slides: list[Slide] = []
    for n, k in enumerate(keep):
        t = k / cfg.fps
        end = keep[n + 1] / cfg.fps if n + 1 < len(keep) else total
        path = frames_dir / f"{n + 1:03d}_t{int(t):05d}.jpg"
        save_frame(video, t, path)
        text, conf = ocr_image(path, cfg.ocr_min_conf)
        slides.append(Slide(n + 1, t, end, path, text, conf))
        log.debug("slide {} @ {:.0f}s: {} chars, conf {}", n + 1, t, len(text), None if conf is None else round(conf, 2))
    log.info("OCR done: {} slides, {} chars total, {} with no text", len(slides), sum(len(s.text) for s in slides),
             sum(1 for s in slides if not s.text))
    return slides
