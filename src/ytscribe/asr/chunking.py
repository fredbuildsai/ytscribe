"""Split long audio into pieces with ffmpeg, preferably at silences.

Needed because mlx-whisper builds the log-mel spectrogram of the WHOLE file at once: a 3 h session exhausts memory on a
16 GB machine (verified 2026-10-09: swap full, no progress after 30 min), while 10-minute pieces use ~1.5 GB.
Cloud ASR also has per-request duration limits.
"""

import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger


def duration_s(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def silence_midpoints(path: str, noise: str = "-35dB", min_s: float = 0.4) -> list[float]:
    err = subprocess.run(["ffmpeg", "-nostats", "-i", path, "-vn", "-af", f"silencedetect=noise={noise}:d={min_s}", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    return [(a + b) / 2 for a, b in zip(starts, ends, strict=False)]


def plan_cuts(total: float, silences: list[float], target_s: float = 600.0, search_s: float = 90.0) -> list[float]:
    """Boundaries [0, c1, ..., total]; each cut is the silence nearest the target within +-search_s, else the target."""
    cuts, t = [0.0], 0.0
    while total - t > target_s + search_s:
        goal = t + target_s
        near = [s for s in silences if abs(s - goal) <= search_s]
        t = min(near, key=lambda s: abs(s - goal)) if near else goal
        cuts.append(t)
    cuts.append(total)
    return cuts


@contextmanager
def chunks(path: str, chunk_s: float = 600.0) -> Iterator[list[tuple[float, Path]]]:
    """Yield [(offset_s, file)] of mono 16 kHz FLAC pieces cut at silences; files are deleted on exit."""
    total = duration_s(path)
    cuts = plan_cuts(total, silence_midpoints(path), chunk_s, search_s=min(90.0, chunk_s * 0.15))
    logger.bind(stage="asr").info("audio {:.0f}s cut into {} pieces of ~{:.0f}s", total, len(cuts) - 1, chunk_s)
    with TemporaryDirectory() as d:
        out = []
        for i, (a, b) in enumerate(zip(cuts, cuts[1:], strict=False)):
            f = Path(d) / f"{i:04d}.flac"
            r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(a), "-t", str(b - a), "-i", path,
                                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(f)],
                               capture_output=True, text=True)
            if r.returncode:
                logger.bind(stage="asr").error("ffmpeg failed on piece {} ({:.0f}-{:.0f}s): {}", i, a, b, r.stderr)
                raise RuntimeError(f"ffmpeg failed cutting piece {i}: {r.stderr.strip()[-300:]}")
            out.append((a, f))
        yield out
