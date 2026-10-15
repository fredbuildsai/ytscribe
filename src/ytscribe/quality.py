"""Cheap sanity checks on a finished transcript. Used when processing and again at export time, so transcripts
produced before a check existed get flagged too."""

from collections.abc import Sequence
from typing import Protocol


class _HasText(Protocol):
    text: str


def quality_warnings(segs: Sequence[_HasText], duration_s: float | None) -> list[str]:
    """Cheap sanity checks that catch ASR hallucination (e.g. 'Thank you.' x N on silence/music) and sparse output."""
    out, texts = [], [x.text.strip().lower() for x in segs]
    words = sum(len(t.split()) for t in texts)
    if len(texts) >= 3 and len(set(texts)) / len(texts) < 0.4:
        out.append(f"repetitive: {len(set(texts))} distinct lines in {len(texts)} segments")
    if duration_s and duration_s >= 20 and words / (duration_s / 60) < 15:
        out.append(f"sparse: {words} words in {duration_s / 60:.1f} min (<15 wpm), maybe silence or music only")
    return out
