"""Evidence check: every quote the model returns must appear in the transcript (rapidfuzz tolerates caption noise)."""

import re

from rapidfuzz import fuzz

from ytscribe.analysis.llm.schemas import TalkAnnotation

_NORM = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _NORM.sub(" ", text.lower())).strip()


# Calibrated on real data (2026-10-15): true quotes the model had lightly compressed scored 76-95; unrelated talks' text 47-50;
# plausible but invented same-topic quotes 50-53. 65 sits in the wide gap, so faithful-but-trimmed quotes pass and inventions fail.
QUOTE_THRESHOLD = 65
_ELLIPSIS = re.compile(r"\.\.\.|…")


def quote_found(quote: str, transcript_norm: str, threshold: float = QUOTE_THRESHOLD) -> bool:
    """True if the quote is in the transcript. A quote with '...' is split and every part (8+ chars) must be found."""
    parts = [normalize(p) for p in _ELLIPSIS.split(quote)]
    parts = [p for p in parts if len(p) >= 8]
    if not parts:
        return False
    return all(p in transcript_norm or fuzz.partial_ratio(p, transcript_norm) >= threshold for p in parts)


def verify(ann: TalkAnnotation, transcript: str) -> tuple[dict, float]:
    """Returns (annotation dict with unverified quotes dropped and `unsupported` flags, verified ratio)."""
    tn = normalize(transcript)
    total = ok = 0
    data = ann.model_dump()
    for key in ("stances", "partners"):
        for item in data[key]:
            kept = []
            for q in item["evidence"]:
                total += 1
                if quote_found(q, tn):
                    ok += 1
                    kept.append(q)
            item["evidence"] = kept
            item["unsupported"] = not kept
    return data, (ok / total if total else 1.0)
