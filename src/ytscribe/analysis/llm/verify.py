"""Evidence check: every quote the model returns must appear in the transcript (rapidfuzz tolerates caption noise)."""

import re

from rapidfuzz import fuzz

from ytscribe.analysis.llm.schemas import TalkAnnotation

_NORM = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _NORM.sub(" ", text.lower())).strip()


def quote_found(quote: str, transcript_norm: str, threshold: float = 88) -> bool:
    q = normalize(quote)
    if len(q) < 8:
        return False
    if q in transcript_norm:
        return True
    return fuzz.partial_ratio(q, transcript_norm) >= threshold


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
