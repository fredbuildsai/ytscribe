"""Deterministic pre-labels handed to the LLM alongside the transcript: lexicon counts, the talk's dominant unsupervised
topics, and tone around anchor topics. They anchor the model (and make disagreements visible); they are hints, not truth."""

import pandas as pd

from ytscribe.analysis.corpus import Doc
from ytscribe.analysis.lexicon import count


def lexicon_counts(doc: Doc, patterns: dict, topics: list[str]) -> dict[str, int]:
    return {n: c for n in topics if (c := count(patterns[n], doc.text))}


def bertopic_profile(passage_topics: pd.DataFrame | None, video_id: str, top: int = 3) -> list[tuple[int, float]]:
    """[(topic id, share of the talk's passages)] for the talk's dominant BERTopic topics (outliers excluded)."""
    if passage_topics is None:
        return []
    sub = passage_topics[passage_topics.video_id == video_id]
    if sub.empty:
        return []
    vc = sub[sub.topic >= 0]["topic"].value_counts() / len(sub)
    return [(int(t), float(s)) for t, s in vc.head(top).items()]


def render(doc: Doc, counts: dict[str, int], profile: list[tuple[int, float]], chapters: list[str] | None = None) -> str:
    parts = [f"Event: {doc.event}; uploaded {doc.upload_date}; length {doc.duration_min} min; {doc.words} words."]
    parts.append("Keyword counts in the transcript (deterministic): " +
                 (", ".join(f"{k}={v}" for k, v in counts.items()) or "none of the tracked terms"))
    if profile:
        parts.append("Dominant unsupervised topics (BERTopic, share of passages): " +
                     ", ".join(f"T{t:02d} {s:.0%}" for t, s in profile))
    if chapters:
        parts.append("Chapters: " + "; ".join(chapters[:12]))
    if doc.description:
        parts.append("Video description: " + " ".join(doc.description.split())[:600])
    return "\n".join(parts)
