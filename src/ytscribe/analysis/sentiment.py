"""Tone around topic mentions, without an LLM.

The topic lexicons hold only neutral terms, so the words around a mention are what a sentiment model judges. Talks are
promotional and positive by nature, so the useful number is RELATIVE: a topic's tone minus the tone of the same event's
ordinary text. Baseline scorer = VADER (lexicon); a trained classifier can be added behind the same `scorer` callable."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from ytscribe.analysis.corpus import Doc
from ytscribe.analysis.lexicon import windows


def vader_scorer() -> Callable[[str], float]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    an = SentimentIntensityAnalyzer()
    return lambda text: an.polarity_scores(text)["compound"]


def _chunks(text: str, size: int):
    toks = text.split()
    for i in range(0, len(toks) - size + 1, size):
        yield " ".join(toks[i:i + size])


def topic_sentiment(docs: list[Doc], patterns: dict, topics: list[str], window_words: int = 30,
                    scorer: Callable[[str], float] | None = None, baseline_cap: int = 400) -> pd.DataFrame:
    """Per (event, topic): number of windows, mean compound score, share positive / negative, and `relative` = mean
    minus the event baseline (mean over evenly spaced neutral chunks of the same event's text)."""
    scorer = scorer or vader_scorer()
    rng = np.random.RandomState(0)
    size = 2 * window_words + 1
    rows = []
    for ev in dict.fromkeys(d.event for d in docs):
        ds = [d for d in docs if d.event == ev and not d.slide_only]
        if not ds:
            continue
        chunks = [c for d in ds for c in _chunks(d.text, size)]
        if len(chunks) > baseline_cap:
            chunks = [chunks[i] for i in rng.choice(len(chunks), baseline_cap, replace=False)]
        base = float(np.mean([scorer(c) for c in chunks])) if chunks else float("nan")
        for name in topics:
            ws = [w for d in ds for w in windows(patterns[name], d.text, window_words)]
            sc = np.array([scorer(w) for w in ws]) if ws else np.array([])
            rows.append({"event": ev, "topic": name, "windows": len(ws), "mean": float(sc.mean()) if len(sc) else np.nan,
                         "pct_pos": float((sc > 0.05).mean() * 100) if len(sc) else np.nan,
                         "pct_neg": float((sc < -0.05).mean() * 100) if len(sc) else np.nan, "baseline": base,
                         "relative": float(sc.mean() - base) if len(sc) else np.nan})
    return pd.DataFrame(rows)
