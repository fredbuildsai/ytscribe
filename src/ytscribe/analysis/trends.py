"""Per-event topic rates and changepoints. Counting is ours (lexicon.py); the step detection is `ruptures`."""

import numpy as np
import pandas as pd
import ruptures as rpt

from ytscribe.analysis.corpus import Doc
from ytscribe.analysis.lexicon import count


def topic_counts(docs: list[Doc], patterns: dict) -> pd.DataFrame:
    rows = []
    for d in docs:
        r = {"video_id": d.video_id, "event": d.event, "t": d.t, "upload_date": d.upload_date, "title": d.title,
             "words": d.words, "origin": d.origin, "slide_only": d.slide_only}
        r.update({name: count(p, d.text) for name, p in patterns.items()})
        rows.append(r)
    return pd.DataFrame(rows)


def by_event(df: pd.DataFrame, topics: list[str], min_words: int = 1000, min_talks: int = 3,
             substantive: int = 3) -> pd.DataFrame:
    """Per event: talks, words, and for each topic mentions per 10k words and % of talks mentioning it. `thin` marks
    events too small to trust (few talks / words) or mostly slide text: drawn hollow, left out of step fits."""
    g = df.groupby(["t", "event"], sort=True)
    out = g.agg(talks=("video_id", "count"), words=("words", "sum"), slide_share=("slide_only", "mean"))
    out["slide_only"] = out["slide_share"] >= 0.5
    out["thin"] = (out["talks"] < min_talks) | (out["words"] <= min_words) | out["slide_only"]
    for name in topics:
        out[f"{name} /10k"] = g[name].sum() / out["words"] * 1e4
        out[f"{name} %talks"] = g[name].apply(lambda s: (s > 0).mean() * 100)  # any mention
        out[f"{name} %talks_sub"] = g[name].apply(lambda s: (s >= substantive).mean() * 100)  # substantive: >= N mentions
        out[f"{name} n_talks"] = g[name].apply(lambda s: int((s > 0).sum()))
    return out.reset_index()


def changepoints(ev: pd.DataFrame, topics: list[str], min_size: int = 2) -> pd.DataFrame:
    """Best single step in each event-level series (ruptures Dynp, l2 cost: exact least squares). Thin events (see by_event)
    are left out. Spiky series (a partner named in one session) fit a step poorly:
    read `variance_explained` before believing a step."""
    solid = ev[~ev["thin"]].reset_index(drop=True)
    rows = []
    for name in topics:
        y = solid[f"{name} /10k"].to_numpy(float).reshape(-1, 1)
        if len(y) < 2 * min_size:
            continue
        k = rpt.Dynp(model="l2", min_size=min_size, jump=1).fit(y).predict(n_bkps=1)[0]
        before, after = y[:k].mean(), y[k:].mean()
        sse = ((y[:k] - before) ** 2).sum() + ((y[k:] - after) ** 2).sum()
        total = ((y - y.mean()) ** 2).sum()
        rows.append({"topic": name, "first_event_after_step": solid["event"][k], "before": before, "after": after,
                     "variance_explained": 1 - sse / total if total else np.nan})
    return pd.DataFrame(rows)


def top_talks(df: pd.DataFrame, topic: str, n: int = 3) -> pd.DataFrame:
    d = df[df[topic] > 0].sort_values(topic, ascending=False).head(n)
    return d[["event", "title", topic]]
