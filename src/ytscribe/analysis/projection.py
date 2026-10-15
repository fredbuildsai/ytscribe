"""Fill in an event whose videos are not all uploaded yet, using its published agenda.

Abstract wording predicts talk content poorly (AI: correlation -0.35 on 11 sessions), but TITLES are informative. So:
1. learn from the other events how a topic in the TITLE shifts the transcript rate (per 10k words), in the topic's own era;
2. apply that to the titles of sessions with no video; 3. validate the same recipe on the sessions that do have a video.
Session -> video matching uses rapidfuzz on titles."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from ytscribe.analysis.config import TopicsConfig
from ytscribe.analysis.corpus import Doc
from ytscribe.analysis.lexicon import count

_PREFIX = re.compile(r"^\d{4} (Spring|Fall) AC\s*[-–]?\s*", re.I)
_OPEN = re.compile(r"^group discussion", re.I)


def clean_title(t: str) -> str:
    return _PREFIX.sub("", t).strip()


def load_agenda(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def match_sessions(sessions: list[dict], docs: list[Doc], threshold: float = 60,
                   overrides: dict[str, str] | None = None) -> dict[str, str]:
    """agenda title -> video id. Explicit `overrides` (agenda title prefix -> video id) win; the rest are matched greedily
    on token_set_ratio between the agenda title and the video title. Check `unmatched_videos` in the result."""
    out, used = {}, set()
    ids = {d.video_id for d in docs}
    for prefix, vid in (overrides or {}).items():
        for s in sessions:
            if s["title"].startswith(prefix) and vid in ids and s["title"] not in out:
                out[s["title"]] = vid
                used.add(vid)
    pairs = sorted(((fuzz.token_set_ratio(s["title"], clean_title(d.title)), s["title"], d.video_id)
                    for s in sessions for d in docs), reverse=True)
    for score, title, vid in pairs:
        if score < threshold:
            break
        if title not in out and vid not in used:
            out[title] = vid
            used.add(vid)
    return out


def title_flags(title: str, patterns: dict, topics: list[str]) -> dict[str, int]:
    t = clean_title(title)
    return {n: int(count(patterns[n], t) > 0) for n in topics}


def project_event(agenda: dict, docs: list[Doc], counts: pd.DataFrame, patterns: dict, topics: list[str],
                  cfg: TopicsConfig, event: str, seed: int = 0) -> dict:
    """Returns {'projection': df, 'validation': df, 'missing': df, 'matched': dict}."""
    pc = cfg.projection
    ev_docs = [d for d in docs if d.event == event]
    sessions = agenda["sessions"]
    matched = match_sessions(sessions, ev_docs, overrides=agenda.get("matches"))
    unmatched_videos = [(d.video_id, d.title) for d in ev_docs if d.video_id not in matched.values()]
    have = counts[counts["video_id"].isin(matched.values())]
    mins = {d.video_id: d.duration_min or 0 for d in ev_docs}
    wpm = have["words"].sum() / max(sum(mins[v] for v in have["video_id"]), 1)

    pool = counts[(counts["event"] != event) & (counts["words"] > 1500) & ~counts["slide_only"]].copy()
    for n in topics:
        pool[f"rate {n}"] = pool[n] / pool["words"] * 1e4
        pool[f"flag {n}"] = pool["title"].map(lambda t, n=n: title_flags(t, patterns, [n])[n])

    def means(topic: str, rng=None):
        d = pool[pool["t"] >= pc.era_start.get(topic, 0)]
        if rng is not None:
            d = d.sample(len(d), replace=True, random_state=rng)
        yes, no = d[d[f"flag {topic}"] == 1][f"rate {topic}"], d[d[f"flag {topic}"] == 0][f"rate {topic}"]
        return (yes.mean() if len(yes) else no.mean()), no.mean()

    miss = []
    for s in sessions:
        if s["title"] in matched or _OPEN.match(s["title"]):
            continue
        m = pc.minutes_panel if s["title"].lower().startswith(("industry panel", "panel")) else pc.minutes_default
        miss.append({"title": s["title"], "minutes": m, "words": m * wpm,
                     **{f"flag {n}": v for n, v in title_flags(s["title"], patterns, topics).items()}})
    miss = pd.DataFrame(miss)

    def expected(rng=None):
        return {n: ((miss[f"flag {n}"] * means(n, rng)[0] + (1 - miss[f"flag {n}"]) * means(n, rng)[1])
                    * miss["words"] / 1e4).sum() for n in topics}

    have_words = have["words"].sum()
    tot_words = have_words + (miss["words"].sum() if len(miss) else 0)
    point = expected()
    boots = [expected(np.random.RandomState(seed + b)) for b in range(pc.bootstrap)] if len(miss) else []
    rows = []
    for n in topics:
        proj = (have[n].sum() + point[n]) / tot_words * 1e4 if len(miss) else have[n].sum() / have_words * 1e4
        bs = [(have[n].sum() + b[n]) / tot_words * 1e4 for b in boots] or [proj]
        rows.append({"topic": n, "transcribed_only": have[n].sum() / have_words * 1e4, "projected": proj,
                     "p10": float(np.quantile(bs, 0.1)), "p90": float(np.quantile(bs, 0.9))})
    # validation: pretend the matched sessions have no video
    vrows = []
    by_vid = {v: t for t, v in matched.items()}
    for n in topics:
        yes, no = means(n)
        w = have["words"].to_numpy(float)
        flags = np.array([title_flags(by_vid[v], patterns, [n])[n] for v in have["video_id"]])
        pred = ((flags * yes + (1 - flags) * no) * w / 1e4).sum() / w.sum() * 1e4
        vrows.append({"topic": n, "predicted": pred, "actual": have[n].sum() / w.sum() * 1e4})
    return {"projection": pd.DataFrame(rows), "validation": pd.DataFrame(vrows), "missing": miss, "matched": matched,
            "unmatched_videos": unmatched_videos}
