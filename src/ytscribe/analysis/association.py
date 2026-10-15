"""Which topics are interconnected? Three views, because each answers a different question and has a different trap.

1. Talk level, pooled: Spearman correlation of topic rates (per 10k words, so a long talk is not favoured) across talks.
   Trap: TIME. Two topics that both rise (or fall) over the years correlate without ever meeting in a talk.
2. Talk level, within event: the same after subtracting each event's mean rate, which removes the time trend and leaves
   "in talks of the same workshop, do talks heavy in A also run heavy in B?".
3. Passage level: do A and B appear in the same ~150-word passage more often than chance? The tightest link (two topics
   in one passage are being discussed together). Reported as phi, odds ratio and lift; the odds ratio is pooled with
   Fisher's exact test AND stratified by event (Cochran-Mantel-Haenszel), so the time trend cannot create it.
p-values are corrected for the number of pairs (Benjamini-Hochberg). Slide-only videos are excluded (OCR repeats text).
Correlation shows co-occurrence, not that one topic causes or uses the other."""

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

from ytscribe.analysis.corpus import Doc
from ytscribe.analysis.discover import passages
from ytscribe.analysis.lexicon import count


def usable_topics(counts: pd.DataFrame, pres: pd.DataFrame, topics: list[str], min_talks: int, min_passages: int):
    """Topics with enough signal to correlate; the rest are returned with their support so the drop is visible."""
    keep, dropped = [], []
    for t in topics:
        n_talks, n_pass = int((counts[t] > 0).sum()), int(pres[t].sum())
        if n_talks >= min_talks and n_pass >= min_passages:
            keep.append(t)
        else:
            dropped.append((t, n_talks, n_pass))
    return keep, dropped


def talk_rates(counts: pd.DataFrame, topics: list[str], min_event_talks: int = 3) -> pd.DataFrame:
    """Speech talks only; rates per 10k words; events with fewer than `min_event_talks` talks are dropped (nothing to demean)."""
    d = counts[~counts["slide_only"]].copy()
    for t in topics:
        d[t] = d[t] / d["words"] * 1e4
    sizes = d.groupby("event")["video_id"].transform("count")
    return d[sizes >= min_event_talks]


def spearman_matrix(rates: pd.DataFrame, topics: list[str], within_event: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = rates[topics].copy()
    if within_event:
        x = x - x.groupby(rates["event"]).transform("mean")
    rho = pd.DataFrame(np.nan, index=topics, columns=topics)
    pv = rho.copy()
    for a, b in itertools.combinations(topics, 2):
        r, p = stats.spearmanr(x[a], x[b])
        rho.loc[a, b] = rho.loc[b, a] = r
        pv.loc[a, b] = pv.loc[b, a] = p
    return rho, pv


def bootstrap_ci(rates: pd.DataFrame, a: str, b: str, within_event: bool, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of Spearman rho by resampling talks (inside each event when `within_event`)."""
    rng = np.random.RandomState(seed)
    x = rates[[a, b]].to_numpy()
    ev = rates["event"].to_numpy()
    idx_by_event = [np.where(ev == e)[0] for e in np.unique(ev)]
    out = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_by_event]) if within_event else rng.choice(len(x), len(x))
        s = x[idx]
        if within_event:
            e_s = ev[idx]
            s = s - pd.DataFrame(s).groupby(e_s).transform("mean").to_numpy()
        r = stats.spearmanr(s[:, 0], s[:, 1])[0]
        if not np.isnan(r):
            out.append(r)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def passage_presence(docs: list[Doc], patterns: dict, topics: list[str], words: int = 150) -> pd.DataFrame:
    rows = []
    for ps in passages(docs, words):
        rows.append({"video_id": ps.video_id, "event": ps.event, **{t: int(count(patterns[t], ps.text) > 0) for t in topics}})
    return pd.DataFrame(rows)


def pair_stats(pres: pd.DataFrame, a: str, b: str) -> dict:
    A, B = pres[a].to_numpy(bool), pres[b].to_numpy(bool)
    n11, n10, n01, n00 = int((A & B).sum()), int((A & ~B).sum()), int((~A & B).sum()), int((~A & ~B).sum())
    n = n11 + n10 + n01 + n00
    pa, pb = (n11 + n10) / n, (n11 + n01) / n
    phi = (n11 * n00 - n10 * n01) / np.sqrt(max((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00), 1))
    odds, p_fisher = stats.fisher_exact([[n11, n10], [n01, n00]])
    lift = (n11 / n) / (pa * pb) if pa * pb else np.nan
    # event-stratified odds ratio (Mantel-Haenszel): the shared time trend cannot produce it
    tables = []
    for _, g in pres.groupby("event"):
        ga, gb = g[a].to_numpy(bool), g[b].to_numpy(bool)
        t = np.array([[(ga & gb).sum(), (ga & ~gb).sum()], [(~ga & gb).sum(), (~ga & ~gb).sum()]], dtype=float) + 0.0
        if t.sum() and (t.sum(axis=0) > 0).all() and (t.sum(axis=1) > 0).all():
            tables.append(t)
    mh_or = mh_lo = mh_hi = mh_p = np.nan
    if len(tables) >= 2:
        try:
            st = StratifiedTable(tables)
            mh_or = st.oddsratio_pooled
            mh_lo, mh_hi = st.oddsratio_pooled_confint()
            mh_p = st.test_null_odds(correction=True).pvalue
        except Exception:
            pass
    expected = (n11 + n10) * (n11 + n01) / n if n else 0.0  # passages with both if A and B were independent
    return {"a": a, "b": b, "expected_both": expected, "passages_both": n11, "passages_a": n11 + n10, "passages_b": n11 + n01, "n": n,
            "phi": phi, "lift": lift, "odds_pooled": odds, "p_fisher": p_fisher,
            "odds_by_event": mh_or, "or_lo": mh_lo, "or_hi": mh_hi, "p_by_event": mh_p}


def passage_table(pres: pd.DataFrame, topics: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([pair_stats(pres, a, b) for a, b in itertools.combinations(topics, 2)])
    for col in ("p_fisher", "p_by_event"):
        ok = df[col].notna()
        q = np.full(len(df), np.nan)
        if ok.any():
            q[ok.to_numpy()] = stats.false_discovery_control(df.loc[ok, col].to_numpy(), method="bh")
        df["q_" + col[2:]] = q
    return df


def fdr_matrix(pv: pd.DataFrame) -> pd.DataFrame:
    topics = list(pv.index)
    pairs = list(itertools.combinations(topics, 2))
    q = stats.false_discovery_control([pv.loc[a, b] for a, b in pairs], method="bh")
    out = pd.DataFrame(np.nan, index=topics, columns=topics)
    for (a, b), v in zip(pairs, q, strict=False):
        out.loc[a, b] = out.loc[b, a] = v
    return out


def matrix_from_table(tab: pd.DataFrame, topics: list[str], col: str) -> pd.DataFrame:
    m = pd.DataFrame(np.nan, index=topics, columns=topics)
    for _, r in tab.iterrows():
        m.loc[r["a"], r["b"]] = m.loc[r["b"], r["a"]] = r[col]
    return m
