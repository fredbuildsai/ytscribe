import numpy as np
import pandas as pd
import pytest

from ytscribe.analysis import association as assoc


def _pres(rows):
    return pd.DataFrame(rows, columns=["event", "A", "B"])


def test_independent_topics_have_odds_near_one_and_linked_ones_do_not():
    rng = np.random.RandomState(0)
    n = 4000
    a = rng.rand(n) < 0.2
    indep = pd.DataFrame({"event": ["e1", "e2"] * (n // 2), "A": a, "B": rng.rand(n) < 0.2}).astype(
        {"A": int, "B": int})
    st = assoc.pair_stats(indep, "A", "B")
    assert 0.7 < st["odds_pooled"] < 1.4 and st["p_fisher"] > 0.01 and abs(st["phi"]) < 0.05
    linked = pd.DataFrame({"event": ["e1", "e2"] * (n // 2), "A": a, "B": np.where(rng.rand(n) < 0.7, a, rng.rand(n) < 0.2)}
                          ).astype({"A": int, "B": int})
    st2 = assoc.pair_stats(linked, "A", "B")
    assert st2["odds_by_event"] > 5 and st2["p_by_event"] < 1e-6 and st2["lift"] > 2


def test_a_shared_time_trend_is_not_mistaken_for_a_link():
    """Both topics are common in event 2 and rare in event 1, but independent inside each event: the pooled odds
    ratio is inflated, the event-stratified one is not."""
    rng = np.random.RandomState(1)
    rows = []
    for ev, p in (("e1", 0.05), ("e2", 0.6)):
        for _ in range(1500):
            rows.append((ev, int(rng.rand() < p), int(rng.rand() < p)))
    st = assoc.pair_stats(_pres(rows), "A", "B")
    assert st["odds_pooled"] > 2.5            # naive pooling sees a strong link
    assert 0.7 < st["odds_by_event"] < 1.4    # stratifying removes it


def test_within_event_correlation_removes_the_trend():
    rng = np.random.RandomState(2)
    rows = []
    for ev, level in (("e1", 1.0), ("e2", 10.0), ("e3", 20.0)):
        for i in range(20):
            rows.append({"event": ev, "video_id": f"{ev}{i}", "words": 5000, "slide_only": False,
                         "A": rng.poisson(level), "B": rng.poisson(level)})
    counts = pd.DataFrame(rows)
    rates = assoc.talk_rates(counts, ["A", "B"])
    pooled, _ = assoc.spearman_matrix(rates, ["A", "B"], within_event=False)
    within, _ = assoc.spearman_matrix(rates, ["A", "B"], within_event=True)
    assert pooled.loc["A", "B"] > 0.6 and abs(within.loc["A", "B"]) < 0.3


def test_expected_overlap_and_usable_topics():
    pres = pd.DataFrame({"event": ["e"] * 100, "A": [1] * 10 + [0] * 90, "B": [1] * 5 + [0] * 95})
    assert assoc.pair_stats(pres, "A", "B")["expected_both"] == pytest.approx(0.5)
    counts = pd.DataFrame({"A": [1] * 9 + [0] * 3, "B": [1, 0] * 6, "video_id": range(12)})
    pres2 = pd.DataFrame({"A": [1] * 20, "B": [1] * 20})
    keep, dropped = assoc.usable_topics(counts, pres2, ["A", "B"], min_talks=8, min_passages=15)
    assert keep == ["A"] and dropped == [("B", 6, 20)]


def test_fdr_matrix_is_symmetric_and_never_smaller_than_p():
    pv = pd.DataFrame([[np.nan, 0.01, 0.04], [0.01, np.nan, 0.5], [0.04, 0.5, np.nan]], index=list("abc"), columns=list("abc"))
    q = assoc.fdr_matrix(pv)
    assert q.loc["a", "b"] == q.loc["b", "a"] and q.loc["a", "b"] >= 0.01 and q.loc["b", "c"] >= 0.5
