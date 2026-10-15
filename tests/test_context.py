import pandas as pd
import pytest

from ytscribe.analysis import context as cx


def monthly(values, start="2023-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"), dtype=float)


def test_level_index_is_percent_of_the_current_level():
    s = monthly([10, 20, 40, 80, 100, 100, 100, 100, 100, 100])
    idx = cx.level_index(s, last_n=6)
    assert idx.iloc[-1] == pytest.approx(100) and idx.iloc[0] == pytest.approx(10)


def test_first_reach_finds_the_crossing_month_and_none_when_never():
    idx = cx.level_index(monthly([5, 12, 30, 60, 100, 100, 100, 100, 100, 100]), last_n=6)
    assert cx.first_reach(idx, 10) == pd.Timestamp("2023-02-01")
    assert cx.first_reach(idx, 50) == pd.Timestamp("2023-04-01")
    assert cx.first_reach(idx, 250) is None
    assert cx.first_reach(idx, 50, smooth=3) >= cx.first_reach(idx, 50)   # smoothing never makes a crossing earlier


def test_months_between_and_missing():
    assert cx.months_between(pd.Timestamp("2025-10-01"), pd.Timestamp("2023-11-01")) == 23
    assert cx.months_between(None, pd.Timestamp("2023-11-01")) is None


def test_every_milestone_has_a_date_label_kind_and_source():
    ctx = cx.load_context("configs/topics/context_allotrope.yaml")
    for m in ctx["milestones"]:
        assert m["source"].startswith("https://") and m["kind"] in ("general", "science", "lab") and len(m["date"]) == 10
    dates = [m["date"] for m in ctx["milestones"]]
    assert dates == sorted(dates)
    assert set(ctx["workshops"].values()) and all(v[5:] in ("05", "10") for v in ctx["workshops"].values())  # May / October rule
