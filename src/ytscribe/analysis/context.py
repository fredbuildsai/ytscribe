"""Put the Allotrope AI trend next to the outside world: public attention (Wikipedia pageviews), scientific adoption (PubMed
papers mentioning LLMs) and dated product / science milestones with sources. Answers 'was Allotrope early, on time or late?'
by measuring WHEN each curve reaches given fractions of its current level, instead of eyeballing."""

import json
import subprocess
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import yaml

UA = "ytscribe-research/0.1 (public-interest analysis)"


def load_context(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _curl(url: str) -> str:
    return subprocess.run(["curl", "-s", "-m", "40", "-A", UA, url], capture_output=True, text=True).stdout


def fetch_wikipedia(articles: list[str], cache: Path, start="20221101", end="20260930") -> pd.DataFrame:
    """Monthly desktop+mobile user pageviews per article (Wikimedia REST API), cached as JSON."""
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for a in articles:
        f = cache / f"wp_{a}.json"
        if not f.exists():
            f.write_text(_curl(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{a}/monthly/{start}/{end}"))
        items = json.loads(f.read_text()).get("items", [])
        out[a] = pd.Series({pd.Timestamp(i["timestamp"][:4] + "-" + i["timestamp"][4:6] + "-01"): i["views"] for i in items})
    return pd.DataFrame(out).sort_index()


def fetch_pubmed(queries: dict[str, str], cache: Path, start="2022-11", end="2026-09") -> pd.DataFrame:
    """Records added to PubMed per month (Entrez date), cached as CSV. Recent months are complete: no publication-date lag."""
    f = cache / "pubmed_edat_monthly.csv"
    cache.mkdir(parents=True, exist_ok=True)
    if f.exists():
        return pd.read_csv(f, index_col=0, parse_dates=True)
    months = pd.period_range(start, end, freq="M")
    rows = {}
    for p in months:
        row = {}
        for k, q in queries.items():
            url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&rettype=count&datetype=edat"
                   f"&mindate={p.start_time:%Y/%m/%d}&maxdate={p.end_time:%Y/%m/%d}&term={urllib.parse.quote(q)}")
            try:
                row[k] = int(json.loads(_curl(url))["esearchresult"]["count"])
            except Exception:
                row[k] = None
            time.sleep(0.4)  # NCBI: at most 3 requests / second without a key
        rows[p.start_time] = row
    df = pd.DataFrame(rows).T.sort_index()
    df.to_csv(f)
    return df


def level_index(s: pd.Series, last_n: int = 6) -> pd.Series:
    """The series as % of its current level (mean of the last `last_n` observations)."""
    return s / s.iloc[-last_n:].mean() * 100


def first_reach(index_s: pd.Series, pct: float, smooth: int = 1) -> pd.Timestamp | None:
    """First date a (optionally smoothed) %-of-current-level series reaches `pct`; None if never."""
    s = index_s.rolling(smooth, min_periods=1).mean() if smooth > 1 else index_s
    hit = s[s >= pct]
    return hit.index[0] if len(hit) else None


def months_between(a: pd.Timestamp | None, b: pd.Timestamp | None) -> int | None:
    return None if a is None or b is None else (a.year - b.year) * 12 + (a.month - b.month)
