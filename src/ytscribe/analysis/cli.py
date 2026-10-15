"""`ys topics ...`: trends, terms, projection, charts, sentiment. Outputs go to data/analysis/."""

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ytscribe.analysis import charts, trends
from ytscribe.analysis import sentiment as sent
from ytscribe.analysis import terms as terms_mod
from ytscribe.analysis.config import load_topics
from ytscribe.analysis.corpus import load_docs
from ytscribe.analysis.lexicon import compile_topics
from ytscribe.analysis.projection import load_agenda, project_event
from ytscribe.db.session import get_session
from ytscribe.settings import get_settings

topics_app = typer.Typer(help="Topic and trend analysis over the transcripts (deterministic, no LLM).", no_args_is_help=True)
console = Console(width=200)
CfgOpt = typer.Option(None, "--topics", help="Topic config (default configs/topics/allotrope.yaml)")


def _out() -> Path:
    d = get_settings().data_dir / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prepare(cfg_path):
    cfg = load_topics(cfg_path)
    patterns = compile_topics(cfg)
    with get_session() as session:
        docs = load_docs(session, cfg)
    counts = trends.topic_counts(docs, patterns)
    return cfg, patterns, docs, counts


def _print(df: pd.DataFrame, title: str, digits: int = 1):
    t = Table(title=title)
    for c in df.columns:
        t.add_column(str(c))
    for _, r in df.iterrows():
        t.add_row(*[f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in r])
    console.print(t)


@topics_app.command("trends")
def trends_cmd(topics: str = CfgOpt):
    """Mentions per 10k words and % of talks per event, plus the best single step per topic (ruptures)."""
    cfg, patterns, docs, counts = _prepare(topics)
    out = _out()
    names = list(cfg.topics)
    ev = trends.by_event(counts, names, cfg.min_words_solid, cfg.min_talks_solid)
    counts.to_csv(out / "topic_counts_by_video.csv", index=False)
    ev.to_csv(out / "topic_trends_by_event.csv", index=False)
    cps = trends.changepoints(ev, names)
    cps.to_csv(out / "changepoints.csv", index=False)
    for group, members in cfg.groups.items():
        _print(ev[["event", "talks", "words"] + [f"{m} /10k" for m in members]], f"{group}: mentions per 10k words")
    _print(cps[cps["topic"].isin(sum(cfg.groups.values(), []))], "single-step changepoints (read variance_explained)", 2)
    console.print(f"wrote {out}/topic_counts_by_video.csv, topic_trends_by_event.csv, changepoints.csv")


@topics_app.command("terms")
def terms_cmd(topics: str = CfgOpt):
    """Distinctive terms per event (Fightin' Words log-odds, filler and speaker names removed)."""
    cfg, _, docs, _ = _prepare(topics)
    df = terms_mod.distinctive_terms(docs, cfg.terms)
    df.to_csv(_out() / "distinctive_terms_by_event.csv", index=False)
    for ev, g in df.groupby("event", sort=False):
        console.print(f"[bold]{ev:12}[/bold] " + ", ".join(g.sort_values("rank")["term"]))


@topics_app.command("project")
def project_cmd(agenda: Path = typer.Argument(..., help="Agenda JSON (data/agenda/*.json)"),
                event: str = typer.Option(..., help='Event label, e.g. "2026 Fall"'), topics: str = CfgOpt):
    """Project an event whose videos are only partly uploaded, from its published agenda (titles)."""
    cfg, patterns, docs, counts = _prepare(topics)
    res = project_event(load_agenda(agenda), docs, counts, patterns, cfg.groups["direction"], cfg, event)
    slug = event.lower().replace(" ", "_")
    res["projection"].to_csv(_out() / f"projection_{slug}.csv", index=False)
    res["validation"].to_csv(_out() / f"projection_{slug}_validation.csv", index=False)
    res["missing"].to_csv(_out() / f"projection_{slug}_missing.csv", index=False)
    console.print(f"matched {len(res['matched'])} agenda sessions to videos; {len(res['missing'])} projected")
    for vid, title in res["unmatched_videos"]:
        console.print(f"[yellow]video with no agenda session:[/yellow] {vid} {title}  (add it under \"matches\" in the agenda file)")
    _print(res["validation"], "validation on matched sessions (title-only prediction vs actual, per 10k words)")
    _print(res["projection"], f"{event}: transcribed-only vs projected (p10-p90)")


@topics_app.command("charts")
def charts_cmd(projection: Path = typer.Option(None, help="projection_*.csv from `ys topics project`"),
               projected_event: str = typer.Option(None), topics: str = CfgOpt):
    """Direction, partners and AI/cloud charts."""
    cfg, _, _, counts = _prepare(topics)
    out = _out()
    ev = trends.by_event(counts, list(cfg.topics), cfg.min_words_solid, cfg.min_talks_solid)
    proj = pd.read_csv(projection) if projection else None
    g = cfg.groups
    paths = [charts.direction_chart(ev, g["direction"], out / "chart_direction.png", proj,
                                    projected_event, title="From the format, to ASM, to AI")]
    if "partners" in g:
        paths.append(charts.small_multiples(ev, g["partners"], out / "chart_partners.png",
                                            "Partners and standards bodies", ncols=2))
    if "ai_cloud" in g:
        paths.append(charts.small_multiples(ev, g["ai_cloud"], out / "chart_ai_cloud.png",
                                            "AI and cloud companies named in the talks", ncols=3,
                                            note="2023 Fall is slide-only text."))
    console.print("\n".join(str(p) for p in paths))


@topics_app.command("sentiment")
def sentiment_cmd(group: str = typer.Option("direction", help="Topic group from the config"), topics: str = CfgOpt):
    """Tone around topic mentions (VADER baseline), relative to each event's ordinary text."""
    cfg, patterns, docs, _ = _prepare(topics)
    df = sent.topic_sentiment(docs, patterns, cfg.groups[group], cfg.sentiment.window_words)
    df.to_csv(_out() / f"topic_sentiment_{group}.csv", index=False)
    piv = df.pivot(index="event", columns="topic", values="relative").reindex(list(dict.fromkeys(d.event for d in docs)))
    piv.insert(0, "event", piv.index)
    _print(piv.reset_index(drop=True), f"{group}: tone around mentions minus the event baseline (VADER compound)", 2)
