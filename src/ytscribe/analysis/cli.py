"""`ys topics ...`: trends, terms, projection, charts, sentiment. Outputs go to data/analysis/."""

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ytscribe.analysis import charts, trends
from ytscribe.analysis import sentiment as sent
from ytscribe.analysis import terms as terms_mod
from ytscribe.analysis.config import load_topics, normalize_event
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
    ev = trends.by_event(counts, names, cfg.min_words_solid, cfg.min_talks_solid, cfg.substantive_mentions)
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
                event: str = typer.Option(..., help='Event label, e.g. "2026 H2" ("2026 Fall" also works)'), topics: str = CfgOpt):
    """Project an event whose videos are only partly uploaded, from its published agenda (titles)."""
    cfg, patterns, docs, counts = _prepare(topics)
    event = normalize_event(cfg, event)
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
    ev = trends.by_event(counts, list(cfg.topics), cfg.min_words_solid, cfg.min_talks_solid, cfg.substantive_mentions)
    proj = pd.read_csv(projection) if projection else None
    g = cfg.groups
    steps = trends.changepoints(ev, g["direction"])
    paths = [charts.stacked_chart(ev, g["direction"], out / "chart_direction.png", proj, projected_event, steps,
                                  title="From the format, to ASM, to AI")]
    paths.append(charts.talk_share_chart(ev, g["direction"], out / "chart_talk_share.png", cfg.substantive_mentions))
    if "partners" in g:
        paths.append(charts.small_multiples(ev, g["partners"], out / "chart_partners.png",
                                            "Partners and standards bodies", ncols=2))
    if "ai_cloud" in g:
        paths.append(charts.small_multiples(ev, g["ai_cloud"], out / "chart_ai_cloud.png",
                                            "AI and cloud companies named in the talks", ncols=3,
                                            note="2023 H2 is slide-only text."))
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


@topics_app.command("discover")
def discover_cmd(min_topic_size: int = typer.Option(10, help="Smallest cluster (in ~150-word passages)"),
                 nr_topics: int = typer.Option(None, help="Merge down to about this many topics"),
                 embedding_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"), topics: str = CfgOpt):
    """Unsupervised topics (BERTopic) -> topics.csv, topic_share_by_event.csv and taxonomy_draft.yaml for human review."""
    from ytscribe.analysis.discover import discover

    cfg, patterns, docs, _ = _prepare(topics)
    res = discover(docs, cfg, patterns, _out() / "discover", embedding_model=embedding_model,
                   min_topic_size=min_topic_size, nr_topics=nr_topics)
    console.print(res)


@topics_app.command("breakdown")
def breakdown_cmd(anchor: list[str] = typer.Option(None, "--anchor", help="Anchor topic(s); default: all in the config"),
                  topics: str = CfgOpt):
    """What an anchor topic (AI, ASM...) is made of: named sub-terms and context themes, stacked per event."""
    from ytscribe.analysis import breakdown as bd

    cfg, patterns, docs, counts = _prepare(topics)
    out = _out()
    ev = trends.by_event(counts, list(cfg.topics), cfg.min_words_solid, cfg.min_talks_solid)
    ev_idx = ev.set_index("event")
    ew_all, ew_speech = bd.event_words_of(docs), bd.event_words_of(docs, include_slides=False)
    for name, acfg in cfg.breakdown.anchors.items():
        if anchor and name not in anchor:
            continue
        slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
        panels = []
        if acfg.terms:
            terms_df = bd.term_breakdown(docs, patterns[name], acfg.terms, ew_all)
            terms_df.to_csv(out / f"breakdown_{slug}_terms.csv")
            panels.append({"title": f"{name}: named sub-terms (each mention counted once)", "rates": terms_df})
            _print(terms_df.reset_index(names="event").round(2), f"{name}: sub-terms per 10k words", 2)
        wins = bd.anchor_windows(docs, patterns[name], cfg.breakdown.window_words)
        clusterer = bd.bertopic_clusters(acfg.min_topic_size or cfg.breakdown.min_topic_size, cfg.breakdown.nr_topics,
                                         stopwords=cfg.terms.extra_stopwords)
        res = bd.context_themes(wins, ew_speech, clusterer)
        res["rates"].to_csv(out / f"breakdown_{slug}_themes.csv")
        res["assignments"].to_csv(out / f"breakdown_{slug}_assignments.csv", index=False)
        panels.append({"title": f"{name}: what it is discussed with ({len(wins)} spoken mentions, by context theme)",
                       "rates": res["rates"]})
        _print(res["rates"].reset_index(names="event").round(2), f"{name}: context themes, mention windows per 10k words", 2)
        path = charts.composition_chart(panels, ev_idx.reset_index(), out / f"chart_breakdown_{slug}.png",
                                        f"{name}: one topic, what it is made of")
        console.print(f"wrote {path}")


@topics_app.command("correlate")
def correlate_cmd(focus: str = typer.Option("AI / ML", help="Topic to report in detail"), topics: str = CfgOpt):
    """Which topics are interconnected? Talk-level (pooled and within-event) and passage-level association."""
    from ytscribe.analysis import association as assoc

    cfg, patterns, docs, counts = _prepare(topics)
    cc = cfg.correlation
    wanted = [t for g in cc.groups for t in cfg.groups[g]]
    pres = assoc.passage_presence(docs, patterns, wanted, cc.passage_words)
    names, dropped = assoc.usable_topics(counts[~counts["slide_only"]], pres, wanted, cc.min_talks, cc.min_passages)
    for t, nt, npas in dropped:
        console.print(f"[yellow]left out[/yellow] {t}: only {nt} talks / {npas} passages mention it (need {cc.min_talks} / {cc.min_passages})")
    out = _out()
    rates = assoc.talk_rates(counts, names)
    rho_p, p_p = assoc.spearman_matrix(rates, names, within_event=False)
    rho_w, p_w = assoc.spearman_matrix(rates, names, within_event=True)
    q_p, q_w = assoc.fdr_matrix(p_p), assoc.fdr_matrix(p_w)
    ptab = assoc.passage_table(pres, names)
    q_pass = assoc.matrix_from_table(ptab, names, "q_by_event")
    rho_p.to_csv(out / "corr_talks_pooled.csv")
    rho_w.to_csv(out / "corr_talks_within_event.csv")
    ptab.to_csv(out / "assoc_passages.csv", index=False)
    console.print(f"{len(rates)} talks in {rates['event'].nunique()} events, {len(pres)} passages, {len(names)} topics")

    rows = []
    for t in names:
        if t == focus:
            continue
        lo_p, hi_p = assoc.bootstrap_ci(rates, focus, t, False, cc.bootstrap)
        lo_w, hi_w = assoc.bootstrap_ci(rates, focus, t, True, cc.bootstrap)
        r = ptab[((ptab.a == focus) & (ptab.b == t)) | ((ptab.a == t) & (ptab.b == focus))].iloc[0]
        rows.append({"topic": t, "rho_pooled": rho_p.loc[focus, t], "ci_pooled": f"{lo_p:.2f}..{hi_p:.2f}",
                     "rho_within_event": rho_w.loc[focus, t], "ci_within": f"{lo_w:.2f}..{hi_w:.2f}",
                     "q_within": q_w.loc[focus, t], "passages_both": r.passages_both, "phi": r.phi, "lift": r.lift,
                     "OR_by_event": r.odds_by_event, "OR_ci": f"{r.or_lo:.1f}..{r.or_hi:.1f}", "q_passage": r.q_by_event})
    focus_df = pd.DataFrame(rows).sort_values("phi", ascending=False)
    focus_df.to_csv(out / f"assoc_focus_{''.join(c if c.isalnum() else '_' for c in focus.lower()).strip('_')}.csv", index=False)
    _print(focus_df, f"{focus} vs every other topic (rho = talk level, phi/OR = same 150-word passage; q = FDR-adjusted p)", 2)

    sub = "Red = topics appear together, blue = one tends to replace the other. * q<0.05, ** q<0.01 (Benjamini-Hochberg over all pairs)."
    p1 = charts.corr_heatmaps([
        {"title": "Talk level, pooled across all events: Spearman rho of topic rates per 10k words", "matrix": rho_p, "q": q_p,
         "scale_label": "Spearman rho"},
        {"title": "Talk level, within event (each event's mean removed, so the time trend cannot drive it)", "matrix": rho_w,
         "q": q_w, "scale_label": "Spearman rho"}], out / "chart_corr_talks.png", "Which topics are interconnected? (talks)", sub)
    ok = ptab.assign(odds_ok=ptab["odds_by_event"].where(ptab["expected_both"] >= 3))  # need >= 3 expected shared passages
    odds = assoc.matrix_from_table(ok, names, "odds_ok")
    lor = np.log2(odds.astype(float).replace(0, np.nan)).clip(-3, 3)
    p2 = charts.corr_heatmaps([{"title": f"Same {cc.passage_words}-word passage: odds of appearing together vs chance "
                                         "(1 = no link; event-stratified)", "matrix": lor, "labels": odds, "fmt": ".2f", "q": q_pass,
                                "vmax": 3.0, "scale_label": "log2 odds ratio (colour); the number shown is the odds ratio"}],
                              out / "chart_corr_passages.png", "Which topics are interconnected? (passages)",
                              sub + " Above 1 = together more than chance, below 1 = less (0.50 = half the odds). – = too few shared passages expected (<3) to judge.")
    console.print(f"{p1}\n{p2}")
