"""`ys llm ...`"""


from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from sqlalchemy import select

from ytscribe.analysis.config import load_topics
from ytscribe.analysis.corpus import load_docs
from ytscribe.analysis.llm import link as lk
from ytscribe.db.models import Annotation
from ytscribe.db.session import get_session, init_db
from ytscribe.settings import get_settings

llm_app = typer.Typer(help="LLM analysis layer (runs on the free-tier router, local Ollama fallback).", no_args_is_help=True)
console = Console(width=200)


@llm_app.command("show-prompt")
def show_prompt(save: bool = typer.Option(True, help="Also write it to data/analysis/llm/")):
    """Print the exact prompt used by `ys llm link` (system + user template)."""
    text = lk.render_prompt()
    console.print(text, markup=False)
    console.print(f"\nprompt version: {lk.PROMPT_VERSION}   hash: {lk.prompt_hash()}", markup=False)
    if save:
        out = get_settings().data_dir / "analysis" / "llm"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"prompt_{lk.PROMPT_VERSION}.txt").write_text(text)


@llm_app.command("link")
def link_cmd(limit: int = typer.Option(None, help="Only the first N talks (smoke test)"),
             video: list[str] = typer.Option(None, "--video"), route: str = typer.Option("link"),
             redo: bool = typer.Option(False, help="Re-assess talks that already have a result for this prompt version"),
             scope: str = typer.Option("both", help="both = talks where AI and ontology terms both appear (default) | all"),
             audit: int = typer.Option(0, help="Also assess N random talks outside the scope, to measure what the keyword scope misses")):
    """Ask the model, per talk, whether the speaker connects AI with ontologies (and how). Resumable."""
    from ytscribe.analysis.llm.client import enabled_deployments, get_router

    init_db()
    cfg = load_topics()
    router = get_router()
    console.print(f"enabled deployments (in route order): {enabled_deployments(router)}", markup=False)
    with get_session() as session:
        docs = load_docs(session, cfg)
        if video:
            docs = [d for d in docs if d.video_id in set(video)]
        done = {a.video_id for a in session.scalars(select(Annotation).where(
            Annotation.kind == lk.KIND, Annotation.prompt_version == lk.PROMPT_VERSION))}
        from ytscribe.analysis.lexicon import compile_topics, count

        pats = compile_topics(cfg)
        both = {d.video_id for d in docs if count(pats["AI / ML"], d.text) and count(pats["Ontologies / semantics"], d.text)}
        scope_of = {d.video_id: ("both" if d.video_id in both else "all") for d in docs}
        in_scope = [d for d in docs if scope == "all" or d.video_id in both]
        outside = [d for d in docs if d not in in_scope]
        import random

        audit_docs = random.Random(0).sample(outside, min(audit, len(outside))) if audit else []
        for d in audit_docs:
            scope_of[d.video_id] = "audit"
        todo = [d for d in in_scope + audit_docs if redo or d.video_id not in done]
        todo = todo[:limit] if limit else todo
        console.print(f"scope={scope}: {len(in_scope)} talks with both topics by keyword, {len(outside)} others "
                      f"({len(audit_docs)} audited)", markup=False)
        console.print(f"{len(todo)} talks to assess ({len(done)} already done), prompt {lk.PROMPT_VERSION} [{lk.prompt_hash()}]",
                      markup=False)
        for i, d in enumerate(todo, 1):
            with logger.contextualize(video_id=d.video_id):
                try:
                    r = lk.assess_talk(router, d, route, scope=scope_of[d.video_id])
                except Exception as e:
                    logger.error("link assessment failed for {}: {}", d.video_id, e)
                    console.print(f"[{i}/{len(todo)}] FAILED {d.video_id}: {str(e)[:120]}", markup=False)
                    continue
                old = session.scalar(select(Annotation).where(
                    Annotation.video_id == d.video_id, Annotation.kind == lk.KIND,
                    Annotation.prompt_version == lk.PROMPT_VERSION, Annotation.model == r.model))
                if old:
                    old.data, old.verified_ratio, old.deployment = r.data, r.verified_ratio, r.deployment
                else:
                    session.add(Annotation(video_id=d.video_id, kind=lk.KIND, prompt_version=lk.PROMPT_VERSION,
                                           deployment=r.deployment, model=r.model, data=r.data, verified_ratio=r.verified_ratio))
                session.commit()
                console.print(f"[{i}/{len(todo)}] {d.event:8} {d.video_id} {r.data['connection']:22} a={r.data['a_coverage']:11} "
                              f"b={r.data['b_coverage']:11} quotes ok {r.verified_ratio:.0%} ({r.model.split('/')[-1]}) "
                              f"{d.title[:40]}", markup=False)


@llm_app.command("link-report")
def link_report_cmd():
    """Event-level results, keyword-scope audit, agreement with keyword counts, chart and a review file of connected talks."""
    import pandas as pd

    from ytscribe.analysis import charts
    from ytscribe.analysis import trends as tr
    from ytscribe.analysis.lexicon import compile_topics
    from ytscribe.analysis.llm import link_report as lr

    cfg = load_topics()
    pats = compile_topics(cfg)
    out = get_settings().data_dir / "analysis" / "llm"
    out.mkdir(parents=True, exist_ok=True)
    with get_session() as session:
        docs = load_docs(session, cfg)
        assess = lr.load_assessments(session)
    talks = lr.talk_table(docs, pats, assess)
    ev = lr.by_event(talks)
    thin = tr.by_event(tr.topic_counts(docs, pats), list(cfg.topics), cfg.min_words_solid, cfg.min_talks_solid)[["event", "thin"]]
    ev = ev.merge(thin, on="event", how="left")
    talks.to_csv(out / "link_talks.csv", index=False)
    ev.to_csv(out / "link_by_event.csv", index=False)
    (out / "link_review.md").write_text(lr.review_markdown(talks, lk.prompt_hash()))
    chart = charts.link_chart(ev, out / "chart_link_ai_ontology.png", "Do talks connect AI with ontologies? (LLM judgement)",
                              f"Prompt {lk.PROMPT_VERSION}. % of ALL talks in the event; talks without both topics by keyword are the blank remainder. "
                              "Hatched = 1-2 talks or slide-only.")
    pd.set_option("display.width", 220)
    console.print(ev[["event", "talks", "sent", "ontology_supports_ai", "ai_supports_ontology", "other_link", "parallel", "none",
                      "not_sent", "pct_connected"]].round(1).to_string(index=False), markup=False)
    console.print(f"overall answers (talks sent): {talks[talks.connection != 'not_sent'].connection.value_counts().to_dict()}", markup=False)
    console.print(f"audit of the keyword scope: {lr.audit(talks)}", markup=False)
    console.print(f"model vs keyword coverage: {lr.agreement_with_keywords(talks)}", markup=False)
    console.print(f"models used: {assess.model.value_counts().to_dict()}; wrote {chart} and {out / 'link_review.md'}", markup=False)


@llm_app.command("show-judge-prompt")
def show_judge_prompt():
    """Print the exact prompt of the passage-level judge (stage 2)."""
    from ytscribe.analysis.llm import judge as jg

    console.print(jg.render_prompt(), markup=False)
    console.print(f"\nprompt version: {jg.PROMPT_VERSION}   hash: {jg.prompt_hash()}", markup=False)
    out = get_settings().data_dir / "analysis" / "llm"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"prompt_{jg.PROMPT_VERSION}.txt").write_text(jg.render_prompt())


@llm_app.command("link-judge")
def link_judge_cmd(route: str = typer.Option("judge_cloud", help="judge_cloud (free-tier model) | judge_local (Ollama)"),
                   limit: int = typer.Option(None)):
    """Judge each co-occurrence passage (AI near an ontology/semantic term) with one model. Run once per route; resumable."""
    from ytscribe.analysis.lexicon import compile_topics
    from ytscribe.analysis.llm import judge as jg
    from ytscribe.analysis.llm.client import enabled_deployments, get_router

    init_db()
    cfg = load_topics()
    pats = compile_topics(cfg)
    router = get_router()
    console.print(f"route {route}: deployments {router.routes[route]} (enabled: {enabled_deployments(router)})", markup=False)
    with get_session() as session:
        docs = load_docs(session, cfg)
        cands = jg.candidates(docs, pats, cfg)
        model_tag = router.routes[route][0]
        existing = {a.video_id: a for a in session.scalars(select(Annotation).where(
            Annotation.kind == jg.KIND, Annotation.prompt_version == jg.PROMPT_VERSION, Annotation.deployment == model_tag))}
        by_video: dict[str, list] = {}
        for c in cands:
            by_video.setdefault(c.video_id, []).append(c)
        console.print(f"{len(cands)} candidate passages in {len(by_video)} talks; prompt {jg.PROMPT_VERSION} [{jg.prompt_hash()}]",
                      markup=False)
        n = 0
        for vid, cs in by_video.items():
            done_ids = {x["cid"] for x in (existing[vid].data["items"] if vid in existing else [])}
            todo = [c for c in cs if c.cid not in done_ids]
            if not todo:
                continue
            items = list(existing[vid].data["items"]) if vid in existing else []
            for c in todo:
                if limit and n >= limit:
                    break
                try:
                    items.append({**jg.judge(router, route, c), "text": c.text})
                except Exception as e:
                    logger.error("judge failed {}: {}", c.cid, e)
                    console.print(f"FAILED {c.cid}: {str(e)[:100]}", markup=False)
                    continue
                n += 1
                console.print(f"[{n}] {c.event:8} {c.cid:16} {items[-1]['relation']:22} {items[-1]['model']}", markup=False)
            if vid in existing:
                existing[vid].data = {"items": items}
            else:
                session.add(Annotation(video_id=vid, kind=jg.KIND, prompt_version=jg.PROMPT_VERSION, deployment=model_tag,
                                       model=model_tag, data={"items": items}))
            session.commit()
            if limit and n >= limit:
                break


@llm_app.command("link-judge-report")
def link_judge_report_cmd():
    """Agreement between the two judges, evidence tiers per talk and event, review file and the gold-label page."""
    from ytscribe.analysis import charts
    from ytscribe.analysis import trends as tr
    from ytscribe.analysis.lexicon import compile_topics
    from ytscribe.analysis.llm import judge_report as jr
    from ytscribe.analysis.llm import link_report as lr

    cfg = load_topics()
    pats = compile_topics(cfg)
    out = get_settings().data_dir / "analysis" / "llm"
    gold = get_settings().data_dir / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    with get_session() as session:
        docs = load_docs(session, cfg)
        assess = lr.load_assessments(session)
        items = jr.load_items(session)
    talks = lr.talk_table(docs, pats, assess)
    tiers = jr.talk_tiers(talks, items)
    ev = jr.tiers_by_event(tiers).merge(
        tr.by_event(tr.topic_counts(docs, pats), list(cfg.topics), cfg.min_words_solid, cfg.min_talks_solid)[["event", "thin"]],
        on="event", how="left")
    items.to_csv(out / "judge_items.csv", index=False)
    tiers.to_csv(out / "judge_talks.csv", index=False)
    ev.to_csv(out / "judge_by_event.csv", index=False)
    (out / "judge_review.md").write_text(jr.review_markdown(items, talks))
    n_gold = jr.gold_html(items, talks, gold / "link_gold_review.html")
    chart = charts.tier_chart(ev, out / "chart_link_tiers.png", "Do speakers relate AI to ontologies? Evidence by strength",
                              f"Bars = % of the event's talks. Label = talks with an explicit, model-confirmed relation / talks. "
                              f"Judge prompt {jr.jg.PROMPT_VERSION}; whole-talk prompt {lk.PROMPT_VERSION}.")
    console.print(f"candidate passages: {len(items)} | agreement between the two judges: {jr.agreement(items)}", markup=False)
    console.print(ev[["event", "talks", "explicit_both_agree", "explicit_models_disagree", "implied_only", "none"]].to_string(index=False),
                  markup=False)
    console.print(f"wrote {chart}, {out / 'judge_review.md'}, and the gold-label page {gold / 'link_gold_review.html'} ({n_gold} passages)",
                  markup=False)


@llm_app.command("gold-import")
def gold_import(path: str = typer.Argument(..., help="gold_link_labels.json downloaded from the review page")):
    """Store your labels (item id -> label) as the reference the models are scored against."""
    import json

    from ytscribe.db.models import GoldLabel

    labels = json.loads(Path(path).read_text())
    init_db()
    with get_session() as session:
        for cid, lab in labels.items():
            vid = cid.split(":")[0]
            row = session.scalar(select(GoldLabel).where(GoldLabel.item_id == cid, GoldLabel.field == "link"))
            if row:
                row.value = lab
            else:
                session.add(GoldLabel(item_id=cid, video_id=vid, field="link", value=lab))
    console.print(f"imported {len(labels)} labels", markup=False)


@llm_app.command("gold-eval")
def gold_eval():
    """Score each judge, their consensus and the keyword baseline against your labels (accuracy and Cohen's kappa)."""
    from sklearn.metrics import cohen_kappa_score

    from ytscribe.analysis.llm import judge as jg
    from ytscribe.analysis.llm import judge_report as jr
    from ytscribe.db.models import GoldLabel

    with get_session() as session:
        items = jr.load_items(session).set_index("cid")
        gold = {g.item_id: g.value for g in session.scalars(select(GoldLabel).where(GoldLabel.field == "link"))}
    g = pd_series(gold)
    items = items.loc[[c for c in items.index if c in g.index]]
    if items.empty:
        console.print("no gold labels imported yet (run `ys llm gold-import`)", markup=False)
        return
    truth = g.loc[items.index].isin(jg.CONNECTED)
    for name, pred in (("cloud judge", items.rel_cloud.isin(jg.CONNECTED)), ("local judge", items.rel_local.isin(jg.CONNECTED)),
                       ("both agree", items.rel_cloud.isin(jg.CONNECTED) & items.rel_local.isin(jg.CONNECTED)),
                       ("keyword co-occurrence alone", pd_series({c: True for c in items.index}))):
        pred = pred.loc[items.index]
        kappa = cohen_kappa_score(truth, pred) if truth.nunique() > 1 or pred.nunique() > 1 else float("nan")
        console.print(f"{name:28} accuracy {float((pred == truth).mean()):.0%}  kappa {kappa:.2f}  (n={len(items)}; "
                      f"human says related: {int(truth.sum())})", markup=False)


def pd_series(d):
    import pandas as pd

    return pd.Series(d)


@llm_app.command("taxonomy-proposal")
def taxonomy_proposal(mapping: str = typer.Option("configs/topics/taxonomy_categories_v1.yaml"),
                      out: str = typer.Option(None, help="default: data/analysis/discover/taxonomy_proposed.yaml")):
    """Write the reviewer-fill taxonomy (17 categories, every key you must set is null). The original draft is never modified."""
    from sqlalchemy import select as sel

    from ytscribe.analysis.llm.taxonomy import build_proposal
    from ytscribe.db.models import Video

    cfg = load_topics()
    d = get_settings().data_dir / "analysis" / "discover"
    target = Path(out) if out else d / "taxonomy_proposed.yaml"
    with get_session() as session:
        titles = {v.id: v.title or "" for v in session.scalars(sel(Video))}
    res = build_proposal(mapping, d / "taxonomy_draft.yaml", d / "passage_topics.csv", titles, target, cfg.season_labels)
    console.print(f"wrote {target}: {res['categories']} categories, {res['unassigned']} unassigned themes, {res['nulls']} keys for you to fill",
                  markup=False)


@llm_app.command("taxonomy-check")
def taxonomy_check(path: str = typer.Argument(...)):
    """List the keys still null in a taxonomy file, or load it if it is complete and approved."""
    import yaml

    from ytscribe.analysis.llm import taxonomy as tx

    raw = yaml.safe_load(Path(path).read_text())
    todo = tx.missing_keys(raw) if "categories" in raw else []
    if todo:
        console.print(f"{len(todo)} keys still null: {', '.join(todo)}", markup=False)
        return
    try:
        t = tx.load_taxonomy(path)
        console.print(f"OK: approved, {len(t.topics)} categories kept, {len(t.anchors)} anchors", markup=False)
    except (tx.TaxonomyNotApproved, tx.TaxonomyIncomplete, ValueError) as e:
        console.print(f"not ready: {e}", markup=False)
