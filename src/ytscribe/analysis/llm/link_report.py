"""Aggregate the per-talk link assessments into event-level results, an audit of the keyword scope, and a review file."""

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ytscribe.analysis.config import TopicsConfig
from ytscribe.analysis.corpus import Doc
from ytscribe.analysis.lexicon import count
from ytscribe.analysis.llm import link as lk
from ytscribe.db.models import Annotation

CLASSES = ["ontology_supports_ai", "ai_supports_ontology", "other_link", "parallel", "none"]


def load_assessments(session: Session) -> pd.DataFrame:
    rows = []
    for a in session.scalars(select(Annotation).where(Annotation.kind == lk.KIND, Annotation.prompt_version == lk.PROMPT_VERSION)):
        d = a.data
        conn_q = [e["quote"] for e in d["evidence"] if e["claim"] == "connection" and e.get("verified")]
        rows.append({"video_id": a.video_id, "model": a.model.split("/")[-1], "scope": d.get("scope", ""),
                     "connection": d["connection"], "confidence": d["confidence"], "a_coverage": d["a_coverage"],
                     "b_coverage": d["b_coverage"], "quotes_verified": a.verified_ratio, "rationale": d["rationale"],
                     "connection_quote": conn_q[0] if conn_q else ""})
    return pd.DataFrame(rows)


def talk_table(docs: list[Doc], patterns: dict, assess: pd.DataFrame) -> pd.DataFrame:
    """One row per talk: keyword counts + the LLM answer (talks not sent are 'not_sent' = keyword scope excluded them)."""
    base = pd.DataFrame([{"video_id": d.video_id, "event": d.event, "t": d.t, "title": d.title, "slide_only": d.slide_only,
                          "kw_ai": count(patterns["AI / ML"], d.text), "kw_ont": count(patterns["Ontologies / semantics"], d.text)}
                         for d in docs])
    out = base.merge(assess, on="video_id", how="left")
    out["connection"] = out["connection"].fillna("not_sent")
    return out


def by_event(talks: pd.DataFrame) -> pd.DataFrame:
    """Per event: talks, how many were sent to the model, and the count of each answer. Talks not sent (no keyword in one of
    the two topics) are reported separately, never silently counted as the model's 'none'."""
    g = talks.groupby(["t", "event"], sort=True)
    out = g.agg(talks=("video_id", "count"), sent=("connection", lambda s: int((s != "not_sent").sum()))).reset_index()
    for c in CLASSES + ["not_sent"]:
        out[c] = g["connection"].apply(lambda s, c=c: int((s == c).sum())).to_numpy()
    out["connected"] = out["ontology_supports_ai"] + out["ai_supports_ontology"] + out["other_link"]
    out["pct_connected"] = out["connected"] / out["talks"] * 100
    out["pct_both_present"] = (out["connected"] + out["parallel"]) / out["talks"] * 100
    return out


def audit(talks: pd.DataFrame) -> dict:
    """What the keyword scope would have missed: random talks outside it that the model WAS asked about."""
    a = talks[talks["scope"] == "audit"]
    return {"audited": len(a), "connection_found": int(a["connection"].isin(lk.CONNECTED).sum()),
            "both_present_by_model": int(((a.a_coverage != "none") & (a.b_coverage != "none")).sum()),
            "model_saw_ai_but_no_keyword": int(((a.a_coverage != "none") & (a.kw_ai == 0)).sum()),
            "model_saw_ontology_but_no_keyword": int(((a.b_coverage != "none") & (a.kw_ont == 0)).sum())}


def agreement_with_keywords(talks: pd.DataFrame) -> dict:
    s = talks[talks["connection"] != "not_sent"]
    return {"assessed": len(s),
            "ai_model_none_but_keyword": int(((s.a_coverage == "none") & (s.kw_ai > 0)).sum()),
            "ai_model_yes_but_no_keyword": int(((s.a_coverage != "none") & (s.kw_ai == 0)).sum()),
            "ont_model_none_but_keyword": int(((s.b_coverage == "none") & (s.kw_ont > 0)).sum()),
            "ont_model_yes_but_no_keyword": int(((s.b_coverage != "none") & (s.kw_ont == 0)).sum())}


def review_markdown(talks: pd.DataFrame, prompt_hash: str) -> str:
    lines = [f"# AI x ontologies: talks the model judged CONNECTED (prompt {lk.PROMPT_VERSION} [{prompt_hash}])", "",
             "Each row: the model's answer, its verified quote (found in the transcript) and its rationale. Read the quote and decide "
             "whether you agree: this file is the thing to check, not the model's say-so.", ""]
    c = talks[talks.connection.isin(lk.CONNECTED)].sort_values(["t", "title"])
    for _, r in c.iterrows():
        lines += [f"## {r.event} | {r.title}", f"- answer: **{r.connection}** (confidence {r.confidence}; AI {r.a_coverage}, ontology "
                  f"{r.b_coverage}; keyword counts AI={r.kw_ai}, ontology={r.kw_ont}; model {r.model}; scope {r.scope})",
                  f"- quote: \"{r.connection_quote}\"", f"- rationale: {r.rationale}", ""]
    return "\n".join(lines)


__all__ = ["load_assessments", "talk_table", "by_event", "audit", "agreement_with_keywords", "review_markdown", "TopicsConfig"]
