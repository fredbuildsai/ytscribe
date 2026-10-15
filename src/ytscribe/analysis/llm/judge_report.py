"""Combine the two judges (different model families) and the whole-talk pass into evidence tiers per talk."""

import html
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sqlalchemy import select
from sqlalchemy.orm import Session

from ytscribe.analysis.llm import judge as jg
from ytscribe.analysis.llm import link as lk
from ytscribe.db.models import Annotation

TIERS = ["explicit_both_agree", "explicit_models_disagree", "implied_only", "none"]


def load_items(session: Session) -> pd.DataFrame:
    """One row per candidate passage with each model's relation, quote and reason."""
    rows = {}
    for a in session.scalars(select(Annotation).where(Annotation.kind == jg.KIND, Annotation.prompt_version == jg.PROMPT_VERSION)):
        side = "cloud" if "openrouter" in a.deployment else "local"
        for it in a.data["items"]:
            r = rows.setdefault(it["cid"], {"cid": it["cid"], "video_id": a.video_id, "text": it["text"]})
            r[f"rel_{side}"], r[f"quote_{side}"], r[f"reason_{side}"] = it["relation"], it["quote"], it["reason"]
    df = pd.DataFrame(rows.values())
    for side in ("cloud", "local"):
        for col in (f"rel_{side}", f"quote_{side}", f"reason_{side}"):
            if col not in df:
                df[col] = None
    return df


def agreement(items: pd.DataFrame) -> dict:
    both = items.dropna(subset=["rel_cloud", "rel_local"])
    if both.empty:
        return {"n": 0}
    a, b = both.rel_cloud.isin(jg.CONNECTED), both.rel_local.isin(jg.CONNECTED)
    kappa = cohen_kappa_score(a, b) if a.nunique() > 1 or b.nunique() > 1 else float("nan")
    return {"n": len(both), "both_connected": int((a & b).sum()), "cloud_only": int((a & ~b).sum()), "local_only": int((~a & b).sum()),
            "neither": int((~a & ~b).sum()), "percent_agree": float((a == b).mean() * 100), "cohen_kappa_connected": float(kappa),
            "direction_agrees_when_both_connected": float((both[a & b].rel_cloud == both[a & b].rel_local).mean() * 100)
            if (a & b).any() else float("nan")}


def item_status(items: pd.DataFrame) -> pd.Series:
    """agree / disagree / none from the two judges; 'incomplete' when either model has no judgement for the passage yet."""
    cc, lc = items.rel_cloud.isin(jg.CONNECTED), items.rel_local.isin(jg.CONNECTED)
    done = items.rel_cloud.notna() & items.rel_local.notna()
    return pd.Series(["incomplete" if not d else "agree" if c and loc else "disagree" if c or loc else "none"
                      for d, c, loc in zip(done, cc, lc, strict=False)], index=items.index)


def talk_tiers(talks: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """talks: link_report.talk_table output (has the whole-talk 'connection'). Tier per talk, strongest evidence wins."""
    it = items.assign(status=item_status(items))
    best = it.groupby("video_id")["status"].agg(lambda s: "agree" if (s == "agree").any() else "disagree" if (s == "disagree").any()
                                                  else "none")
    t = talks.copy()
    t["judge"] = t.video_id.map(best).fillna("no_candidate")
    implied = t.connection.isin(lk.CONNECTED)
    t["tier"] = "none"
    t.loc[implied, "tier"] = "implied_only"
    t.loc[t.judge == "disagree", "tier"] = "explicit_models_disagree"
    t.loc[t.judge == "agree", "tier"] = "explicit_both_agree"
    return t


def tiers_by_event(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby(["t", "event"], sort=True)
    out = g.agg(talks=("video_id", "count")).reset_index()
    for c in TIERS:
        out[c] = g["tier"].apply(lambda s, c=c: int((s == c).sum())).to_numpy()
    out["explicit_any"] = out["explicit_both_agree"] + out["explicit_models_disagree"]
    out["pct_agree"] = out["explicit_both_agree"] / out["talks"] * 100
    return out


def review_markdown(items: pd.DataFrame, talks: pd.DataFrame) -> str:
    title = dict(zip(talks.video_id, talks.event + " | " + talks.title, strict=False))
    st = item_status(items)
    lines = [f"# AI x ontologies: passage-level judgements (prompt {jg.PROMPT_VERSION} [{jg.prompt_hash()}])", "",
             "Two different model families judged each passage independently. READ THE PASSAGES: that is the evidence.", ""]
    for status, head in (("agree", "Both models say the speaker relates AI and ontologies"), ("disagree", "Models disagree (decide yourself)")):
        lines += [f"## {head}", ""]
        for _, r in items[st == status].sort_values("cid").iterrows():
            lines += [f"### {title.get(r.video_id, r.video_id)}  ({r.cid})",
                      f"- cloud ({r.rel_cloud}): \"{r.quote_cloud}\" , {r.reason_cloud}",
                      f"- local ({r.rel_local}): \"{r.quote_local}\" , {r.reason_local}", f"- passage: {r.text}", ""]
    return "\n".join(lines)


def gold_html(items: pd.DataFrame, talks: pd.DataFrame, path: Path, n_agree_sample: int = 6, n_none_sample: int = 6) -> int:
    """Self-contained review page: every disagreement, plus samples of agreed-connected and agreed-none (so you can also
    catch cases where BOTH models are wrong). Choices persist in the browser; 'Download JSON' exports them for `ys llm gold-import`."""
    st = item_status(items)
    pick = pd.concat([items[st == "disagree"], items[st == "agree"].sample(min(n_agree_sample, (st == "agree").sum()), random_state=0),
                      items[st == "none"].sample(min(n_none_sample, (st == "none").sum()), random_state=0)]).sort_values("cid")
    title = dict(zip(talks.video_id, talks.event + " | " + talks.title, strict=False))
    opts = [("no_relation", "No relation"), ("ontology_supports_ai", "Ontology supports AI"),
            ("ai_supports_ontology", "AI supports ontology"), ("other_link", "Other explicit link")]
    cards = []
    for _, r in pick.iterrows():
        radios = "".join(f'<label><input type="radio" name="{r.cid}" value="{v}"> {html.escape(lbl)}</label>' for v, lbl in opts)
        cards.append(f'<section data-cid="{r.cid}"><h3>{html.escape(title.get(r.video_id, r.video_id))}</h3>'
                     f'<p class="passage">{html.escape(r.text)}</p>'
                     f'<p class="m"><b>Cloud model:</b> {html.escape(str(r.rel_cloud))}. {html.escape(str(r.quote_cloud or ""))}</p>'
                     f'<p class="m"><b>Local model:</b> {html.escape(str(r.rel_local))}. {html.escape(str(r.quote_local or ""))}</p>'
                     f'<div class="opts">{radios}</div></section>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold labels: AI x ontologies</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#383835;--acc:#3987e5}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:16px}}
section{{border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:14px 0}}h3{{font-size:15px;margin:0 0 6px}}
.passage{{background:color-mix(in srgb,var(--acc) 7%,transparent);padding:8px 10px;border-radius:6px}}.m{{color:var(--ink2);font-size:13px;margin:4px 0}}
.opts{{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:8px}}button{{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:14px;cursor:pointer}}
.bar{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line)}}</style></head><body>
<div class="bar"><b>Does the speaker, in this passage, explicitly relate AI and ontologies / semantic models?</b>
<span id="n"></span> <button onclick="dl()">Download my labels (JSON)</button></div>
<p style="color:var(--ink2)">Pick the label you believe is right. The model answers are shown for reference only; they can both be wrong.
A relation means the passage itself says how one is used for / with the other. Mere co-mention, "data is needed for AI", or a format that
is never called an ontology are <b>No relation</b>. {len(pick)} passages. Your choices are saved in this browser as you go.</p>
{''.join(cards)}
<script>
const KEY="gold-link-v1";let s=JSON.parse(localStorage.getItem(KEY)||"{{}}");
document.querySelectorAll("section").forEach(sec=>{{const c=sec.dataset.cid;sec.querySelectorAll("input").forEach(i=>{{
 if(s[c]===i.value)i.checked=true;i.onchange=()=>{{s[c]=i.value;localStorage.setItem(KEY,JSON.stringify(s));cnt()}}}})}});
function cnt(){{document.getElementById("n").textContent=" "+Object.keys(s).length+" / {len(pick)} labelled "}}cnt();
function dl(){{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(s,null,1)],{{type:"application/json"}}));
 a.download="gold_link_labels.json";a.click()}}
</script></body></html>"""
    path.write_text(page)
    return len(pick)


__all__ = ["load_items", "agreement", "item_status", "talk_tiers", "tiers_by_event", "review_markdown", "gold_html", "json"]
