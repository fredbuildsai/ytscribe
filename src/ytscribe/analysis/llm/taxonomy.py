"""The approved taxonomy the LLM is allowed to use. A draft (data/analysis/discover/taxonomy_draft.yaml) must be edited and its
`status` changed to start with APPROVED before anything here will load it: the LLM never sees an unreviewed taxonomy."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class TaxonomyNotApproved(RuntimeError):
    pass


@dataclass
class Taxonomy:
    topics: dict[str, str]  # id -> "label: description"
    anchors: dict[str, str]  # id -> label (fixed lexicon topics, always tracked)
    version: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def topic_ids(self) -> set[str]:
        return set(self.topics) | set(self.anchors)

    def describe(self) -> str:
        lines = [f"- {i}: {d}" for i, d in {**self.anchors, **self.topics}.items()]
        return "\n".join(lines)


class TaxonomyIncomplete(ValueError):
    """The file still has null keys the reviewer has to fill."""


def _is_approved(raw: dict) -> bool:
    return str(raw.get("status") or "").strip().upper().startswith("APPROVED")


def missing_keys(raw: dict) -> list[str]:
    """Every key the reviewer must still fill (null), as readable paths. Empty list = nothing left to do."""
    out = [] if raw.get("status") else ["status"]
    for c in raw.get("categories", []):
        out += [f"{c['id']}.{k}" for k in ("label", "description", "keep") if c.get(k) is None]
    for t in raw.get("unassigned_themes", []):
        if t.get("decision") is None:
            out.append(f"{t['id']}.decision")
    return out


def load_taxonomy(path: Path | str) -> Taxonomy:
    raw = yaml.safe_load(Path(path).read_text())
    if "categories" in raw:  # reviewer-fill format (taxonomy_proposed.yaml)
        todo = missing_keys(raw)
        if todo:
            raise TaxonomyIncomplete(f"{path}: {len(todo)} keys are still null: {', '.join(todo[:12])}{' ...' if len(todo) > 12 else ''}")
        if not _is_approved(raw):
            raise TaxonomyNotApproved(f"{path}: status is {raw.get('status')!r}; it must start with APPROVED")
        kept = {c["id"]: c for c in raw["categories"] if c["keep"] is True}
        topics = {cid: f"{c['label']}: {c['description'].strip()}" for cid, c in kept.items()}
        for t in raw.get("unassigned_themes", []):
            d = str(t["decision"]).strip()
            if d.lower() != "drop" and d not in kept:
                raise ValueError(f"{t['id']}.decision is {d!r}: use 'drop' or the id of a category you kept")
        anchors = {a["id"]: a["label"] for a in raw.get("anchors", [])}
        return Taxonomy(topics=topics, anchors=anchors, version=str(raw["status"]), raw=raw)
    if not _is_approved(raw):
        raise TaxonomyNotApproved(f"{path}: status is {str(raw.get('status', ''))[:60]!r}; a human must review the draft and "
                                  "set `status: APPROVED ...` before the LLM layer will use it")
    topics: dict[str, str] = {}
    merges: dict[str, str] = {}
    for t in raw.get("topics", []):
        if not t.get("keep", True):
            continue
        if t.get("merge_into"):
            merges[t["id"]] = t["merge_into"]
            continue
        desc = (t.get("description") or "").strip()
        topics[t["id"]] = f"{t['label']}" + (f": {desc}" if desc else "")
    for src, dst in merges.items():  # merged topics must point at a kept topic
        if dst not in topics:
            raise ValueError(f"{src} merges into {dst}, which is not a kept topic")
    anchors = {a["id"]: a["label"] for a in raw.get("anchors", [])}
    return Taxonomy(topics=topics, anchors=anchors, version=str(raw.get("version", raw.get("status", ""))), raw=raw)


def build_proposal(mapping_yaml: Path | str, draft_yaml: Path | str, passage_topics_csv: Path | str, titles: dict[str, str],
                   out_path: Path | str, season_labels: dict[str, str] | None = None) -> dict:
    """Write the reviewer-fill YAML: 17 categories, every key the reviewer must set is null, evidence shown for context."""
    import json
    import re

    import pandas as pd

    mp = yaml.safe_load(Path(mapping_yaml).read_text())
    draft = yaml.safe_load(Path(draft_yaml).read_text())
    pt = pd.read_csv(passage_topics_csv)
    labels = season_labels or {}

    def ev_label(e: str) -> str:
        for season, lab in labels.items():
            e = re.sub(rf"\b{season}\b", lab, e)
        return e

    pt["event"] = pt["event"].map(ev_label)
    totals = pt.groupby("event").size()
    theme_terms = {t["id"]: t["top_terms"] for t in draft["topics"]}
    all_themes = {t["id"] for t in draft["topics"]}
    placed = {t for c in mp["categories"] for t in c["themes"]} | set(mp.get("unassigned_themes", {}))
    if placed != all_themes:
        raise ValueError(f"themes not placed anywhere: {sorted(all_themes - placed)}; unknown: {sorted(placed - all_themes)}")

    def evidence(theme_ids: list[str]) -> dict:
        sub = pt[pt["topic"].isin([int(t[1:]) for t in theme_ids])]
        share = (sub.groupby("event").size() / totals).dropna()
        terms: dict[str, float] = {}
        for t in theme_ids:
            n = int((pt["topic"] == int(t[1:])).sum())
            for rank, w in enumerate(theme_terms[t][:6]):
                terms[w] = terms.get(w, 0) + n / (rank + 1)
        top_talks = sub["video_id"].value_counts().head(3)
        return {"themes": theme_ids, "passages": len(sub), "talks": int(sub["video_id"].nunique()),
                "peak_event": share.idxmax() if len(share) else None,
                "top_terms": [w for w, _ in sorted(terms.items(), key=lambda kv: -kv[1])[:8]],
                "example_talks": [re.sub(r"^\d{4} (Spring|Fall) AC\s*[-]?\s*", "", titles.get(v, v))[:70] for v in top_talks.index]}

    q = json.dumps  # a JSON string is a valid YAML double-quoted scalar
    L = ["# TAXONOMY PROPOSAL: 17 categories made from the 46 discovered themes. Fill every key that is null; nothing else needs editing.",
         "#   status       -> write \"APPROVED <initials/date>\" when you are done (the LLM layer refuses to load the file otherwise)",
         "#   label        -> your final category name (copy suggested_label or change it)",
         "#   description  -> one or two sentences saying what belongs in the category (the model reads this to decide)",
         "#   keep         -> true to use the category, false to drop it",
         "#   decision     -> for the two unassigned themes: a category id such as C16, or \"drop\"",
         "# Check your progress with:  uv run ys llm taxonomy-check data/analysis/discover/taxonomy_proposed.yaml", "",
         "status: null", "", "anchors:   # fixed lexicon topics that are always tracked: no action needed"]
    L += [f"  - {{id: {a['id']}, label: {q(a['label'])}, group: {a['group']}}}" for a in draft["anchors"]]
    L += ["", "categories:"]
    for c in mp["categories"]:
        e = evidence(c["themes"])
        L += [f"  - id: {c['id']}", f"    suggested_label: {q(c['suggested_label'])}", "    label: null", "    description: null",
              "    keep: null", "    context:   # for reference only",
              f"      themes: [{', '.join(e['themes'])}]", f"      passages: {e['passages']}", f"      talks: {e['talks']}",
              f"      peak_event: {q(e['peak_event'])}", f"      top_terms: {q(e['top_terms'])}", f"      example_talks: {q(e['example_talks'])}", ""]
    L += ["unassigned_themes:   # left out of the categories on purpose; say where each belongs, or drop it"]
    for tid, why in mp["unassigned_themes"].items():
        e = evidence([tid])
        L += [f"  - id: {tid}", f"    why: {q(why)}", "    decision: null", f"    context: {{passages: {e['passages']}, talks: {e['talks']}, "
              f"top_terms: {q(e['top_terms'][:6])}}}"]
    Path(out_path).write_text("\n".join(L) + "\n")
    return {"categories": len(mp["categories"]), "unassigned": len(mp["unassigned_themes"]), "nulls": len(missing_keys(yaml.safe_load(
        Path(out_path).read_text())))}
