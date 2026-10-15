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


def load_taxonomy(path: Path | str) -> Taxonomy:
    raw = yaml.safe_load(Path(path).read_text())
    if not str(raw.get("status", "")).strip().upper().startswith("APPROVED"):
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
