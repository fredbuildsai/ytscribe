"""Stage 2: judge ONE short passage at a time. Candidate passages come from deterministic co-occurrence (an AI term and an
ontology / semantic term within ~120 words); the model sees only that passage, so it cannot infer a link from the rest of the
talk (the failure mode of the whole-talk pass). Each passage is judged independently by two different model families; only
agreement counts as an explicit, confirmed relation."""

import hashlib
from dataclasses import dataclass
from typing import Literal

import regex
from pydantic import BaseModel, Field

from ytscribe.analysis.config import TopicsConfig
from ytscribe.analysis.corpus import Doc
from ytscribe.analysis.llm.link import TOPIC_A, TOPIC_B
from ytscribe.analysis.llm.verify import normalize, quote_found

PROMPT_VERSION = "link-judge-v1"
KIND = "link_judge"

SYSTEM = f"""You code short passages from conference talk transcripts for a research study. For each passage you answer ONE question: \
does the speaker, IN THIS PASSAGE, explicitly relate TOPIC A to TOPIC B?

TOPIC A, {TOPIC_A[0]}: {TOPIC_A[1]}
TOPIC B, {TOPIC_B[0]}: {TOPIC_B[1]}

Rules
1. Use ONLY the passage. Do not use outside knowledge and do not guess what the speaker means elsewhere.
2. The passage is machine-generated captions (names may be misspelled, no punctuation); read through the errors.
3. A relation requires that the passage ITSELF says how one topic is used for, built on, improved by, constrained by, or applied to \
the other, AND that both topics are named or unmistakably referred to in the passage. These are NOT relations: both topics merely \
appearing near each other; general statements that data, data quality, standards, file formats or metadata are needed for AI; a \
standard or format (ADF, ASM...) that the speaker does not call an ontology or a semantic model.
4. "quote" must be an EXACT quote from the passage, at most 40 words, containing the words that state the relation. You may shorten \
it ONLY with "...". If relation is "no_relation", quote must be "".
5. Reply with a single JSON object and nothing else.

relation is exactly one of
  "ontology_supports_ai" = ontologies / semantic models provide context, grounding, knowledge, constraints or meaning for AI / LLMs / agents
  "ai_supports_ontology" = AI / LLMs are used to create, extend, map, validate, query or maintain ontologies / semantic models
  "other_link"           = another explicit relation (say which in "reason")
  "no_relation"          = anything else, including doubt
Fields: {{"relation": ..., "quote": "...", "reason": "<one sentence>"}}"""

USER = """PASSAGE (from the talk "{title}", {event}):
\"\"\"{passage}\"\"\"

Return the JSON object now."""

Relation = Literal["ontology_supports_ai", "ai_supports_ontology", "other_link", "no_relation"]


class Judgement(BaseModel):
    relation: Relation
    quote: str = Field(default="", max_length=400)
    reason: str = Field(default="", max_length=400)


CONNECTED = ("ontology_supports_ai", "ai_supports_ontology", "other_link")


def prompt_hash() -> str:
    return hashlib.sha256((SYSTEM + USER).encode()).hexdigest()[:12]


def render_prompt() -> str:
    return f"[SYSTEM]\n{SYSTEM}\n\n[USER]\n{USER.format(title='<talk title>', event='<workshop>', passage='<~120-word passage>')}"


@dataclass
class Candidate:
    cid: str  # "<video_id>:<n>"
    video_id: str
    event: str
    title: str
    text: str


def candidates(docs: list[Doc], patterns: dict, cfg: TopicsConfig) -> list[Candidate]:
    """Windows around every A-term mention that also contain a B-term (topic lexicon + b_extra). Overlapping windows are merged."""
    lc = cfg.link
    pa = patterns[lc.a_topic]
    pb = regex.compile("|".join([patterns[lc.b_topic].pattern] + [f"(?:{x})" for x in lc.b_extra]), regex.IGNORECASE | regex.BESTMATCH)
    out: list[Candidate] = []
    for d in docs:
        toks = d.text.split()
        offs, pos = [], 0
        for t in toks:
            offs.append(pos)
            pos += len(t) + 1
        import bisect

        last_hi, n = -1, 0
        for m in pa.finditer(d.text):
            i = max(0, bisect.bisect_right(offs, m.start()) - 1)
            lo, hi = max(0, i - lc.window_words), min(len(toks), i + lc.window_words + 1)
            win = " ".join(toks[lo:hi])
            if not pb.search(win) or lo <= last_hi:
                continue
            last_hi = hi - 1
            out.append(Candidate(f"{d.video_id}:{n}", d.video_id, d.event, d.title, win))
            n += 1
    return out


def make_validator(passage: str):
    from llmrouter_free import json_validator

    base = json_validator(Judgement)
    pn = normalize(passage)

    def validate(text: str) -> Judgement:
        j = base(text)
        if j.relation != "no_relation" and not quote_found(j.quote, pn):
            raise ValueError("a relation was claimed with a quote that is not in the passage")
        if j.relation == "no_relation" and j.quote.strip():
            j.quote = ""
        return j

    return validate


def judge(router, route: str, c: Candidate, max_tokens: int = 350) -> dict:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(title=c.title, event=c.event, passage=c.text)}]
    res = router.complete(route, msgs, validate=make_validator(c.text), temperature=0.0, max_tokens=max_tokens, json_mode=True,
                          use_cache=True)
    j: Judgement = res.parsed
    return {"cid": c.cid, "relation": j.relation, "quote": j.quote, "reason": j.reason, "model": res.model.split("/")[-1]}
