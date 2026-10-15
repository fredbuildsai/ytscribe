"""Focused LLM question: does the speaker CONNECT two topics (here AI and ontologies), and how?

Why this exists: counting words shows two topics appear in the same talk or passage; it cannot tell whether the speaker relates
them, or relates them without using the keywords. The model gets ONLY the transcript (no keyword counts, no clusters), so its
answer is an independent check of the word-based result. Every claim needs an exact quote, verified afterwards against the
transcript. The full prompt text is the constants below (PROMPT_VERSION changes with any wording change)."""

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ytscribe.analysis.corpus import Doc
from ytscribe.analysis.llm.verify import normalize, quote_found

PROMPT_VERSION = "link-ai-ontology-v2"
KIND = "link:ai_ontology"

TOPIC_A = ("AI", "artificial intelligence in any form: machine learning, deep learning, large language models (LLMs) and generative AI, "
           "chatbots and AI agents, named AI products or companies (for example ChatGPT, Claude, OpenAI, Anthropic, Copilot), and "
           "protocols for AI agents (for example MCP).")
TOPIC_B = ("ONTOLOGY", "ontologies and semantic data models: formal ontologies (OWL, RDF; for example the Allotrope Foundation "
           "Ontologies, AFO), knowledge graphs, controlled vocabularies and taxonomies, semantic layers, and any structured, "
           "machine-readable definition of what data MEANS. A file format or database schema counts only if the speaker describes it "
           "as semantic or ontology-based.")

SYSTEM = f"""You are a careful research assistant who codes conference talk transcripts for a study of how a standards community's \
workshops change over time. For each talk you answer ONE question: does the speaker connect two topics, and how?

TOPIC A, {TOPIC_A[0]}: {TOPIC_A[1]}
TOPIC B, {TOPIC_B[0]}: {TOPIC_B[1]}

Rules
1. Use ONLY the transcript you are given. Do not use outside knowledge about the people, products or organisation. Do not expand \
or "correct" acronyms unless the transcript itself does.
2. The transcript is machine-generated captions: names and acronyms may be misspelled and there is no punctuation. Read through \
the errors. Lines that start with "[slide @ mm:ss]" are text read off the slides, not speech; treat them as part of the talk.
3. Be conservative. Two topics appearing in the same talk is NOT a connection. A connection requires the speaker to explicitly \
relate them: one is used for, built on, improved by, constrained by, or applied to the other. Example of what is NOT a connection: \
an AI assistant that converts instrument data into a standard file format, when the speaker never calls that standard an ontology \
or a semantic model.
4. Every item in "evidence" must be an EXACT quote copied from the transcript, at most 30 words. You may shorten a long sentence \
ONLY by writing "..." in place of the words you leave out. Never paraphrase inside a quote.
5. The "connection" quotes must themselves show the speaker relating A and B (together they must mention or clearly refer to both). \
If you cannot find such a quote, the answer is "parallel" (both present) or "none", not a connection.
6. If the talk does not support an answer, choose "none". Do not guess.
7. Reply with a single JSON object and nothing else.

Answer these fields
- "a_coverage": how much the talk covers TOPIC A (AI): "none" = not present; "mentioned" = named in passing; "substantial" = a meaningful \
part of the talk.
- "b_coverage": the same for TOPIC B (ONTOLOGY).
- "connection": exactly one of
    "none"                 = at most one of the two topics is present
    "parallel"             = both are present but the speaker never relates them
    "ontology_supports_ai" = ontologies / semantic models are described as providing context, grounding, knowledge, constraints or \
meaning for AI, LLMs or agents
    "ai_supports_ontology" = AI / LLMs are used to create, extend, map, validate, query or maintain ontologies / semantic models
    "other_link"           = the speaker explicitly relates them in some other way (say how in "rationale")
- "evidence": a list of up to 4 objects {{"claim": "a" | "b" | "connection", "quote": "<exact quote>"}}. Give at least one quote for \
every coverage above "none" and at least one "connection" quote whenever "connection" is not "none" or "parallel".
- "confidence": "low", "medium" or "high" in your connection answer.
- "rationale": one or two sentences explaining the connection answer."""

USER = """TITLE: {title}
WORKSHOP: {event}

TRANSCRIPT:
{transcript}

Return the JSON object now."""

Coverage = Literal["none", "mentioned", "substantial"]
Connection = Literal["none", "parallel", "ontology_supports_ai", "ai_supports_ontology", "other_link"]


class Evidence(BaseModel):
    claim: Literal["a", "b", "connection"]
    quote: str


class LinkAssessment(BaseModel):
    a_coverage: Coverage
    b_coverage: Coverage
    connection: Connection
    evidence: list[Evidence] = Field(default_factory=list, max_length=8)
    confidence: Literal["low", "medium", "high"]
    rationale: str = Field(max_length=700)

    @model_validator(mode="after")
    def consistent(self):
        both = self.a_coverage != "none" and self.b_coverage != "none"
        if self.connection in ("ontology_supports_ai", "ai_supports_ontology", "other_link", "parallel") and not both:
            raise ValueError("a connection or 'parallel' requires both topics to be present")
        return self


def prompt_hash() -> str:
    return hashlib.sha256((SYSTEM + USER).encode()).hexdigest()[:12]


def render_prompt(title: str = "<talk title>", event: str = "<workshop>", transcript: str = "<full transcript text>") -> str:
    return f"[SYSTEM]\n{SYSTEM}\n\n[USER]\n{USER.format(title=title, event=event, transcript=transcript)}"


def transcript_for(doc: Doc, max_words: int = 9500) -> tuple[str, bool]:
    """The doc text as the model sees it; very long talks are cut (flagged), none currently exceed the limit."""
    toks = doc.text.split()
    return (" ".join(toks[:max_words]), len(toks) > max_words)


@dataclass
class LinkResult:
    video_id: str
    data: dict
    deployment: str
    model: str
    verified_ratio: float
    truncated: bool
    tokens_in: int = 0
    tokens_out: int = 0


def verify_evidence(a: LinkAssessment, transcript: str) -> tuple[dict, float]:
    tn = normalize(transcript)
    data = a.model_dump()
    kept, ok = [], 0
    for e in data["evidence"]:
        found = quote_found(e["quote"], tn)
        ok += found
        kept.append({**e, "verified": found})
    data["evidence"] = kept
    return data, (ok / len(kept) if kept else 1.0)


CONNECTED = ("ontology_supports_ai", "ai_supports_ontology", "other_link")


def make_validator(transcript: str):
    """Router validate-callback: schema check, then 'a claimed connection must have a quote that verifies against the
    transcript'. Raising ValueError makes the router retry (another attempt / deployment) instead of storing an unsupported claim."""
    from llmrouter_free import json_validator

    base = json_validator(LinkAssessment)

    def validate(text: str) -> LinkAssessment:
        parsed = base(text)
        if parsed.connection in CONNECTED:
            data, _ = verify_evidence(parsed, transcript)
            if not any(e["claim"] == "connection" and e["verified"] for e in data["evidence"]):
                raise ValueError("connection claimed without a connection quote that verifies against the transcript")
        return parsed

    return validate


def assess_talk(router, doc: Doc, route: str = "link", temperature: float = 0.0, max_tokens: int = 900,
                scope: str = "both") -> LinkResult:
    text, truncated = transcript_for(doc)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(title=doc.title, event=doc.event, transcript=text)}]
    res = router.complete(route, msgs, validate=make_validator(text), temperature=temperature, max_tokens=max_tokens,
                          json_mode=True, use_cache=True)
    parsed: LinkAssessment = res.parsed
    data, ratio = verify_evidence(parsed, text)
    data["prompt_version"] = PROMPT_VERSION
    data["prompt_hash"] = prompt_hash()
    data["scope"] = scope  # "both" = both topics present by keyword; "audit" = random talk that was NOT keyword-selected
    return LinkResult(doc.video_id, data, res.deployment, res.model, ratio, truncated, res.tokens_in, res.tokens_out)


def dumps(r: LinkResult) -> str:
    return json.dumps(r.data, ensure_ascii=False)
