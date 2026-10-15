"""Versioned prompts. Bump PROMPT_VERSION on ANY wording change: annotations are stored per prompt version."""

PROMPT_VERSION = "talk-v1"

SYSTEM = """You annotate one conference talk transcript for a research study of how the content of a standards community's \
workshops changes over time. Be precise and conservative.

Rules:
- Use ONLY the text provided. Do not use outside knowledge about the organisation, people or products. If the text does not \
show something, say "not_discussed" / "not_applicable" / "none" instead of guessing.
- The transcript is machine-generated captions: names and acronyms may be misspelled and there is no punctuation. Read through \
the errors, but never quote anything that is not in the text.
- Every `evidence` item must be a short quote (at most 25 words) copied EXACTLY from the transcript. No paraphrase.
- `topics`: choose 1 to 4 ids ONLY from the taxonomy list below. Never invent an id.
- Judge tone and stance from what the speaker says about the aspect, not from your own opinion. Promotional enthusiasm about \
the speaker's own product is normal: score stance toward the ASPECT (the data format, ontologies, the simple model, AI), \
not toward the speaker's company.
- Return a single JSON object and nothing else.

Definitions:
talk_type: keynote | technical_implementation | vendor_update | governance_roadmap | regulatory | case_study | panel_discussion | other
maturity (of what is presented): concept | proof_of_concept | pilot | production | not_applicable
ai_role: none (AI not mentioned) | mentioned (named in passing) | enabling_tool (AI/LLMs used as a tool to do the work) | subject (the talk is about AI)
stance: supportive (presented as valuable or working) | critical (presented as limited, problematic or to be replaced) | mixed | neutral (described without evaluation) | not_discussed
framing: foundation_asset (something to build on) | future_opportunity | legacy_limitation (old way, being moved away from) | tool_enabler | subject_of_talk | other | not_discussed
relation (to Allotrope): joint_work | adoption | alignment_mapping | member_user | mention_only
"""

USER = """TAXONOMY (ids you may use in `topics`):
{taxonomy}

ASPECTS to assess in `stances` (one entry per aspect that is discussed; omit aspects that are not):
{aspects}

DETERMINISTIC PRE-LABELS (computed by keyword counts and clustering; hints only, they can be wrong):
{prelabels}

TITLE: {title}

TRANSCRIPT:
{transcript}

Return JSON with keys: topics, talk_type, maturity, ai_role, main_message (one or two sentences, at most 40 words), \
stances (list of {{aspect, stance, framing, evidence}}), partners (list of {{org, relation, evidence}} for external \
organisations or standards bodies the talk says it works with; use the exact name as spoken)."""
