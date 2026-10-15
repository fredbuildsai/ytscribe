"""What the LLM returns for one talk. Enumerations keep answers comparable across talks and events."""

from typing import Literal

from pydantic import BaseModel, Field

TalkType = Literal["keynote", "technical_implementation", "vendor_update", "governance_roadmap", "regulatory",
                   "case_study", "panel_discussion", "other"]
Maturity = Literal["concept", "proof_of_concept", "pilot", "production", "not_applicable"]
AiRole = Literal["none", "mentioned", "enabling_tool", "subject"]
Stance = Literal["supportive", "neutral", "critical", "mixed", "not_discussed"]
Framing = Literal["foundation_asset", "future_opportunity", "legacy_limitation", "tool_enabler", "subject_of_talk",
                  "other", "not_discussed"]
Relation = Literal["joint_work", "adoption", "alignment_mapping", "member_user", "mention_only"]


class AspectStance(BaseModel):
    aspect: str  # one of the anchor aspects (ADF, ontologies, ASM, AI), validated against the config
    stance: Stance
    framing: Framing
    evidence: list[str] = Field(default_factory=list, description="verbatim quotes, each at most 25 words")


class PartnerRelation(BaseModel):
    org: str
    relation: Relation
    evidence: list[str] = Field(default_factory=list)


class TalkAnnotation(BaseModel):
    topics: list[str] = Field(min_length=1, max_length=4, description="taxonomy ids")
    talk_type: TalkType
    maturity: Maturity
    ai_role: AiRole
    main_message: str = Field(max_length=400)
    stances: list[AspectStance] = Field(default_factory=list)
    partners: list[PartnerRelation] = Field(default_factory=list)
