# Roadmap / open items

## HTML tool to configure a new topic (design TBD)
Requested 2026-10-15. A simple HTML-based tool to make it easy to set up a new topic (today: hand-edit
`configs/topics/*.yaml`). **Design is to be determined**; do not start building before the design is agreed.

Context for whoever designs it:
- A topic = a name + one or more regex patterns, compiled with the `regex` module (fuzzy matching, e.g. `{e<=1}`, is how
  we absorb YouTube auto-caption garbling such as "pistolia" / "chat gbt").
- Rule: topic terms must be neutral (names, acronyms, technologies), never subjective words ("hard", "easy"). Tone is
  measured separately by sentiment on the text around each mention.
- A useful tool would likely need: live match counts and sample hits against the existing corpus (per event), the
  spellings the captions actually produce (to propose fuzzy variants), a check for subjective words, and export to the
  YAML format. Group membership (direction / partners / ai_cloud) and the projection era settings live in the same file.
