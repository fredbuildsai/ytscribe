import json

import pytest

from ytscribe.analysis.corpus import Doc
from ytscribe.analysis.llm import link as lk
from ytscribe.analysis.llm.verify import normalize, quote_found

TRANSCRIPT = ("the first part will deal with a machine learning life cycle ontology which as the name implies is an ontology "
              "that aims to represent the actual machine learning artifacts but also in the new time to also enable grounding "
              "for AI agents and finally a semantically consistent way to represent connections to the domain")


def test_quote_check_accepts_trimmed_true_quotes_and_rejects_invented_ones():
    tn = normalize(TRANSCRIPT)
    assert quote_found("Ontologies... enable grounding for AI agents.", tn)                       # ellipsis, punctuation, caps
    assert quote_found("a machine learning life cycle ontology which aims to represent the actual machine learning artifacts", tn)
    assert not quote_found("Ontologies give large language models the semantic backbone they need to reason reliably.", tn)
    assert not quote_found("hi", tn)


def doc():
    return Doc(video_id="v1", title="T", upload_date="2026-05-19", event_year=2026, event_season="Spring", t=2026.25,
               event="2026 H1", origin="captions:auto:en", slide_only=False, description="", text=TRANSCRIPT, duration_min=20.0)


def reply(connection="ontology_supports_ai", quote="enable grounding for AI agents", a="substantial", b="substantial"):
    return json.dumps({"a_coverage": a, "b_coverage": b, "connection": connection, "confidence": "high", "rationale": "r",
                       "evidence": [{"claim": "connection", "quote": quote}]})


def test_validator_rejects_an_unsupported_connection_and_inconsistent_answers():
    pytest.importorskip("llmrouter_free")  # optional sibling dependency (see README)
    v = lk.make_validator(TRANSCRIPT)
    assert v(reply()).connection == "ontology_supports_ai"
    with pytest.raises(ValueError, match="verifies"):
        v(reply(quote="the ontology lets the chatbot answer every question perfectly"))
    with pytest.raises(ValueError):                           # a connection needs both topics present
        v(reply(a="none"))
    assert v(reply(connection="none", a="none", b="substantial", quote="enable grounding for AI agents")).connection == "none"


class FakeRouter:
    def __init__(self, texts):
        self.texts, self.calls = list(texts), []

    def complete(self, route, messages, validate=None, **kw):
        self.calls.append((route, messages, kw))
        text = self.texts.pop(0)
        from types import SimpleNamespace

        return SimpleNamespace(parsed=validate(text), deployment="fake", model="fake/model", tokens_in=10, tokens_out=5)


def test_assess_talk_sends_only_the_transcript_and_records_prompt_version_and_verified_quotes():
    pytest.importorskip("llmrouter_free")
    r = FakeRouter([reply()])
    out = lk.assess_talk(r, doc(), scope="both")
    msgs = r.calls[0][1]
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == lk.SYSTEM
    assert "grounding for AI agents" in msgs[1]["content"] and "keyword" not in msgs[1]["content"].lower()  # no pre-labels leak in
    assert out.data["connection"] == "ontology_supports_ai" and out.verified_ratio == 1.0
    assert out.data["prompt_version"] == lk.PROMPT_VERSION and out.data["scope"] == "both"
    assert r.calls[0][2]["temperature"] == 0.0


def test_prompt_is_printable_and_hash_is_stable():
    text = lk.render_prompt()
    assert "[SYSTEM]" in text and "[USER]" in text and "TOPIC A, AI" in text and "TOPIC B, ONTOLOGY" in text
    assert lk.prompt_hash() == lk.prompt_hash() and lk.PROMPT_VERSION.endswith("v2")


def _talk_rows():
    import pandas as pd

    def r(vid, ev, conn, scope="both", a="substantial", b="substantial", kw_ai=3, kw_ont=3):
        return {"video_id": vid, "event": ev, "t": 2026.25 if ev == "2026 H1" else 2024.75, "title": vid, "slide_only": False,
                "kw_ai": kw_ai, "kw_ont": kw_ont, "model": "m", "scope": scope, "connection": conn, "confidence": "high",
                "a_coverage": a, "b_coverage": b, "quotes_verified": 1.0, "rationale": "r", "connection_quote": "q"}
    return pd.DataFrame([r("a", "2026 H1", "ontology_supports_ai"), r("b", "2026 H1", "parallel"),
                         r("c", "2026 H1", "ai_supports_ontology"), r("d", "2026 H1", "not_sent", scope="", kw_ai=0, kw_ont=4),
                         r("e", "2024 H2", "none", scope="audit", a="mentioned", kw_ai=0, kw_ont=2),
                         r("f", "2024 H2", "not_sent", scope="", kw_ai=0, kw_ont=0)])


pd = __import__("pandas")


def test_by_event_never_counts_unsent_talks_as_the_models_none():
    from ytscribe.analysis.llm import link_report as lr

    ev = lr.by_event(_talk_rows()).set_index("event")
    h1 = ev.loc["2026 H1"]
    assert (h1.talks, h1.sent, h1.connected, h1.parallel, h1.not_sent) == (4, 3, 2, 1, 1)
    assert h1.pct_connected == 50.0 and h1.pct_both_present == 75.0
    h2 = ev.loc["2024 H2"]
    assert (h2.sent, h2["none"], h2.not_sent, h2.connected) == (1, 1, 1, 0)


def test_audit_counts_what_the_keyword_scope_would_miss():
    from ytscribe.analysis.llm import link_report as lr

    a = lr.audit(_talk_rows())
    assert a["audited"] == 1 and a["connection_found"] == 0 and a["model_saw_ai_but_no_keyword"] == 1
    assert lr.agreement_with_keywords(_talk_rows())["ai_model_yes_but_no_keyword"] == 1


def test_review_file_lists_only_connected_talks_with_their_quote():
    from ytscribe.analysis.llm import link_report as lr

    md = lr.review_markdown(_talk_rows(), "abc123")
    assert md.count("## ") == 2 and "ontology_supports_ai" in md and "ai_supports_ontology" in md and "parallel" not in md.split("##", 1)[1]


def test_judge_report_status_tiers_and_incomplete_items():
    import pandas as pd

    from ytscribe.analysis.llm import judge_report as jr

    items = pd.DataFrame([
        {"cid": "a:0", "video_id": "a", "rel_cloud": "ontology_supports_ai", "rel_local": "ontology_supports_ai", "text": "t"},
        {"cid": "b:0", "video_id": "b", "rel_cloud": "ontology_supports_ai", "rel_local": "no_relation", "text": "t"},
        {"cid": "c:0", "video_id": "c", "rel_cloud": "no_relation", "rel_local": "no_relation", "text": "t"},
        {"cid": "d:0", "video_id": "d", "rel_cloud": None, "rel_local": "ontology_supports_ai", "text": "t"}])
    assert list(jr.item_status(items)) == ["agree", "disagree", "none", "incomplete"]
    ag = jr.agreement(items)
    assert ag["n"] == 3 and ag["both_connected"] == 1 and ag["cloud_only"] == 1
    talks = pd.DataFrame({"video_id": list("abcde"), "event": "2026 H1", "t": 2026.25, "title": "x",
                          "connection": ["none", "none", "none", "ontology_supports_ai", "none"]})
    tiers = jr.talk_tiers(talks, items).set_index("video_id").tier
    assert tiers["a"] == "explicit_both_agree" and tiers["b"] == "explicit_models_disagree"
    assert tiers["c"] == "none" and tiers["d"] == "implied_only" and tiers["e"] == "none"
