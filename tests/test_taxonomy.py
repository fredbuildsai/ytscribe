import pandas as pd
import pytest
import yaml

from ytscribe.analysis.llm import taxonomy as tx

MAPPING = {"categories": [{"id": "C01", "suggested_label": "Alpha", "themes": ["T00", "T01"]},
                          {"id": "C02", "suggested_label": "Beta", "themes": ["T02"]}],
           "unassigned_themes": {"T03": "meta talk"}}
DRAFT = {"anchors": [{"id": "A01", "label": "ADF", "group": "direction"}],
         "topics": [{"id": f"T0{i}", "top_terms": [f"w{i}a", f"w{i}b"]} for i in range(4)]}


def build(tmp_path):
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(MAPPING))
    (tmp_path / "d.yaml").write_text(yaml.safe_dump(DRAFT))
    pd.DataFrame({"video_id": list("aabbccdd"), "event": ["2026 Fall"] * 4 + ["2024 Spring"] * 4,
                  "topic": [0, 1, 2, -1, 0, 3, 1, 2]}).to_csv(tmp_path / "pt.csv", index=False)
    out = tmp_path / "proposed.yaml"
    res = tx.build_proposal(tmp_path / "m.yaml", tmp_path / "d.yaml", tmp_path / "pt.csv", {"a": "2026 Fall AC - Talk A"}, out,
                            {"Spring": "H1", "Fall": "H2"})
    return out, res


def test_proposal_has_exactly_the_reviewer_keys_as_null_and_context_filled(tmp_path):
    out, res = build(tmp_path)
    raw = yaml.safe_load(out.read_text())
    assert res == {"categories": 2, "unassigned": 1, "nulls": 1 + 2 * 3 + 1}      # status + 3 per category + 1 decision
    c1 = raw["categories"][0]
    assert c1["label"] is None and c1["description"] is None and c1["keep"] is None and raw["status"] is None
    assert c1["suggested_label"] == "Alpha" and c1["context"]["themes"] == ["T00", "T01"]
    assert c1["context"]["passages"] == 4 and c1["context"]["peak_event"] in ("2026 H2", "2024 H1")   # H1/H2 labels
    assert raw["unassigned_themes"][0]["decision"] is None
    assert "Talk A" in " ".join(c1["context"]["example_talks"])


def test_every_discovered_theme_must_be_placed_exactly_once(tmp_path):
    (tmp_path / "m.yaml").write_text(yaml.safe_dump({**MAPPING, "unassigned_themes": {}}))
    (tmp_path / "d.yaml").write_text(yaml.safe_dump(DRAFT))
    pd.DataFrame({"video_id": ["a"], "event": ["2026 Fall"], "topic": [0]}).to_csv(tmp_path / "pt.csv", index=False)
    with pytest.raises(ValueError, match="not placed"):
        tx.build_proposal(tmp_path / "m.yaml", tmp_path / "d.yaml", tmp_path / "pt.csv", {}, tmp_path / "o.yaml")


def test_loader_refuses_incomplete_then_unapproved_then_accepts(tmp_path):
    out, _ = build(tmp_path)
    with pytest.raises(tx.TaxonomyIncomplete, match="status"):
        tx.load_taxonomy(out)
    raw = yaml.safe_load(out.read_text())
    for c in raw["categories"]:
        c.update(label=c["suggested_label"], description="what belongs here", keep=True)
    raw["unassigned_themes"][0]["decision"] = "drop"
    raw["status"] = "DRAFT still"
    f = tmp_path / "filled.yaml"
    f.write_text(yaml.safe_dump(raw))
    with pytest.raises(tx.TaxonomyNotApproved):
        tx.load_taxonomy(f)
    raw["status"] = "APPROVED fc 2026-10-15"
    f.write_text(yaml.safe_dump(raw))
    t = tx.load_taxonomy(f)
    assert set(t.topics) == {"C01", "C02"} and t.topics["C01"].startswith("Alpha: what belongs") and "A01" in t.anchors


def test_loader_respects_keep_false_and_validates_decisions(tmp_path):
    out, _ = build(tmp_path)
    raw = yaml.safe_load(out.read_text())
    for c in raw["categories"]:
        c.update(label="L", description="d", keep=True)
    raw["categories"][1]["keep"] = False
    raw["status"] = "APPROVED x"
    raw["unassigned_themes"][0]["decision"] = "C02"          # points at a category that was dropped
    f = tmp_path / "f.yaml"
    f.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="kept"):
        tx.load_taxonomy(f)
    raw["unassigned_themes"][0]["decision"] = "C01"
    f.write_text(yaml.safe_dump(raw))
    assert set(tx.load_taxonomy(f).topics) == {"C01"}


def test_old_draft_format_still_loads_when_approved(tmp_path):
    f = tmp_path / "old.yaml"
    f.write_text(yaml.safe_dump({"status": "APPROVED ok", "anchors": [], "topics": [
        {"id": "T00", "label": "A", "description": "d", "keep": True}, {"id": "T01", "label": "B", "keep": True, "merge_into": "T00"}]}))
    assert set(tx.load_taxonomy(f).topics) == {"T00"}
