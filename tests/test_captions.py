from pathlib import Path

from ytscribe.captions import parse_vtt, pick_track
from ytscribe.settings import CaptionsCfg

FX = Path(__file__).parent / "fixtures"


def test_manual_vtt_keeps_cue_text():
    segs = parse_vtt(FX / "manual.vtt", auto=False)
    assert [s.text for s in segs] == ["Welcome to Allotrope Connect.", "Today we discuss the Allotrope Simple Model."]
    assert segs[0].start_s == 1.0 and segs[1].end_s == 6.0


def test_auto_vtt_rollup_is_deduplicated():
    segs = parse_vtt(FX / "auto.vtt", auto=True)
    text = " ".join(s.text for s in segs)
    assert text == "welcome to allotrope connect everyone today we talk"
    assert all(a.end_s <= b.start_s + 1e-6 for a, b in zip(segs, segs[1:], strict=False))


def test_pick_track_prefers_manual_then_auto():
    cfg = CaptionsCfg()
    assert pick_track({"subtitles": {"en": [1]}, "automatic_captions": {"en": [1]}}, cfg) == ("manual", "en")
    assert pick_track({"subtitles": {}, "automatic_captions": {"en": [1]}}, cfg) == ("auto", "en")
    assert pick_track({"subtitles": {"fr": [1]}, "automatic_captions": {"fr": [1]}}, cfg) is None
    assert pick_track({"automatic_captions": {"en-CA": [1]}}, cfg) == ("auto", "en-CA")
    assert pick_track({"subtitles": {"en": [1]}, "automatic_captions": {"en": [1]}},
                      CaptionsCfg(prefer_manual=False)) == ("auto", "en")
