from pathlib import Path

from ytscribe import pipeline
from ytscribe import ytdlp_runner as yt
from ytscribe.asr.backends.base import AsrSegment
from ytscribe.db.models import Segment, Video
from ytscribe.export import export_video
from ytscribe.pipeline import Runner, upsert_source
from ytscribe.settings import Config, Settings, SlidesCfg
from ytscribe.slides import Slide, merge_near_duplicates, select_slide_frames, stable_run_ends
from ytscribe.sources.resolve import Resolved, VideoRef


class H(int):
    """Stand-in for an imagehash value: `a - b` is the hamming distance."""

    def __sub__(self, other):
        return abs(int(self) - int(other))


def hashes(*vals):
    return [H(v) for v in vals]


def test_stable_runs_drop_transitions():
    # slide A (idx 0-3), fade (4), slide B (5-8), fade (9), slide C (10-12)
    h = hashes(0, 1, 0, 2, 100, 200, 201, 200, 202, 300, 400, 400, 401)
    ends = stable_run_ends(h, same_thresh=8, min_run=2)
    assert ends == [3, 8, 12]  # last frame of A, B, C; the singleton fade frames (idx 4 and 9) are dropped


def test_builds_of_one_slide_are_merged_keeping_the_later():
    h = hashes(0, 0, 10, 10, 500, 500)  # frames 0-1 slide, 2-3 same slide + one more bullet, 4-5 new slide
    kept = stable_run_ends(h, same_thresh=4, min_run=2)
    assert kept == [1, 3, 5]
    assert merge_near_duplicates(kept, h, dup_thresh=12) == [3, 5]


def test_select_slide_frames_uses_cfg_min_run_in_seconds():
    h = hashes(0, 0, 100, 200, 200)
    assert select_slide_frames(h, SlidesCfg(fps=1, min_run_s=2)) == [1, 4]
    assert select_slide_frames(h, SlidesCfg(fps=1, min_run_s=1)) == [1, 2, 4]  # singleton kept when allowed


def _setup(session, tmp_path, monkeypatch, asr_texts):
    cfg = Config()
    settings = Settings(data_dir=tmp_path / "data", logs_dir=tmp_path / "logs")
    upsert_source(session, Resolved("video", "https://x/v", "V", [VideoRef("a" * 11, "A")]))
    v = session.get(Video, "a" * 11)
    v.status, v.duration_s, v.origin = "done", 120.0, "asr:fake-q4"
    for i, t in enumerate(asr_texts):
        v.segments.append(Segment(start_s=i, end_s=i + 1, text=t, origin="asr:fake-q4"))
    session.flush()
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"x")
    monkeypatch.setattr(yt, "extract", lambda *a, **k: {})
    monkeypatch.setattr("ytscribe.video.download_video", lambda *a, **k: vid)
    frames = tmp_path / "f"
    frames.mkdir()

    def fake_extract(video, vid_id, dur, frames_dir, scfg):
        return [Slide(1, 5, 60, frames / "1.jpg", "Title slide\nBenchling", 0.99),
                Slide(2, 60, 120, frames / "2.jpg", "", 0.0),  # image-only slide: frame kept, no text segment
                Slide(3, 120, 130, frames / "3.jpg", "Key findings 70%", 0.9)]

    monkeypatch.setattr("ytscribe.slides.extract_slides", fake_extract)
    return v, Runner(cfg, settings, "reuse", session), settings


def test_run_slides_replaces_hallucinated_asr_and_exports_markers(session, tmp_path, monkeypatch):
    v, r, settings = _setup(session, tmp_path, monkeypatch, ["Thank you."] * 4)
    from loguru import logger

    info = r.run_slides(v, logger)
    assert info["frames"] == 3 and info["dropped_asr"] == 4
    assert [s.origin for s in v.segments] == ["slides:ocr:apple-vision"] * 2
    assert v.origin == "slides:ocr:apple-vision" and len(v.frames) == 3
    d = export_video(v, settings.data_dir)
    txt = (d / "transcript.txt").read_text()
    assert "[slide @ 00:05]\nTitle slide\nBenchling" in txt and "Thank you" not in txt
    assert len(__import__("json").loads((d / "slides.json").read_text())) == 3


def test_run_slides_keeps_real_speech(session, tmp_path, monkeypatch):
    v, r, _ = _setup(session, tmp_path, monkeypatch, ["welcome everyone", "to the talk about asm"])
    from loguru import logger

    info = r.run_slides(v, logger)
    assert info["dropped_asr"] == 0
    assert v.origin == "asr:fake-q4+slides:ocr:apple-vision"
    assert [s.origin for s in v.segments].count("asr:fake-q4") == 2


def test_rerun_replaces_old_slides_not_duplicates(session, tmp_path, monkeypatch):
    v, r, _ = _setup(session, tmp_path, monkeypatch, ["welcome everyone"])
    from loguru import logger

    r.run_slides(v, logger)
    r.run_slides(v, logger)
    assert len(v.frames) == 3 and sum(1 for s in v.segments if s.origin.startswith("slides")) == 2


def test_auto_slides_failure_keeps_transcript(session, tmp_path, monkeypatch):
    from ytscribe import captions
    from ytscribe.pipeline import run_batch, select_videos

    cfg = Config()
    cfg.ytdlp.sleep_between_videos_s = 0
    cfg.captions.api_fallback = False
    settings = Settings(data_dir=tmp_path / "data", logs_dir=tmp_path / "logs")
    upsert_source(session, Resolved("video", "https://x/v", "V", [VideoRef("a" * 11)]))
    monkeypatch.setattr(yt, "extract", lambda *a, **k: {"title": "T", "duration": 120})
    monkeypatch.setattr(captions, "fetch_ytdlp", lambda *a, **k: None)
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    monkeypatch.setattr(pipeline, "download_audio", lambda *a, **k: audio)

    class Fake:
        name, quant = "fake", "q4"

        def transcribe(self, *a, **k):
            return [AsrSegment(i, i + 1, "Thank you.") for i in range(4)]

    def boom(self, video, log):
        raise RuntimeError("no video stream")

    monkeypatch.setattr(Runner, "run_slides", boom)
    stats = run_batch(Runner(cfg, settings, "reuse", session, backend=Fake()), select_videos(session, None, False, None))
    v = session.get(Video, "a" * 11)
    assert stats["done"] == 1 and v.status == "done" and "no video stream" in v.meta["slides_error"]
    assert Path(settings.data_dir / "transcripts" / ("a" * 11) / "transcript.txt").exists()
