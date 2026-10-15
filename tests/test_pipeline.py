import json

from ytscribe import captions, pipeline
from ytscribe import ytdlp_runner as yt
from ytscribe.asr.backends.base import AsrSegment
from ytscribe.db.models import Video
from ytscribe.pipeline import Runner, run_batch, select_videos, upsert_source, word_overlap
from ytscribe.settings import Config, Settings
from ytscribe.sources.resolve import Resolved, VideoRef, classify


def mk(tmp_path):
    cfg = Config()
    cfg.ytdlp.sleep_between_videos_s = 0
    return cfg, Settings(data_dir=tmp_path / "data", logs_dir=tmp_path / "logs")


def fake_info(**kw):
    return {"title": "T", "channel": "C", "upload_date": "20240101", "duration": 60, **kw}


def test_classify():
    assert classify("https://www.youtube.com/@AllotropeFoundation") == "channel"
    assert classify("https://www.youtube.com/playlist?list=PLx") == "playlist"
    assert classify("https://www.youtube.com/watch?v=abcdefghijk") == "video"


def test_upsert_is_idempotent(session):
    r = Resolved("playlist", "https://x/p", "P", [VideoRef("a" * 11, "A"), VideoRef("b" * 11, "B")])
    _, new1 = upsert_source(session, r)
    _, new2 = upsert_source(session, r)
    assert (new1, new2) == (2, 0)


def test_captions_path_writes_db_and_files(session, tmp_path, monkeypatch):
    cfg, settings = mk(tmp_path)
    upsert_source(session, Resolved("video", "https://x/v", "V", [VideoRef("a" * 11, "A")]))
    monkeypatch.setattr(yt, "extract", lambda *a, **k: fake_info())
    segs = [AsrSegment(0, 2, "hello there"), AsrSegment(2, 4, "general kenobi")]
    monkeypatch.setattr(captions, "fetch_ytdlp", lambda *a, **k: captions.CaptionResult(segs, "manual", "en", "yt-dlp"))
    monkeypatch.setattr(pipeline, "make_backend", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no ASR")))
    r = Runner(cfg, settings, "reuse", session)
    stats = run_batch(r, select_videos(session, None, False, None))
    v = session.get(Video, "a" * 11)
    assert stats["done"] == 1 and v.status == "done" and v.origin == "captions:manual:en"
    d = settings.data_dir / "transcripts" / ("a" * 11)
    assert "hello there general kenobi" in (d / "transcript.txt").read_text()
    assert "00:00:02,000 --> 00:00:04,000" in (d / "transcript.srt").read_text()
    assert json.loads((d / "meta.json").read_text())["transcript"]["origin"] == "captions:manual:en"


def test_falls_back_to_asr_when_no_captions(session, tmp_path, monkeypatch):
    cfg, settings = mk(tmp_path)
    cfg.captions.api_fallback = False
    upsert_source(session, Resolved("video", "https://x/v", "V", [VideoRef("a" * 11)]))
    monkeypatch.setattr(yt, "extract", lambda *a, **k: fake_info())
    monkeypatch.setattr(captions, "fetch_ytdlp", lambda *a, **k: None)
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    monkeypatch.setattr(pipeline, "download_audio", lambda *a, **k: audio)

    class Fake:
        name, quant = "fake", "q4"

        def transcribe(self, path, lang, **kw):
            return [AsrSegment(0, 1, "from asr")]

    r = Runner(cfg, settings, "reuse", session, backend=Fake())
    run_batch(r, select_videos(session, None, False, None))
    v = session.get(Video, "a" * 11)
    assert v.status == "done" and v.origin == "asr:fake-q4" and not audio.exists()


def test_failure_is_recorded_and_batch_continues(session, tmp_path, monkeypatch):
    cfg, settings = mk(tmp_path)
    upsert_source(session, Resolved("playlist", "https://x/p", "P", [VideoRef("a" * 11), VideoRef("b" * 11)]))

    def extract(url, *a, **k):
        if "a" * 11 in url:
            raise yt.YtdlpFailure("boom")
        raise yt.VideoUnavailable("Private video")

    monkeypatch.setattr(yt, "extract", extract)
    stats = run_batch(Runner(cfg, settings, "reuse", session), select_videos(session, None, False, None))
    assert stats["failed"] == 1 and stats["unavailable"] == 1
    a = session.get(Video, "a" * 11)
    assert a.status == "failed" and "boom" in a.error and a.attempts == 1
    assert [v.id for v in select_videos(session, None, True, None)] == ["a" * 11]


def test_word_overlap():
    r = word_overlap("allotrope simple model data", "allotrope simple models data")
    assert r["recall"] == 0.75


def test_caption_fetch_error_falls_back_instead_of_failing(session, tmp_path, monkeypatch):
    cfg, settings = mk(tmp_path)
    upsert_source(session, Resolved("video", "https://x/v", "V", [VideoRef("a" * 11)]))
    monkeypatch.setattr(yt, "extract", lambda *a, **k: fake_info())

    def boom(*a, **k):
        raise yt.YtdlpFailure("HTTP Error 429: Too Many Requests")

    monkeypatch.setattr(captions, "fetch_ytdlp", boom)
    monkeypatch.setattr(captions, "fetch_api", lambda *a, **k: captions.CaptionResult(
        [AsrSegment(0, 1, "via api")], "auto", "en", "transcript-api"))
    run_batch(Runner(cfg, settings, "reuse", session), select_videos(session, None, False, None))
    v = session.get(Video, "a" * 11)
    assert v.status == "done" and v.meta["captions_via"] == "transcript-api"


def test_quality_warnings_flag_hallucination_and_sparse():
    from ytscribe.pipeline import quality_warnings

    halluc = [AsrSegment(i, i + 1, "Thank you.") for i in range(4)]
    assert len(quality_warnings(halluc, 118)) == 2
    normal = [AsrSegment(i, i + 1, f"line number {i} with some real content words here") for i in range(30)]
    assert quality_warnings(normal, 60) == []


def test_log_context_reaches_nested_bound_loggers(tmp_path):
    """ASR lines use logger.bind(stage='asr'); they must still carry the video_id set by process()."""
    from loguru import logger

    from ytscribe.logging_setup import setup_logging

    setup_logging(tmp_path / "logs")
    seen = []
    sink = logger.add(lambda m: seen.append(m.record["extra"]), level="DEBUG")
    try:
        with logger.contextualize(video_id="vid123", attempt=2):
            logger.bind(stage="asr").info("chunk 1/3")
    finally:
        logger.remove(sink)
    assert seen[-1]["video_id"] == "vid123" and seen[-1]["stage"] == "asr" and seen[-1]["attempt"] == 2


def _prefetch_runner(session, tmp_path, monkeypatch, has_captions):
    cfg, settings = mk(tmp_path)
    audio = tmp_path / "data" / "audio" / "n.m4a"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"x")
    downloads = []
    monkeypatch.setattr(yt, "extract", lambda *a, **k: fake_info(subtitles={"en": [1]} if has_captions else {}))
    monkeypatch.setattr(pipeline, "download_audio", lambda *a, **k: downloads.append(1) or audio)
    return Runner(cfg, settings, "reuse", session), audio, downloads


def test_prefetch_downloads_audio_only_when_no_captions(session, tmp_path, monkeypatch):
    r, audio, downloads = _prefetch_runner(session, tmp_path, monkeypatch, has_captions=False)
    r.prefetch("n" * 11)
    assert r._take_audio("n" * 11) == audio and len(downloads) == 1  # taken from prefetch, not downloaded twice
    r.close()


def test_prefetch_skipped_when_captions_exist(session, tmp_path, monkeypatch):
    r, _, downloads = _prefetch_runner(session, tmp_path, monkeypatch, has_captions=True)
    r.prefetch("n" * 11)
    r._pool.shutdown(wait=True)
    assert downloads == []


def test_unneeded_prefetched_audio_is_deleted(session, tmp_path, monkeypatch):
    r, audio, _ = _prefetch_runner(session, tmp_path, monkeypatch, has_captions=False)
    r.prefetch("n" * 11)
    r._discard_prefetched("n" * 11)
    assert not audio.exists()
    r.close()
