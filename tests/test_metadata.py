import csv
import json

from sqlalchemy import create_engine, text

from ytscribe import pipeline
from ytscribe import ytdlp_runner as yt
from ytscribe.db.models import Video
from ytscribe.db.session import add_missing_columns, init_db
from ytscribe.export import export_video, write_catalog
from ytscribe.pipeline import apply_info, enrich, upsert_source
from ytscribe.settings import Settings
from ytscribe.sources.resolve import Resolved, VideoRef, event_from_title, playlist_id

INFO = {"title": "Talk", "channel": "Allotrope Foundation", "channel_id": "UC1", "upload_date": "20260517",
        "duration": 1901, "description": "About ASM", "tags": ["asm", "ai"], "categories": ["Science"],
        "chapters": [{"start_time": 0, "end_time": 60, "title": "Intro", "extra": 1}], "view_count": 10,
        "like_count": 2, "language": "en"}


def test_event_and_playlist_id_parsing():
    assert event_from_title("2026 Spring Allotrope Connect Workshop") == (2026, "Spring")
    assert event_from_title("2020 Fall Allotrope Connect Day 1") == (2020, "Fall")
    assert event_from_title("2017 Allotrope Connect") == (2017, None)
    assert event_from_title("Modeling with Allotrope Framework - Technical") == (None, None)
    assert playlist_id("https://www.youtube.com/playlist?list=PLabc-_12") == "PLabc-_12"


def test_old_database_gets_new_columns(tmp_path):
    e = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with e.begin() as c:
        c.execute(text("CREATE TABLE videos (id VARCHAR(32) PRIMARY KEY, title VARCHAR(512), status VARCHAR(16))"))
        c.execute(text("INSERT INTO videos VALUES ('a', 'T', 'done')"))
    init_db(e)
    add_missing_columns(e)  # idempotent
    with e.connect() as c:
        cols = {r[1] for r in c.execute(text("PRAGMA table_info(videos)"))}
        assert {"description", "chapters", "metadata_fetched_at", "view_count"} <= cols
        assert c.execute(text("select title from videos")).scalar() == "T"  # data preserved


def test_meta_json_and_catalog_carry_playlist_date_length(session, tmp_path):
    r = Resolved("playlist", "https://www.youtube.com/playlist?list=PLx1", "2026 Spring Allotrope Connect Workshop",
                 [VideoRef("a" * 11, "Talk")], external_id="PLx1")
    upsert_source(session, r)
    v = session.get(Video, "a" * 11)
    apply_info(v, INFO)
    v.status, v.origin, v.meta = "done", "captions:auto:en", {"segments": 2, "words": 5}
    session.flush()
    settings = Settings(data_dir=tmp_path / "data")
    export_video(v, settings.data_dir)
    m = json.loads((settings.data_dir / "transcripts" / ("a" * 11) / "meta.json").read_text())
    assert m["upload_date"] == "2026-05-17" and m["duration_s"] == 1901 and m["duration_min"] == 31.7
    assert m["playlists"] == [{"id": "PLx1", "title": "2026 Spring Allotrope Connect Workshop",
                               "url": "https://www.youtube.com/playlist?list=PLx1", "position": 0,
                               "event_year": 2026, "event_season": "Spring"}]
    assert (m["event_year"], m["event_season"]) == (2026, "Spring")
    assert m["tags"] == ["asm", "ai"] and m["chapters"] == [{"start_time": 0, "end_time": 60, "title": "Intro"}]
    csv_path, jsonl_path, n = write_catalog(session, settings.data_dir)
    row = next(csv.DictReader(csv_path.open()))
    assert n == 1 and row["playlist_ids"] == "PLx1" and row["upload_date"] == "2026-05-17" and row["duration_min"] == "31.7"
    assert json.loads(jsonl_path.read_text().splitlines()[0])["video_id"] == "a" * 11


def test_enrich_only_touches_metadata(session, tmp_path, monkeypatch):
    upsert_source(session, Resolved("video", "https://x/v", "V", [VideoRef("a" * 11), VideoRef("b" * 11)]))
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)

    def fake(url, *a, **k):
        if "b" * 11 in url:
            raise yt.YtdlpFailure("boom")
        return INFO

    monkeypatch.setattr(yt, "extract", fake)
    stats = enrich(session, list(session.query(Video).order_by(Video.id)), {})
    assert stats == {"enriched": 1, "failed": 1}
    a, b = session.get(Video, "a" * 11), session.get(Video, "b" * 11)
    assert a.upload_date == "20260517" and a.metadata_fetched_at and a.status == "discovered"
    assert b.metadata_fetched_at is None
