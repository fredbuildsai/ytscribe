"""Write finished transcripts to data/transcripts/<video_id>/ and a cross-video catalog to data/catalog.{csv,jsonl}.

Per video: meta.json (everything needed to place the talk in time and context), transcript.json (timed segments),
transcript.srt, transcript.txt. The catalog is one row per video sorted by date, for time-evolution analysis.
"""

import csv
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ytscribe.db.models import Video
from ytscribe.quality import quality_warnings


def _ts(s: float) -> str:
    ms = round(s * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    return f"{h:02d}:{m:02d}:{ms // 1000:02d},{ms % 1000:03d}"


def iso_date(yyyymmdd: str | None) -> str | None:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}" if yyyymmdd and len(yyyymmdd) == 8 else None


def transcript_dir(data_dir: Path, video_id: str) -> Path:
    return data_dir / "transcripts" / video_id


def playlists_of(video: Video) -> list[dict]:
    out = [{"id": vs.source.external_id, "title": vs.source.title, "url": vs.source.url, "position": vs.position,
            "event_year": vs.source.event_year, "event_season": vs.source.event_season}
           for vs in video.sources if vs.source.kind == "playlist"]
    return sorted(out, key=lambda p: (p["event_year"] or 0, p["title"] or ""))


def event_of(video: Video) -> tuple[int | None, str | None]:
    """The workshop a talk belongs to: first playlist with a parsed year, else the upload year."""
    for p in playlists_of(video):
        if p["event_year"]:
            return p["event_year"], p["event_season"]
    d = iso_date(video.upload_date)
    return (int(d[:4]) if d else None), None


def video_meta(video: Video) -> dict:
    ev_year, ev_season = event_of(video)
    extra = video.meta or {}
    return {
        "id": video.id,
        "url": video.url,
        "title": video.title,
        "channel": video.channel,
        "channel_id": video.channel_id,
        "upload_date": iso_date(video.upload_date),
        "upload_year": int(video.upload_date[:4]) if video.upload_date else None,
        "event_year": ev_year,
        "event_season": ev_season,
        "duration_s": video.duration_s,
        "duration_min": round(video.duration_s / 60, 1) if video.duration_s else None,
        "playlists": playlists_of(video),
        "description": video.description,
        "tags": video.tags,
        "categories": video.categories,
        "chapters": video.chapters,
        "language": video.language,
        "view_count": video.view_count,
        "like_count": video.like_count,
        "comment_count": video.comment_count,
        "thumbnail": video.thumbnail,
        "metadata_fetched_at": video.metadata_fetched_at.isoformat() if video.metadata_fetched_at else None,
        "transcript": {"origin": video.origin,
                       "segments": len(video.segments) if video.status == "done" else extra.get("segments"),
                       "words": sum(len(x.text.split()) for x in video.segments) if video.status == "done"
                       else extra.get("words"),
                       "captions_via": extra.get("captions_via"), "caption_langs": extra.get("caption_langs"),
                       "verify": extra.get("verify"), "warnings": quality_warnings(video.segments, video.duration_s) if video.status == "done"
                       else extra.get("warnings")},
        "slides": {"frames": len(video.frames), **(extra.get("slides") or {})} if video.frames else None,
        "processing": {"ytdlp_version": extra.get("ytdlp_version"), "policy": extra.get("policy"),
                       "language": extra.get("language")},
    }


def _plain_text(segs) -> str:
    """Speech as running text; slide text on its own lines, marked so it is not mistaken for speech."""
    out, speech = [], []
    for s in segs:
        if s.origin.startswith("slides"):
            if speech:
                out.append(" ".join(speech))
                speech = []
            m, sec = divmod(int(s.start_s), 60)
            out.append(f"[slide @ {m:02d}:{sec:02d}]\n{s.text}\n")
        else:
            speech.append(s.text)
    if speech:
        out.append(" ".join(speech))
    return "\n".join(out) + "\n"


def export_video(video: Video, data_dir: Path) -> Path:
    d = transcript_dir(data_dir, video.id)
    d.mkdir(parents=True, exist_ok=True)
    segs = video.segments
    (d / "meta.json").write_text(json.dumps(video_meta(video), ensure_ascii=False, indent=2))
    (d / "transcript.json").write_text(json.dumps(
        [{"start": s.start_s, "end": s.end_s, "text": s.text, "origin": s.origin} for s in segs],
        ensure_ascii=False, indent=1))
    (d / "transcript.srt").write_text("\n".join(
        f"{i}\n{_ts(s.start_s)} --> {_ts(s.end_s)}\n{s.text}\n" for i, s in enumerate(segs, 1)))
    (d / "transcript.txt").write_text(_plain_text(segs))
    if video.frames:
        (d / "slides.json").write_text(json.dumps(
            [{"index": f.idx, "start": f.start_s, "end": f.end_s, "image": f.path, "text": f.ocr_text,
              "ocr_confidence": f.ocr_conf} for f in video.frames], ensure_ascii=False, indent=1))
    return d


CATALOG_FIELDS = ["video_id", "url", "title", "upload_date", "event_year", "event_season", "playlist_ids",
                  "playlist_titles", "duration_min", "words", "segments", "origin", "warnings", "channel", "tags",
                  "chapters", "slides", "view_count", "like_count", "status", "transcript_dir"]


def catalog_rows(session: Session, data_dir: Path) -> list[dict]:
    rows = []
    for v in session.scalars(select(Video)).unique():
        m = video_meta(v)
        rows.append({
            "video_id": v.id, "url": v.url, "title": v.title, "upload_date": m["upload_date"],
            "event_year": m["event_year"], "event_season": m["event_season"],
            "playlist_ids": "|".join(p["id"] or "" for p in m["playlists"]),
            "playlist_titles": "|".join(p["title"] or "" for p in m["playlists"]),
            "duration_min": m["duration_min"], "words": m["transcript"]["words"],
            "segments": m["transcript"]["segments"], "origin": v.origin,
            "warnings": "; ".join(m["transcript"]["warnings"] or []), "channel": v.channel,
            "tags": "|".join(v.tags or []), "chapters": len(v.chapters or []),
            "slides": len(v.frames), "view_count": v.view_count,
            "like_count": v.like_count, "status": v.status,
            "transcript_dir": str(transcript_dir(data_dir, v.id)) if v.status == "done" else "",
        })
    return sorted(rows, key=lambda r: (r["upload_date"] or "9999", r["video_id"]))


def write_catalog(session: Session, data_dir: Path) -> tuple[Path, Path, int]:
    rows = catalog_rows(session, data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path, jsonl_path = data_dir / "catalog.csv", data_dir / "catalog.jsonl"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    jsonl_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return csv_path, jsonl_path, len(rows)
