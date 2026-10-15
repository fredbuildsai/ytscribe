"""Load the transcribed videos from the database as analysis documents (one per video)."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from ytscribe.analysis.config import TopicsConfig
from ytscribe.db.models import Video
from ytscribe.export import event_of, iso_date, playlists_of


@dataclass
class Doc:
    video_id: str
    title: str
    upload_date: str | None
    event_year: int | None
    event_season: str
    t: float  # year + season fraction: the event's position on the time axis
    event: str  # "2026 H2" (labels come from season_labels in the topic config)
    origin: str
    slide_only: bool
    description: str
    text: str
    duration_min: float | None = None

    @property
    def words(self) -> int:
        return len(self.text.split())


def load_docs(session: Session, cfg: TopicsConfig) -> list[Doc]:
    docs = []
    for v in session.scalars(select(Video).where(Video.status == "done")).unique():
        year, season = event_of(v)
        if year is None:
            continue
        season = season or ""
        text = " ".join(s.text for s in v.segments)
        docs.append(Doc(
            video_id=v.id, title=v.title or "", upload_date=iso_date(v.upload_date), event_year=year,
            event_season=season, t=year + cfg.season_order.get(season, 0.5), event=f"{year} {cfg.season_labels.get(season, season)}".strip(),
            origin=v.origin or "", slide_only="slides" in (v.origin or ""),
            description=v.description or "", text=text,
            duration_min=round(v.duration_s / 60, 1) if v.duration_s else None))
    return sorted(docs, key=lambda d: (d.t, d.upload_date or "", d.video_id))


def playlists(session: Session, video_id: str) -> list[dict]:
    v = session.get(Video, video_id)
    return playlists_of(v) if v else []
