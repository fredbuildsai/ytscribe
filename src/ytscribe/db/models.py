from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Video.status: discovered | done | failed | unavailable
STATUSES = ("discovered", "done", "failed", "unavailable")


class Base(DeclarativeBase):
    pass


class Source(Base):
    """A channel, playlist or single video the user asked for."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))  # channel | playlist | video
    url: Mapped[str] = mapped_column(String(512), unique=True)
    external_id: Mapped[str | None] = mapped_column(String(64), index=True)  # YouTube playlist / channel id
    title: Mapped[str | None] = mapped_column(String(512))
    event_year: Mapped[int | None] = mapped_column(Integer)  # parsed from the title, e.g. "2026 Spring ..." -> 2026
    event_season: Mapped[str | None] = mapped_column(String(16))  # Spring | Fall
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    videos: Mapped[list["VideoSource"]] = relationship(back_populates="source", cascade="all, delete-orphan")


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # YouTube id
    title: Mapped[str | None] = mapped_column(String(512))
    channel: Mapped[str | None] = mapped_column(String(256))
    upload_date: Mapped[str | None] = mapped_column(String(8))  # YYYYMMDD
    duration_s: Mapped[float | None] = mapped_column(Float)
    description: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list | None] = mapped_column(JSON)
    categories: Mapped[list | None] = mapped_column(JSON)
    chapters: Mapped[list | None] = mapped_column(JSON)  # [{start_time, end_time, title}]
    channel_id: Mapped[str | None] = mapped_column(String(64))
    language: Mapped[str | None] = mapped_column(String(16))
    view_count: Mapped[int | None] = mapped_column(Integer)
    like_count: Mapped[int | None] = mapped_column(Integer)
    comment_count: Mapped[int | None] = mapped_column(Integer)
    thumbnail: Mapped[str | None] = mapped_column(String(512))
    metadata_fetched_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(16), default="discovered", index=True)
    origin: Mapped[str | None] = mapped_column(String(128))  # how the transcript was produced
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict | None] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    sources: Mapped[list["VideoSource"]] = relationship(back_populates="video", cascade="all, delete-orphan")
    segments: Mapped[list["Segment"]] = relationship(back_populates="video", cascade="all, delete-orphan",
                                                     order_by="Segment.start_s")
    frames: Mapped[list["Frame"]] = relationship(back_populates="video", cascade="all, delete-orphan",
                                                 order_by="Frame.idx")

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.id}"


class VideoSource(Base):
    """Membership of a video in a source (and, for playlists, a playlist the video appears in)."""

    __tablename__ = "video_sources"
    __table_args__ = (UniqueConstraint("video_id", "source_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    position: Mapped[int | None] = mapped_column(Integer)

    video: Mapped[Video] = relationship(back_populates="sources")
    source: Mapped[Source] = relationship(back_populates="videos")


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    origin: Mapped[str] = mapped_column(String(128))  # captions:manual:en | captions:auto:en | asr:<model>-<quant>
    words: Mapped[list | None] = mapped_column(JSON)

    video: Mapped[Video] = relationship(back_populates="segments")


class Frame(Base):
    """A distinct slide/frame extracted from the video, with its OCR text (slide-only videos)."""

    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    path: Mapped[str] = mapped_column(String(512))
    ocr_text: Mapped[str | None] = mapped_column(Text)
    ocr_conf: Mapped[float | None] = mapped_column(Float)

    video: Mapped[Video] = relationship(back_populates="frames")


class Annotation(Base):
    """One LLM (or other) structured annotation of a talk. Versioned by prompt so re-runs never overwrite history."""

    __tablename__ = "annotations"
    __table_args__ = (UniqueConstraint("video_id", "kind", "prompt_version", "model"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # talk
    prompt_version: Mapped[str] = mapped_column(String(32))
    deployment: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    data: Mapped[dict] = mapped_column(JSON)
    verified_ratio: Mapped[float | None] = mapped_column(Float)  # share of evidence quotes found in the transcript
    taxonomy_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class GoldLabel(Base):
    """A human-chosen label for one field of one item: the reference the LLM and the baselines are scored against."""

    __tablename__ = "gold_labels"
    __table_args__ = (UniqueConstraint("item_id", "field"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[str] = mapped_column(String(64), index=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    field: Mapped[str] = mapped_column(String(32))
    value: Mapped[str] = mapped_column(String(128))
    note: Mapped[str | None] = mapped_column(Text)
    labeled_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
