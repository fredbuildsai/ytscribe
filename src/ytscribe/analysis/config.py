import re
from pathlib import Path

import yaml
from pydantic import BaseModel

from ytscribe.settings import get_settings


class ProjectionCfg(BaseModel):
    era_start: dict[str, float] = {}
    minutes_default: int = 30
    minutes_panel: int = 45
    bootstrap: int = 500


class TermsCfg(BaseModel):
    extra_stopwords: list[str] = []
    prior_sigma: float = 10.0
    top: int = 15


class SentimentCfg(BaseModel):
    window_words: int = 30


class AnchorBreakdownCfg(BaseModel):
    terms: dict[str, list[str]] = {}
    min_topic_size: int | None = None  # overrides breakdown.min_topic_size for this anchor


class BreakdownCfg(BaseModel):
    window_words: int = 40
    min_topic_size: int = 10
    nr_topics: int | None = 8
    anchors: dict[str, AnchorBreakdownCfg] = {}


class CorrelationCfg(BaseModel):
    groups: list[str] = ["direction", "partners"]
    min_talks: int = 8
    min_passages: int = 15
    passage_words: int = 150
    bootstrap: int = 2000


class TopicsConfig(BaseModel):
    season_order: dict[str, float] = {"Spring": 0.25, "Fall": 0.75}
    season_labels: dict[str, str] = {"Spring": "H1", "Fall": "H2"}  # how events are shown: "2026 H2"
    min_words_solid: int = 1000
    min_talks_solid: int = 3
    substantive_mentions: int = 3  # a talk 'substantively' covers a topic from this many mentions up
    groups: dict[str, list[str]] = {}
    topics: dict[str, list[str]] = {}
    terms: TermsCfg = TermsCfg()
    projection: ProjectionCfg = ProjectionCfg()
    sentiment: SentimentCfg = SentimentCfg()
    breakdown: BreakdownCfg = BreakdownCfg()
    correlation: CorrelationCfg = CorrelationCfg()


def load_topics(path: Path | str | None = None) -> TopicsConfig:
    p = Path(path) if path else get_settings().configs_dir / "topics" / "allotrope.yaml"
    return TopicsConfig(**(yaml.safe_load(p.read_text()) or {}))


def normalize_event(cfg: TopicsConfig, event: str) -> str:
    """Accept either naming on the command line: "2026 Fall" and "2026 H2" both give the configured label."""
    for season, label in cfg.season_labels.items():
        event = re.sub(rf"\b{season}\b", label, event, flags=re.I)
    return event.strip()
