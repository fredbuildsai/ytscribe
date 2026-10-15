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


class TopicsConfig(BaseModel):
    season_order: dict[str, float] = {"Spring": 0.25, "Fall": 0.75}
    min_words_solid: int = 1000
    min_talks_solid: int = 3
    groups: dict[str, list[str]] = {}
    topics: dict[str, list[str]] = {}
    terms: TermsCfg = TermsCfg()
    projection: ProjectionCfg = ProjectionCfg()
    sentiment: SentimentCfg = SentimentCfg()


def load_topics(path: Path | str | None = None) -> TopicsConfig:
    p = Path(path) if path else get_settings().configs_dir / "topics" / "allotrope.yaml"
    return TopicsConfig(**(yaml.safe_load(p.read_text()) or {}))
