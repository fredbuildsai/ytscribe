"""Runtime settings (env prefix YS_) and the YAML pipeline config. PROJECT_ROOT is the current working directory."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path.cwd()


def load_env(env_file: Path | None = None) -> None:
    """Export `.env` into os.environ so API clients (MISTRAL_API_KEY) see their keys."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YS_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    logs_dir: Path = PROJECT_ROOT / "logs"
    configs_dir: Path = PROJECT_ROOT / "configs"
    database_url: str = ""
    log_level: str = "INFO"

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir / 'ytscribe.db'}"


@lru_cache
def get_settings() -> Settings:
    return Settings()


class CaptionsCfg(BaseModel):
    prefer_manual: bool = True
    languages: list[str] = ["en", "en-US", "en-GB", "en-orig"]
    api_fallback: bool = True


class MistralCfg(BaseModel):
    model: str = "voxtral-mini-latest"
    diarize: bool = False


class AsrCfg(BaseModel):
    backend: str = "whisper"
    quant: str | None = "q4"
    chunk_s: float = 600.0
    mistral: MistralCfg = MistralCfg()


class YtdlpCfg(BaseModel):
    auto_update: bool = True
    sleep_between_videos_s: float = 3.0
    retries: int = 3
    cookies_from_browser: str | None = None


class SlidesCfg(BaseModel):
    auto: bool = True            # run slide extraction when a transcript is low-content (silent / music-only slide video)
    fps: float = 1.0             # frames sampled per second for change detection
    hash_size: int = 16          # dHash size: 16 -> 256 bits
    same_thresh: int = 8         # hamming distance <= this: same slide (animation noise)
    dup_thresh: int = 12         # consecutive kept slides closer than this are merged (animation builds); keep the later
    min_run_s: float = 2.0       # a slide must be stable at least this long (drops fades / transitions)
    max_height: int = 720
    ocr_min_conf: float = 0.3
    keep_video: bool = False


class Config(BaseModel):
    transcript_policy: str = "reuse"
    language: str = "en"
    captions: CaptionsCfg = CaptionsCfg()
    verify_minutes: int = 10
    asr: AsrCfg = AsrCfg()
    ytdlp: YtdlpCfg = YtdlpCfg()
    slides: SlidesCfg = SlidesCfg()
    keep_audio: bool = False
    min_free_disk_gb: float = 5.0


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def load_config(name: str = "default.yaml") -> Config:
    return Config(**load_yaml(get_settings().configs_dir / name))
