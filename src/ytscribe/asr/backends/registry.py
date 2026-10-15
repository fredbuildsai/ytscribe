"""Pick the ASR backend from config. Default is local whisper; mistral is opt-in and needs MISTRAL_API_KEY."""

from pathlib import Path

from ytscribe.asr.backends.base import AsrBackend
from ytscribe.settings import AsrCfg


def make_backend(cfg: AsrCfg, models_dir: Path = Path("models")) -> AsrBackend:
    if cfg.backend == "whisper":
        from ytscribe.asr.backends.whisper import WhisperBackend

        return WhisperBackend(cfg.quant, cache_dir=models_dir, chunk_s=cfg.chunk_s)
    if cfg.backend == "mistral":
        from ytscribe.asr.backends.mistral import MistralBackend

        return MistralBackend(cfg.mistral.model, diarize=cfg.mistral.diarize, chunk_s=cfg.chunk_s)
    raise ValueError(f"unknown asr.backend {cfg.backend!r} (whisper | mistral)")


def origin_of(backend: AsrBackend) -> str:
    return f"asr:{backend.name}-{backend.quant}"
