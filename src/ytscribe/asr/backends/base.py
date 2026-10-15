from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class AsrSegment:
    start_s: float
    end_s: float
    text: str
    words: list[dict] = field(default_factory=list)  # [{word, start, end}]


class AsrBackend(Protocol):
    name: str
    quant: str

    def transcribe(self, audio_path: str, language: str) -> list[AsrSegment]: ...


class UnquantizedModelError(ValueError):
    """Local models must be quantized (project rule). Raised instead of silently loading full precision."""


def require_quant(quant: str | None) -> str:
    if quant not in ("q4", "q8"):
        raise UnquantizedModelError(f"backend config needs quant: q4 | q8, got {quant!r}")
    return quant
