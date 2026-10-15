"""Mistral Voxtral Mini Transcribe (cloud). Optional backend: audio leaves the machine, costs money per minute.

Not a local model, so the q4/q8 rule does not apply; `quant` is recorded as "cloud". Gives segment timestamps and,
with diarize=True, a speaker_id per segment (labels restart in every chunk, so speakers are namespaced per chunk
and must be reconciled later against the official record). `context_bias` nudges the model toward given terms
(deputy names, acronyms). Key: MISTRAL_API_KEY in the environment / .env.
"""

import os
from pathlib import Path

from loguru import logger

from ytscribe.asr.backends.base import AsrSegment
from ytscribe.asr.chunking import chunks


class MistralBackend:
    name = "voxtral-mini-latest"
    quant = "cloud"

    def __init__(self, model: str = "voxtral-mini-latest", diarize: bool = True, chunk_s: float = 600.0,
                 context_bias: list[str] | None = None, client=None):
        self.model, self.diarize, self.chunk_s = model, diarize, chunk_s
        self.context_bias = (context_bias or [])[:100]
        self._client = client
        if client is None and not os.environ.get("MISTRAL_API_KEY"):  # fail fast, before any download
            raise RuntimeError("asr.backend is 'mistral' but MISTRAL_API_KEY is not set (add it to .env)")

    @property
    def client(self):
        if self._client is None:
            from mistralai.client import Mistral

            key = os.environ.get("MISTRAL_API_KEY")
            if not key:
                raise RuntimeError("MISTRAL_API_KEY is not set (add it to .env)")
            self._client = Mistral(api_key=key)
        return self._client

    def transcribe(self, audio_path: str, language: str) -> list[AsrSegment]:
        out: list[AsrSegment] = []
        with chunks(audio_path, self.chunk_s) as pieces:
            for idx, (offset, f) in enumerate(pieces):
                out += self._one(f, language, offset, idx)
        return out

    def _one(self, f: Path, language: str, offset: float, idx: int) -> list[AsrSegment]:
        kwargs = {"model": self.model, "language": language, "diarize": self.diarize,
                  "timestamp_granularities": ["segment"]}
        if self.context_bias:
            kwargs["context_bias"] = self.context_bias
        logger.bind(stage="asr").info("mistral {} piece {} @ {:.0f}s", self.model, idx, offset)
        with open(f, "rb") as fh:
            res = self.client.audio.transcriptions.complete(file={"content": fh, "file_name": f.name}, **kwargs)
        segs = []
        for s in res.segments or []:
            spk = getattr(s, "speaker_id", None)
            words = [{"speaker": f"c{idx}:{spk}"}] if spk is not None else []
            segs.append(AsrSegment(offset + s.start, offset + s.end, s.text.strip(), words))
        return segs
