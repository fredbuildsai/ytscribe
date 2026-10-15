"""mlx-whisper large-v3-turbo, quantized. Chosen by the M0 bake-off (ParlWatch M0 report).

mlx-whisper 0.4.3 loads only `weights.safetensors` / `weights.npz`. The published q4 repo ships weights.npz; the
newer 8-bit repo ships `model.safetensors`, so it is exposed under the expected name via a local symlink dir.
"""

from pathlib import Path

from loguru import logger

from ytscribe.asr.backends.base import AsrSegment, require_quant

REPOS = {"q4": "mlx-community/whisper-large-v3-turbo-q4", "q8": "mlx-community/whisper-large-v3-turbo-8bit"}


def _model_path(quant: str, cache_dir: Path) -> str:
    if quant == "q4":
        return REPOS["q4"]
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(REPOS["q8"]))
    dst = cache_dir / "whisper-turbo-q8"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        link = dst / ("weights.safetensors" if f.name == "model.safetensors" else f.name)
        link.unlink(missing_ok=True)
        link.symlink_to(f.resolve())
    return str(dst)


class WhisperBackend:
    name = "whisper-large-v3-turbo"

    def __init__(self, quant: str | None, cache_dir: Path = Path("models"), chunk_s: float = 600.0):
        self.quant = require_quant(quant)
        self.cache_dir = cache_dir
        self.chunk_s = chunk_s

    def transcribe(self, audio_path: str, language: str, checkpoint_dir: Path | None = None) -> list[AsrSegment]:
        """Transcribe in ~10 min pieces (see chunking.py). With `checkpoint_dir`, each piece's result is saved as JSON
        and reused on restart, so a killed run resumes instead of starting over."""
        import json
        import time

        import mlx.core as mx
        import mlx_whisper

        from ytscribe.asr.chunking import chunks

        log = logger.bind(stage="asr")
        model_path, out = _model_path(self.quant, self.cache_dir), []
        log.info("whisper {} {} model={} language={}", self.name, self.quant, model_path, language)
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with chunks(audio_path, self.chunk_s) as pieces:  # mlx-whisper needs the whole file in memory: see chunking.py
            for i, (offset, f) in enumerate(pieces):
                ck = checkpoint_dir / f"chunk_{i:03d}.json" if checkpoint_dir else None
                if ck and ck.exists():
                    out += [AsrSegment(**d) for d in json.loads(ck.read_text())]
                    log.info("chunk {}/{} @ {:.0f}s (cached)", i + 1, len(pieces), offset)
                    continue
                t0 = time.monotonic()
                res = mlx_whisper.transcribe(
                    str(f),
                    path_or_hf_repo=model_path,
                    language=language,  # forced: never auto-detect on talk audio
                    condition_on_previous_text=False,  # avoids runaway repetition on long sessions
                    word_timestamps=True,
                )
                segs = [
                    AsrSegment(offset + s["start"], offset + s["end"], s["text"].strip(),
                               [{"word": w["word"], "start": offset + w["start"], "end": offset + w["end"]}
                                for w in s.get("words", [])])
                    for s in res["segments"]
                ]
                if ck:
                    ck.write_text(json.dumps([s.__dict__ for s in segs], ensure_ascii=False))
                out += segs
                del res
                mx.clear_cache()  # MLX otherwise keeps every piece's GPU buffers: 11 GB after 5 pieces (2026-10-09)
                log.info("chunk {}/{} @ {:.0f}s: {} segments in {:.0f}s, mem active {:.1f} GB cache {:.1f} GB",
                         i + 1, len(pieces), offset, len(segs), time.monotonic() - t0,
                         mx.get_active_memory() / 1e9, mx.get_cache_memory() / 1e9)
        return out
