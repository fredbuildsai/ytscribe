# ytscribe

Get transcripts for YouTube videos, playlists and channels. Existing captions are used when available (manual preferred
over auto-generated); otherwise the audio is downloaded and transcribed by local Whisper (default) or the Mistral API.

```bash
uv sync --extra dev --extra asr        # add --extra mistral for the API backend
uv run ys add configs/sources/allotrope.yaml   # register every "Allotrope Connect" playlist (no download)
uv run ys fetch --limit 3              # captions first, ASR fallback; resumable
uv run ys status
uv run ys fetch --policy always_asr --video <id>
```

Output: `data/ytscribe.db` (SQLite) and `data/transcripts/<video_id>/{meta.json,transcript.json,.srt,.txt}`.
Logs: `logs/ytscribe-YYYYMMDD.log` (DEBUG), `logs/ytscribe.jsonl` (structured). Config: `configs/default.yaml`.

yt-dlp self-heals: on an extractor failure it upgrades itself in the venv, reloads and retries once per run.
