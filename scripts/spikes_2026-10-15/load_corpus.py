"""Shared loader for the topic experiments: one record per transcribed video with event metadata and full text."""
import csv
import re
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
SEASON = {"Spring": 0.25, "Fall": 0.75, "": 0.5}


def load():
    docs = []
    for r in csv.DictReader(open(DATA / "catalog.csv")):
        if r["status"] != "done":
            continue
        text = (DATA / "transcripts" / r["video_id"] / "transcript.txt").read_text()
        text = re.sub(r"\[slide @ \d\d:\d\d\]", " ", text)
        year, season = int(r["event_year"]), r["event_season"]
        docs.append({**r, "text": text, "year": year, "t": year + SEASON[season],
                     "event": f"{year} {season}".strip(), "n_words": len(text.split())})
    return sorted(docs, key=lambda d: (d["t"], d["upload_date"]))
