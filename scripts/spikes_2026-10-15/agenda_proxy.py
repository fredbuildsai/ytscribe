"""Experiment 02: use the published 2026 Fall agenda as a proxy for talks not yet on YouTube. No LLM.

Agenda abstracts are ~40 words, talks ~2,500: raw rates are not comparable. So we (1) score agenda text with the same
lexicons, (2) check on the sessions that exist in both forms how well abstract mentions predict transcript rates, and
(3) only then project the missing sessions. Outputs data/analysis/agenda_2026_fall_scored.csv."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from topics_words import GROUP_DIRECTION, LEXICONS  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
# agenda title (prefix) -> uploaded video id; matched by hand against the 11 uploaded 2026 Fall videos
MATCH = {
    "Welcome and Introduction to Allotrope": "1TGRgvAS66w",
    "Keynote: Welcome and an Overview of USP": "ngvwQ6MJ6jI",
    "Advancing ASM for Digital Analytical Methods": "aJNu8hKcmwo",
    "From Paper to Digital": "gyR8ay43xe8",
    "Future Opportunities Utilizing ASM": "nLSezlowMpM",
    "Pistoia Alliance Open-Source": "AEPgvMal254",
    "ASM as the Semantic Contract": "7flw13137Lw",
    "Automating ASM Generation": "CRT6w6mT0LY",
    "ASM Custom Manifest": "sl8eajz8veU",
    "Connecting the Laboratory and the Manufacturing": "RBA4-6BK0DA",
    "Providing Universal Context": "pbuItc7SBto",
}


def score(text):
    return {n: len(re.findall(p, text, f)) for n, (p, f) in LEXICONS.items()}


ag = json.load(open(DATA / "agenda" / "2026_fall.json"))["sessions"]
cnt = pd.read_csv(DATA / "analysis" / "topic_counts_by_video.csv").set_index("video_id")
rows = []
for s in ag:
    vid = next((v for k, v in MATCH.items() if s["title"].startswith(k)), None)
    sc = score(s["title"] + ". " + s["description"])
    r = {"slot": s["slot"], "title": s["title"][:70], "video": vid or "", "agenda_words": len((s["title"] + " " + s["description"]).split())}
    r.update({f"ag {n}": sc[n] for n in GROUP_DIRECTION})
    if vid:
        w = cnt.loc[vid, "words"]
        r["tr_words"] = w
        r.update({f"tr {n}": cnt.loc[vid, n] / w * 1e4 for n in GROUP_DIRECTION})
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(DATA / "analysis" / "agenda_2026_fall_scored.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)

print("=== agenda sessions scored (abstract mention counts) ===")
print(df[["slot", "title", "video"] + [f"ag {n}" for n in GROUP_DIRECTION]].to_string(index=False))

m = df[df["video"] != ""]
print(f"\n=== calibration on the {len(m)} sessions available in both forms ===")
for n in GROUP_DIRECTION:
    a, t = m[f"ag {n}"].to_numpy(float), m[f"tr {n}"].to_numpy(float)
    corr = np.corrcoef(a, t)[0, 1] if a.std() and t.std() else float("nan")
    hit = ((a > 0) & (t >= 5)).sum(), (a > 0).sum(), ((a == 0) & (t >= 5)).sum()
    print(f"{n:24} corr(abstract count, transcript rate)={corr:5.2f} | abstract>0: {hit[1]} sessions, of which transcript>=5/10k: {hit[0]}"
          f" | abstract==0 but transcript>=5/10k: {hit[2]} of {(a == 0).sum()}")

un = df[df["video"] == ""]
print(f"\n=== agenda share of sessions mentioning each topic: matched ({len(m)}) vs missing ({len(un)}) ===")
for n in GROUP_DIRECTION:
    print(f"{n:24} matched {100*(m[f'ag {n}']>0).mean():4.0f}%   missing {100*(un[f'ag {n}']>0).mean():4.0f}%   all {100*(df[f'ag {n}']>0).mean():4.0f}%")


# ---------------------------------------------------------------------------------------------------------------
# Part 2: title-based projection. Learn from the corpus how a topic in the TITLE shifts the transcript rate, then apply
# to the titles of sessions with no video yet. Validated on the sessions we do have.
# ---------------------------------------------------------------------------------------------------------------
print("\n\n##### PART 2: title-based projection #####")
all_counts = pd.read_csv(DATA / "analysis" / "topic_counts_by_video.csv")
pool = all_counts[(all_counts["words"] > 1500) & ~all_counts["origin"].str.contains("slides") & (all_counts["event"] != "2026 Fall")].copy()


def title_flags(title):
    t = re.sub(r"^\d{4} (Spring|Fall) AC\s*[-–]?\s*", "", title)
    return {n: int(len(re.findall(p, t, f)) > 0) for n, (p, f) in LEXICONS.items() if n in GROUP_DIRECTION}


for n in GROUP_DIRECTION:
    pool[f"rate {n}"] = pool[n] / pool["words"] * 1e4
    pool[f"flag {n}"] = pool["title"].map(lambda t, n=n: title_flags(t)[n])
ERA_START = {"ADF (data format)": 2023.0, "ASM (simple model)": 2021.5, "AI / ML": 2024.5}  # baselines moved over time


def group_means(topic, rng=None):
    d = pool[pool["t"] >= ERA_START.get(topic, 0)]
    if rng is not None:
        d = d.sample(len(d), replace=True, random_state=rng)
    yes, no = d[d[f"flag {topic}"] == 1][f"rate {topic}"], d[d[f"flag {topic}"] == 0][f"rate {topic}"]
    return (yes.mean() if len(yes) else no.mean()), no.mean(), len(yes), len(no)


print("rate per 10k words by title flag (pool = other events' talks; era per topic: ADF 2023+, ASM 2021 Fall+, AI 2024 Fall+):")
for n in GROUP_DIRECTION:
    y, no, ny, nn = group_means(n)
    print(f"  {n:24} topic in title: {y:6.1f} (n={ny:3})   not in title: {no:6.1f} (n={nn})")

mins = {"Industry Panel": 45}
rows2 = []
wpm = (cnt.loc[[v for v in MATCH.values()], "words"].sum()) / (
    sum(float(x) for x in [11.0, 49.5, 26.9, 14.1, 22.8, 25.8, 17.4, 29.7, 21.0, 18.2, 21.7]))  # catalog minutes
for s in ag:
    vid = next((v for k, v in MATCH.items() if s["title"].startswith(k)), None)
    if vid or s["title"].startswith("Group Discussion"):
        continue  # have the video / open discussion (not projected)
    d = 45 if s["title"].startswith("Industry Panel") else 30
    rows2.append({"title": s["title"], "min": d, "words": d * wpm, **{f"flag {n}": title_flags(s["title"])[n] for n in GROUP_DIRECTION}})
miss = pd.DataFrame(rows2)


def project(seed=None):
    rng = np.random.RandomState(seed) if seed is not None else None
    est = {}
    for n in GROUP_DIRECTION:
        y, no, _, _ = group_means(n, rng)
        est[n] = (miss[f"flag {n}"] * y + (1 - miss[f"flag {n}"]) * no) * miss["words"] / 1e4  # expected mentions per session
    return est


# validation: pretend the matched sessions have no video; predict their rate from titles only
val = pd.DataFrame([{"title": s["title"], "w": cnt.loc[v, "words"], **{n: title_flags(s["title"])[n] for n in GROUP_DIRECTION}}
                    for s in ag for k, v in MATCH.items() if s["title"].startswith(k)])
print("\nvalidation on the 11 sessions we have (title-only prediction vs actual transcript, per 10k words):")
for n in GROUP_DIRECTION:
    y, no, _, _ = group_means(n)
    pred = ((val[n] * y + (1 - val[n]) * no) * val["w"] / 1e4).sum() / val["w"].sum() * 1e4
    act = sum(cnt.loc[v, n] for v in MATCH.values()) / val["w"].sum() * 1e4
    print(f"  {n:24} predicted {pred:5.1f}   actual {act:5.1f}")

have = cnt.loc[list(MATCH.values())]
tot_words_have = have["words"].sum()
boots = []
for b in range(500):
    est = project(b)
    boots.append({n: (have[n].sum() + est[n].sum()) / (tot_words_have + miss["words"].sum()) * 1e4 for n in GROUP_DIRECTION})
bt = pd.DataFrame(boots)
point = {n: (have[n].sum() + project()[n].sum()) / (tot_words_have + miss["words"].sum()) * 1e4 for n in GROUP_DIRECTION}
print(f"\nprojected sessions without video ({len(miss)}), assumed ~{wpm:.0f} words/min:")
print(miss[["title", "min"] + [f"flag {n}" for n in GROUP_DIRECTION]].to_string(index=False, max_colwidth=58))
out = []
print("\nfull-event 2026 Fall estimate (per 10k words): transcribed-only | projected incl. missing talks (p10-p90)")
for n in GROUP_DIRECTION:
    only = have[n].sum() / tot_words_have * 1e4
    lo, hi = bt[n].quantile([0.1, 0.9])
    print(f"  {n:24} {only:5.1f} | {point[n]:5.1f}  ({lo:4.1f}-{hi:4.1f})")
    out.append({"topic": n, "transcribed_only": only, "projected": point[n], "p10": lo, "p90": hi})
pd.DataFrame(out).to_csv(DATA / "analysis" / "projection_2026_fall.csv", index=False)
