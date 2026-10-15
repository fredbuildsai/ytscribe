"""Experiment 01 (2026-10-15): word-level topic analysis of the Allotrope Connect transcripts. No LLM.

1. Lexicon trends: mentions per 10k words and share of talks mentioning, per event (format/ontology vs ASM vs AI, partners).
2. Changepoint: where a single step in each series fits best.
3. Distinctive terms per event (TF-IDF over event-level documents), discovered without a predefined list.

Outputs go to data/analysis/. Lexicons include the spellings the YouTube auto-captions actually produce
(e.g. 'pistoia' -> 'pistolia', 'pistoria', 'pistol alliance'), found by probing the corpus.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

sys.path.insert(0, str(Path(__file__).parent))
from load_corpus import DATA, load  # noqa: E402

OUT = DATA / "analysis"
OUT.mkdir(exist_ok=True)
I = re.I

LEXICONS: dict[str, tuple[str, int]] = {
    # --- the direction question: format + ontologies vs ASM vs AI ------------------------------------------------
    "ADF (data format)": (r"\bADF\b|\bHDF ?5\b|allotrope data format|\bdata cubes?\b|\bdata packages?\b", I),
    "Ontologies / semantics": (r"\bontolog\w*|\bAFO\b|\bRDF\b|\bOWL\b|\bSHACL\b|\bSPARQL\b|triple ?stores?|"
                               r"knowledge graphs?|semantic web|linked data|\btaxonom\w*", I),
    "ASM (simple model)": (r"\bASM\b|\bsimple models?\b", I),
    "AI / ML": (r"\bAI\b|artificial intelligence|machine learning|deep learning|\bLLMs?\b|large language models?|"
                r"chat ?GPT|\bGPT\b|generative|agentic|\bcopilot\b|foundation models?|neural networks?|"
                r"natural language processing|\bNLP\b", 0),  # case-sensitive: 'AI' only (lowercase 'ai' is noise)
    # --- partners / ecosystem -----------------------------------------------------------------------------------
    "Pistoia Alliance": (r"\b(pistoia|pistolia|pistoria|pistoya|pistoa|pisto|pistol)\b(?= alliance| airlines?|"
                         r" allian)|\b(pistoia|pistolia|pistoria|pistoya)\b", I),
    "OPC UA / LADS": (r"\bOPC[- ]?UAS?\b|\bopca\b|\bLADS\b|laboratory and analytical device|lab(oratory)? analytical "
                      r"devices?|\bSPECTARIS\b|\bspectorus\b", I),
    "SiLA": (r"\bSiLA\b", 0),
    "ISA-88 / 95": (r"\bISA[- ]?(88|95)\b", I),
    "IDMP": (r"\bIDMP\b", I),
    "FAIR": (r"\bFAIR\b|\bfair (data|principles)\b", 0),
    # --- AI and cloud companies ('jean-claude' is a person; captions sometimes hear 'Claude' as 'cloud' after Anthropic) ---
    "Anthropic (company)": (r"\banthropic\b|\bantropic\b", I),
    "Claude (AI)": (r"(?<!jean-)\bclaude\b", I),
    "OpenAI (company)": (r"\bopen ?AI\b", I),
    "ChatGPT / GPT (AI)": (r"\bchat ?g(pt|bt|pd)\w*|\bg(pt|bt|pd)[- ]?\d\w*|\bGPT\b", I),
    "AWS (cloud)": (r"\bAWS\b|(?i:amazon web services?)", 0),
    "Amazon (other)": (r"\bamazon\b(?! web service)", I),
    "Microsoft / Azure (cloud)": (r"\bmicrosoft\b|\bazure\b", I),
    "Copilot (AI)": (r"\bcopilot\b", I),
    "Google (company)": (r"\bgoogle\b", I),
    "Gemini (AI)": (r"\bgemini\b", I),
    "MCP (protocol)": (r"\bMCP\b", 0),
}
GROUP_DIRECTION = ["ADF (data format)", "Ontologies / semantics", "ASM (simple model)", "AI / ML"]
PARTNERS = ["Pistoia Alliance", "OPC UA / LADS", "SiLA", "ISA-88 / 95", "IDMP", "FAIR"]
AI_CLOUD = ["Anthropic (company)", "Claude (AI)", "OpenAI (company)", "ChatGPT / GPT (AI)", "AWS (cloud)", "Amazon (other)",
            "Microsoft / Azure (cloud)", "Copilot (AI)", "Google (company)", "Gemini (AI)", "MCP (protocol)"]

FILLER = {"uh", "um", "yeah", "okay", "ok", "gonna", "going", "know", "like", "just", "kind", "sort", "really", "thing",
          "things", "lot", "think", "want", "need", "got", "let", "say", "said", "make", "right", "mean", "actually",
          "thank", "thanks", "slide", "slides", "next", "talk", "today", "hello", "welcome", "good", "great", "yes",
          "bit", "way", "use", "using", "used", "come", "look", "see", "time", "work", "working", "people",
          "allotrope", "foundation", "presentation", "session", "question", "questions", "morning", "afternoon"}


def build_counts(docs):
    rows = []
    for d in docs:
        row = {"video_id": d["video_id"], "event": d["event"], "t": d["t"], "upload_date": d["upload_date"],
               "title": d["title"], "words": d["n_words"], "origin": d["origin"]}
        for name, (pat, flags) in LEXICONS.items():
            row[name] = len(re.findall(pat, d["text"], flags))
        rows.append(row)
    return pd.DataFrame(rows)


def by_event(df):
    g = df.groupby(["t", "event"], sort=True)
    out = g.agg(talks=("video_id", "count"), words=("words", "sum"))
    for name in LEXICONS:
        out[f"{name} /10k"] = g[name].sum() / out["words"] * 1e4
        out[f"{name} %talks"] = g[name].apply(lambda s: (s > 0).mean() * 100)
    return out.reset_index()


def changepoint(series: pd.Series, t: pd.Series, weights: pd.Series) -> dict:
    """Best single step (least squares, word-weighted) in an event-level series: the split index maximising the
    drop in squared error. Returns where the level changes and the before/after means."""
    y, w = series.to_numpy(float), weights.to_numpy(float)
    best = None
    for k in range(1, len(y)):  # events [0:k) vs [k:)
        a, b = np.average(y[:k], weights=w[:k]), np.average(y[k:], weights=w[k:])
        sse = (w[:k] * (y[:k] - a) ** 2).sum() + (w[k:] * (y[k:] - b) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, k, a, b)
    sse, k, a, b = best
    total = (w * (y - np.average(y, weights=w)) ** 2).sum()
    return {"first_event_after_step": t.iloc[k], "before": a, "after": b, "variance_explained": 1 - sse / total}


def distinctive_terms(docs, top=14):
    ev = {}
    for d in docs:
        ev.setdefault(d["event"], []).append(d["text"])
    names = list(ev)
    stop = list(ENGLISH_STOP_WORDS | FILLER)
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop, min_df=1, max_df=0.5, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", lowercase=True)
    m = vec.fit_transform([" ".join(ev[n]) for n in names])
    terms = np.array(vec.get_feature_names_out())
    return {n: list(terms[np.argsort(-m[i].toarray()[0])[:top]]) for i, n in enumerate(names)}


def main():
    docs = load()
    df = build_counts(docs)
    df.to_csv(OUT / "topic_counts_by_video.csv", index=False)
    ev = by_event(df)
    ev.to_csv(OUT / "topic_trends_by_event.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\n=== mentions per 10k words, by event (talks, words) ===")
    cols = ["event", "talks", "words"] + [f"{n} /10k" for n in GROUP_DIRECTION]
    print(ev[cols].round(1).to_string(index=False))
    print("\n=== partners: mentions per 10k words | % of talks mentioning ===")
    print(ev[["event", "talks"] + [f"{n} /10k" for n in PARTNERS]].round(1).to_string(index=False))
    print(ev[["event"] + [f"{n} %talks" for n in PARTNERS]].round(0).to_string(index=False))

    print("\n=== AI and cloud companies: mentions per 10k words | % of talks mentioning ===")
    print(ev[["event", "talks"] + [f"{n} /10k" for n in AI_CLOUD]].round(1).to_string(index=False))
    print(ev[["event"] + [f"{n} %talks" for n in AI_CLOUD]].round(0).to_string(index=False))
    for n in AI_CLOUD[:-1]:
        tt = df[df[n] > 0].sort_values(n, ascending=False).head(3)
        print(f"  top talks for {n} ({int(df[n].sum())} mentions in {int((df[n] > 0).sum())} talks): " +
              "; ".join(f"{int(r[n])}x {r['event']} '{r['title'][14:62]}'" for _, r in tt.iterrows()))

    print("\n=== single-step changepoints (word-weighted) ===")
    solid = ev[ev["words"] > 1000]  # 2023 Fall is slide-only/tiny: exclude from step fitting
    for n in GROUP_DIRECTION + PARTNERS[:2]:
        c = changepoint(solid[f"{n} /10k"], solid["event"], solid["words"])
        print(f"{n:24} steps at {c['first_event_after_step']:11} {c['before']:6.1f} -> {c['after']:6.1f} per 10k "
              f"(explains {c['variance_explained']:.0%})")
    fmt = df.assign(fmt=df["ADF (data format)"] + df["Ontologies / semantics"])
    g = fmt.groupby("event").agg(fmt=("fmt", "sum"), ai=("AI / ML", "sum"), t=("t", "first"))
    g["ai_share_of_ai_plus_fmt"] = g["ai"] / (g["ai"] + g["fmt"]) * 100
    print("\nAI share of (AI + format/ontology) mentions, by event:")
    print(g.sort_values("t")[["fmt", "ai", "ai_share_of_ai_plus_fmt"]].round(0).to_string())

    print("\n=== most AI-dense talks (AI/ML mentions per 10k words, talks > 1500 words) ===")
    top = df[df["words"] > 1500].assign(rate=lambda x: x["AI / ML"] / x["words"] * 1e4).nlargest(12, "rate")
    for _, r in top.iterrows():
        print(f"{r['upload_date']} {r['event']:11} {r['rate']:6.1f}  {r['title'][:75]}")
    print("\n=== earliest talks with AI/ML rate > 15 per 10k words ===")
    first = df[(df["words"] > 1500)].assign(rate=lambda x: x["AI / ML"] / x["words"] * 1e4)
    for _, r in first[first["rate"] > 15].sort_values(["t", "upload_date"]).head(8).iterrows():
        print(f"{r['upload_date']} {r['event']:11} {r['rate']:6.1f}  {r['title'][:75]}")

    terms = distinctive_terms(docs)
    pd.DataFrame({k: pd.Series(v) for k, v in terms.items()}).to_csv(OUT / "distinctive_terms_by_event.csv", index=False)
    print("\n=== distinctive terms per event (TF-IDF, no predefined list) ===")
    for k, v in terms.items():
        print(f"{k:12} {', '.join(v)}")


if __name__ == "__main__":
    main()
