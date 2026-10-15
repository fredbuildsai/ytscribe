"""Unsupervised topic discovery with BERTopic (embeddings -> UMAP -> HDBSCAN -> c-TF-IDF), plus `topics_over_time`.

Role in the design: an independent cross-check of the word-list trends, and the raw material for the TAXONOMY the LLM
layer will use (a human approves the taxonomy before any LLM sees it). Everything is seeded for reproducibility.
Slide-only videos are left out (OCR footers and logos would dominate their passages)."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from ytscribe.analysis.config import TopicsConfig
from ytscribe.analysis.corpus import Doc
from ytscribe.analysis.lexicon import count


@dataclass
class Passage:
    video_id: str
    event: str
    t: float
    text: str


def passages(docs: list[Doc], words: int = 150, min_words: int = 60) -> list[Passage]:
    """Non-overlapping ~`words`-word passages of every speech transcript (last short piece dropped)."""
    out = []
    for d in docs:
        if d.slide_only:
            continue
        toks = d.text.split()
        for i in range(0, len(toks), words):
            piece = toks[i:i + words]
            if len(piece) >= min_words:
                out.append(Passage(d.video_id, d.event, d.t, " ".join(piece)))
    return out


def discover(docs: list[Doc], cfg: TopicsConfig, patterns: dict, out_dir: Path, *, embedding_model: str =
             "sentence-transformers/all-MiniLM-L6-v2", min_topic_size: int = 10, passage_words: int = 150, seed: int = 42,
             nr_topics: int | None = None) -> dict:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    ps = passages(docs, passage_words)
    events = list(dict.fromkeys(p.event for p in sorted(ps, key=lambda p: p.t)))
    ev_index = {e: i for i, e in enumerate(events)}
    texts = [p.text for p in ps]
    emb = SentenceTransformer(embedding_model).encode(texts, show_progress_bar=False, batch_size=64)
    stop = list(ENGLISH_STOP_WORDS | set(cfg.terms.extra_stopwords))
    model = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed),
        hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean", cluster_selection_method="eom",
                              prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=3,
                                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b"),
        nr_topics=nr_topics, calculate_probabilities=False, verbose=False)
    topic_ids, _ = model.fit_transform(texts, emb)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"video_id": [p.video_id for p in ps], "event": [p.event for p in ps], "topic": topic_ids,
                       "text": texts})
    df[["video_id", "event", "topic"]].to_csv(out_dir / "passage_topics.csv", index=False)
    titles = {d.video_id: d.title for d in docs}
    tot = df.groupby("event").size()
    prof = (df[df.topic >= 0].groupby(["topic", "event"]).size().unstack(fill_value=0)
            .reindex(columns=events, fill_value=0))
    share = prof.div(tot.reindex(events), axis=1) * 100  # % of the event's passages
    share.round(2).to_csv(out_dir / "topic_share_by_event.csv")
    try:
        tot_df = model.topics_over_time(texts, [ev_index[p.event] for p in ps], global_tuning=True,
                                        evolution_tuning=True, nr_bins=None)
        tot_df.to_csv(out_dir / "topics_over_time_bertopic.csv", index=False)
    except Exception as e:  # the plain share table above is the authoritative output; this one is the library's view
        (out_dir / "topics_over_time_bertopic.error.txt").write_text(repr(e))

    rows, draft = [], []
    info = model.get_topic_info()
    for _, r in info[info.Topic >= 0].iterrows():
        tid = int(r.Topic)
        terms = [w for w, _ in model.get_topic(tid)][:10]
        sub = df[df.topic == tid]
        top_talks = sub["video_id"].value_counts().head(5)
        reps = sub.sample(min(3, len(sub)), random_state=seed)["text"].str.slice(0, 280).tolist()
        lex = {m: sum(1 for t in sub["text"] if count(patterns[m], t)) for m in cfg.groups.get("direction", [])}
        rows.append({"topic": tid, "passages": len(sub), "talks": sub["video_id"].nunique(), "terms": ", ".join(terms),
                     "top_talks": " | ".join(f"{titles[v][:60]} ({n})" for v, n in top_talks.items()),
                     "peak_event": share.loc[tid].idxmax(), **{f"%{e}": round(share.loc[tid, e], 1) for e in events}})
        draft.append({"id": f"T{tid:02d}", "label": " / ".join(terms[:3]), "description": "", "keep": True,
                      "merge_into": None, "top_terms": terms, "passages": len(sub), "talks": sub["video_id"].nunique(),
                      "peak_event": share.loc[tid].idxmax(), "example_talks": [titles[v] for v in top_talks.index[:3]],
                      "lexicon_overlap_passages": {k: v for k, v in lex.items() if v}, "examples": reps[:2]})
    pd.DataFrame(rows).to_csv(out_dir / "topics.csv", index=False)
    anchors = [{"id": f"A{i:02d}", "label": n, "group": g, "source": "lexicon (configs/topics)"}
               for i, (g, n) in enumerate([(g, n) for g, ms in cfg.groups.items() for n in ms], 1)]
    (out_dir / "taxonomy_draft.yaml").write_text(yaml.safe_dump(
        {"status": "DRAFT: edit label/description, set keep: false to drop, merge_into: Txx to merge. "
                   "anchors are the fixed lexicon topics (always tracked). The LLM layer reads only an approved taxonomy.",
         "anchors": anchors, "generated_from":
         {"passages": len(ps), "topics": len(draft), "outlier_passages": int((df.topic < 0).sum()),
          "embedding_model": embedding_model, "seed": seed}, "topics": draft},
        sort_keys=False, allow_unicode=True, width=120))
    return {"passages": len(ps), "topics": len(draft), "outliers": int((df.topic < 0).sum()), "out": str(out_dir)}
