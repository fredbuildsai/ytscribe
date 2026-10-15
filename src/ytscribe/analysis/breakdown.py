"""What an anchor topic (AI, ASM...) is made of, over time. The anchor stays ONE topic in all the main numbers; this module
splits it two ways without changing its total:

1. named sub-terms (Claude, MCP, LLMs...): each anchor match is assigned to the first sub-pattern that matches it, so the
   parts always sum to the anchor;
2. context themes: the ~40 words around each mention are clustered (BERTopic) and each mention belongs to exactly one theme,
   which shows what the anchor is discussed IN CONNECTION WITH (data modelling, converters...). Themes are labelled by their
   top terms; naming them properly is a job for the LLM layer.
Counts are per 10k words of the event, the unit of the main charts."""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import regex

from ytscribe.analysis.config import TopicsConfig
from ytscribe.analysis.corpus import Doc
from ytscribe.analysis.lexicon import FLAGS


@dataclass
class Window:
    video_id: str
    event: str
    t: float
    text: str


def anchor_windows(docs: list[Doc], pattern, words_each_side: int = 40) -> list[Window]:
    from ytscribe.analysis.lexicon import windows

    return [Window(d.video_id, d.event, d.t, w) for d in docs if not d.slide_only
            for w in windows(pattern, d.text, words_each_side)]


def term_breakdown(docs: list[Doc], anchor_pattern, terms: dict[str, list[str]], event_words: pd.Series) -> pd.DataFrame:
    """Rows = events, columns = sub-term, values = mentions per 10k words. First matching sub-pattern wins, so each row sums
    to the anchor's own rate (provided the last sub-pattern is a catch-all)."""
    subs = [(name, regex.compile("|".join(f"(?:{p})" for p in pats), FLAGS)) for name, pats in terms.items()]
    counts: dict[tuple[str, str], int] = {}
    for d in docs:
        for m in anchor_pattern.finditer(d.text):
            hit = m.group(0)
            for name, p in subs:
                if p.search(hit):
                    counts[(d.event, name)] = counts.get((d.event, name), 0) + 1
                    break
    df = pd.DataFrame(0.0, index=event_words.index, columns=[n for n, _ in subs])
    for (ev, name), c in counts.items():
        df.loc[ev, name] = c / event_words[ev] * 1e4
    return df


ClusterFn = Callable[[list[str]], tuple[list[int], dict[int, str]]]


def bertopic_clusters(min_topic_size: int, nr_topics: int | None, seed: int = 42,
                      embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                      stopwords: list[str] | None = None) -> ClusterFn:
    """Default clustering: sentence embeddings -> UMAP -> HDBSCAN -> c-TF-IDF labels (BERTopic). -1 = unclassified."""
    def run(texts: list[str]) -> tuple[list[int], dict[int, str]]:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
        from umap import UMAP

        emb = SentenceTransformer(embedding_model).encode(texts, show_progress_bar=False, batch_size=64)
        stop = list(ENGLISH_STOP_WORDS | set(stopwords or []))
        model = BERTopic(
            umap_model=UMAP(n_neighbors=12, n_components=5, min_dist=0.0, metric="cosine", random_state=seed),
            hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean", cluster_selection_method="eom"),
            vectorizer_model=CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=2,
                                             token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b"),
            nr_topics=nr_topics, calculate_probabilities=False, verbose=False)
        ids, _ = model.fit_transform(texts, emb)
        labels = {int(t): ", ".join(w for w, _ in model.get_topic(int(t))[:4]) for t in set(ids) if t >= 0}
        return [int(i) for i in ids], labels

    return run


def context_themes(windows: list[Window], event_words: pd.Series, cluster_fn: ClusterFn) -> dict:
    """Cluster the mention windows; return {'rates': events x theme (windows per 10k words), 'themes': id -> label,
    'assignments': DataFrame}. Theme -1 is shown as 'Unclassified' (kept visible, never folded into a real theme)."""
    ids, labels = cluster_fn([w.text for w in windows])
    asg = pd.DataFrame({"video_id": [w.video_id for w in windows], "event": [w.event for w in windows], "theme": ids,
                        "text": [w.text for w in windows]})
    names = {t: f"{lab}" for t, lab in labels.items()} | {-1: "Unclassified"}
    order = [t for t, _ in sorted(asg[asg.theme >= 0]["theme"].value_counts().items(), key=lambda kv: -kv[1])] + [-1]
    rates = pd.DataFrame(0.0, index=event_words.index, columns=[names[t] for t in order if t in set(asg.theme)])
    for (ev, t), n in asg.groupby(["event", "theme"]).size().items():
        rates.loc[ev, names[t]] = n / event_words[ev] * 1e4
    return {"rates": rates, "themes": names, "assignments": asg}


def event_words_of(docs: list[Doc], include_slides: bool = True) -> pd.Series:
    """Words per event. Sub-terms use all text (so they tie to the main chart); context themes use speech only."""
    s = pd.Series({d.event: 0 for d in docs}, dtype=float)
    for d in docs:
        if include_slides or not d.slide_only:
            s[d.event] += d.words
    return s[s > 0]


__all__ = ["Window", "anchor_windows", "term_breakdown", "context_themes", "bertopic_clusters", "event_words_of",
           "TopicsConfig"]
