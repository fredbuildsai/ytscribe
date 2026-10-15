"""Distinctive terms per event: Fightin' Words weighted log-odds with an informative Dirichlet prior (Monroe, Colaresi &
Quinn 2008), as implemented in `scattertext`. Filler words and speaker names are removed first: on this corpus they
otherwise dominate ("like", "right", presenter first names)."""

import re

import numpy as np
import pandas as pd
import scattertext as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from ytscribe.analysis.config import TermsCfg
from ytscribe.analysis.corpus import Doc

_PRESENTED = re.compile(r"presented by:?\s*(.+)", re.I)


def speaker_stopwords(docs: list[Doc]) -> set[str]:
    """Words from the 'Presented by ...' line of each video description (names, companies)."""
    out: set[str] = set()
    for d in docs:
        for m in _PRESENTED.finditer(d.description):
            out |= {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", m.group(1))}
    return out


def distinctive_terms(docs: list[Doc], cfg: TermsCfg) -> pd.DataFrame:
    stop = list(ENGLISH_STOP_WORDS | set(cfg.extra_stopwords) | speaker_stopwords(docs))
    cv = CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=3,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b")
    X = cv.fit_transform([d.text for d in docs])
    terms = np.array(cv.get_feature_names_out())
    prior = np.asarray(X.sum(0)).ravel().astype(float)
    scorer = st.LogOddsRatioInformativeDirichletPrior(prior, cfg.prior_sigma)
    events = [d.event for d in docs]
    rows = []
    for ev in dict.fromkeys(events):
        mask = np.array([e == ev for e in events])
        a = np.asarray(X[mask].sum(0)).ravel().astype(float)
        b = np.asarray(X[~mask].sum(0)).ravel().astype(float)
        z = np.asarray(scorer.get_zeta_i_j_given_separate_counts(a, b))
        for rank, i in enumerate(np.argsort(-z)[: cfg.top], 1):
            rows.append({"event": ev, "rank": rank, "term": terms[i], "z": float(z[i])})
    return pd.DataFrame(rows)
