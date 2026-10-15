import re

import pandas as pd
import pytest

from ytscribe.analysis import terms, trends
from ytscribe.analysis.config import TermsCfg, load_topics
from ytscribe.analysis.corpus import Doc
from ytscribe.analysis.lexicon import compile_topics, count, windows
from ytscribe.analysis.projection import match_sessions
from ytscribe.analysis.sentiment import topic_sentiment

CFG = load_topics()
PAT = compile_topics(CFG)
# subjective words must never be part of a topic term: tone is measured by sentiment, not by the lexicon
SUBJECTIVE = {"hard", "easy", "difficult", "good", "bad", "great", "powerful", "exciting", "amazing", "better", "best",
              "worst", "simple", "complex", "painful", "love", "hate", "awesome", "terrible"}


def doc(i, event, text, title="T", origin="captions:auto:en", dur=20.0, desc=""):
    year, _, season = event.partition(" ")
    return Doc(video_id=f"v{i}", title=title, upload_date="2024-01-01", event_year=int(year), event_season=season,
               t=int(year) + (0.25 if season == "Spring" else 0.75), event=event, origin=origin,
               slide_only="slides" in origin, description=desc, text=text, duration_min=dur)


def test_config_is_consistent():
    for group, members in CFG.groups.items():
        for m in members:
            assert m in CFG.topics, f"{group} lists unknown topic {m}"


def test_topic_terms_are_neutral():
    for name, pats in CFG.topics.items():
        words = set(re.findall(r"[a-z]{3,}", " ".join(pats).lower()))
        assert not (words & SUBJECTIVE), f"{name} uses subjective terms {words & SUBJECTIVE}"


@pytest.mark.parametrize("text,topic,n", [
    ("the pistoia alliance and the pistolia alliance and pistoria", "Pistoia Alliance", 3),
    ("a pistol alliance project, a pisto airlines slip", "Pistoia Alliance", 2),
    ("he shot a pistol and ate pasta", "Pistoia Alliance", 0),          # fuzzy must not turn every 'pistol' into Pistoia
    ("we use chat gbt and chat GPD40 and ChatGPT", "ChatGPT / GPT (AI)", 3),
    ("talk by jean-claude and then Claude Sonnet", "Claude (AI)", 1),
    ("AI is big but ai in french is a verb", "AI / ML", 1),              # case-sensitive 'AI'
    ("machine learning and Large Language Models and llm", "AI / ML", 3),
    ("Amazon Web Services and AWS and amazon", "AWS (cloud)", 2),
    ("Amazon Web Services and AWS and amazon", "Amazon (other)", 1),
    ("OPC UA and opca and LADS", "OPC UA / LADS", 3),
    ("the ontology, ontologies and AFO", "Ontologies / semantics", 3),
])
def test_lexicon_matches(text, topic, n):
    assert count(PAT[topic], text) == n


def test_windows_cover_context_and_merge_overlaps():
    text = " ".join(f"w{i}" for i in range(100)) + " ontology " + " ".join(f"x{i}" for i in range(100))
    w = windows(PAT["Ontologies / semantics"], text, words_each_side=5)
    assert len(w) == 1 and w[0].split()[5] == "ontology" and len(w[0].split()) == 11
    twice = "alpha ontology beta gamma ontologies delta"
    assert len(windows(PAT["Ontologies / semantics"], twice, 5)) == 1  # overlapping hits share a window


def _counts(rows):
    return trends.topic_counts([doc(i, e, t, origin=o) for i, (e, t, o) in enumerate(rows)], PAT)


def test_by_event_rates_and_thin_flags():
    big = "word " * 1500
    rows = [("2020 Fall", big + "ADF ADF", "captions:auto:en"), ("2020 Fall", big + "ADF", "captions:auto:en"),
            ("2020 Fall", big, "captions:auto:en"), ("2021 Spring", big + "ADF", "captions:auto:en"),
            ("2022 Fall", "slide text " * 700 + "ADF", "slides:ocr:apple-vision")]
    ev = trends.by_event(_counts(rows), ["ADF (data format)"], min_words=1000, min_talks=3)
    r = ev.set_index("event")
    assert r.loc["2020 Fall", "talks"] == 3 and not r.loc["2020 Fall", "thin"]
    assert r.loc["2020 Fall", "ADF (data format) /10k"] == pytest.approx(3 / (3 * 1500 + 3) * 1e4, rel=1e-3)
    assert r.loc["2020 Fall", "ADF (data format) %talks"] == pytest.approx(200 / 3)
    assert r.loc["2021 Spring", "thin"] and r.loc["2022 Fall", "thin"] and r.loc["2022 Fall", "slide_only"]


def test_changepoint_finds_the_step_and_skips_thin_events():
    ev = pd.DataFrame({"event": [f"{y} Fall" for y in range(2018, 2026)], "talks": [10] * 8, "words": [5000] * 8,
                       "slide_only": [False] * 8, "thin": [False, False, False, True, False, False, False, False],
                       "AI /10k": [1, 1, 2, 90, 1, 20, 21, 20]})
    cp = trends.changepoints(ev, ["AI"]).iloc[0]
    assert cp["first_event_after_step"] == "2023 Fall"       # thin 2021 outlier ignored
    assert cp["before"] < 3 < 15 < cp["after"] and cp["variance_explained"] > 0.9


def test_distinctive_terms_surface_the_events_own_vocabulary_not_fillers_or_speakers():
    filler = "like right thing know "
    docs = []
    for i in range(4):
        docs.append(doc(i, "2020 Fall", filler * 40 + "shacl shapes validation " * 15 + "common words here " * 20,
                        desc="Presented by: Zachary Smith"))
        docs.append(doc(10 + i, "2024 Fall", filler * 40 + "zachary smith " * 5 + "chromatogram viewer " * 15 + "common words here " * 20))
    df = terms.distinctive_terms(docs, TermsCfg(extra_stopwords=["like", "right", "thing", "know"], top=5))
    top2020 = list(df[df["event"] == "2020 Fall"]["term"])
    assert top2020[0] in {"shacl", "shapes", "validation", "shacl shapes", "shapes validation"}
    assert not {"like", "right", "zachary", "smith"} & set(df["term"])


def test_match_sessions_overrides_beat_fuzzy_and_one_video_is_used_once():
    docs = [doc(1, "2026 Fall", "x", title="2026 Fall AC - Automating ASM Generation at Scale"),
            doc(2, "2026 Fall", "x", title="2026 Fall AC - Introduction to US Pharmacopeia and our work on digital standards")]
    sessions = [{"title": "Automating ASM Generation at Scale"},
                {"title": "Keynote: Welcome and an Overview of USP and the Digital Projects"}]
    assert match_sessions(sessions, docs) == {"Automating ASM Generation at Scale": "v1"}   # keynote not matched by words
    m = match_sessions(sessions, docs, overrides={"Keynote: Welcome": "v2"})
    assert m["Keynote: Welcome and an Overview of USP and the Digital Projects"] == "v2" and len(m) == 2


def test_sentiment_is_relative_to_the_event_baseline():
    pos, neg = "this is wonderful and great ", "this is terrible and awful "
    docs = [doc(1, "2024 Fall", (neg * 30) + "ontology " + (neg * 30) + (pos * 400)),
            doc(2, "2024 Fall", (pos * 400))]
    scorer = lambda t: 1.0 if "wonderful" in t else (-1.0 if "terrible" in t else 0.0)  # noqa: E731
    df = topic_sentiment(docs, PAT, ["Ontologies / semantics"], window_words=5, scorer=scorer)
    r = df.iloc[0]
    assert r["windows"] == 1 and r["mean"] == -1.0 and r["relative"] < -1.0      # far below an overall-positive event
