"""Dictionary-based topic matching. Patterns live in configs/topics/*.yaml and are compiled with the `regex` module,
whose fuzzy syntax ({e<=1}) absorbs the way YouTube auto-captions garble names."""

import regex

from ytscribe.analysis.config import TopicsConfig

FLAGS = regex.IGNORECASE | regex.BESTMATCH


def compile_topics(cfg: TopicsConfig) -> dict[str, "regex.Pattern"]:
    return {name: regex.compile("|".join(f"(?:{p})" for p in pats), FLAGS) for name, pats in cfg.topics.items()}


def count(pattern, text: str) -> int:
    return sum(1 for _ in pattern.finditer(text))


def windows(pattern, text: str, words_each_side: int = 30) -> list[str]:
    """The text around each mention: `words_each_side` words before and after the matched term (the match itself is
    kept). Neutral by construction: the topic lexicon holds no subjective words, so what surrounds a mention is what
    a sentiment model should judge."""
    toks = text.split()
    if not toks:
        return []
    starts, pos = [], 0
    for tok in toks:  # char offset -> token index
        starts.append(pos)
        pos += len(tok) + 1
    import bisect

    out, last_end = [], -1
    for m in pattern.finditer(text):
        i = max(0, bisect.bisect_right(starts, m.start()) - 1)
        lo, hi = max(0, i - words_each_side), min(len(toks), i + words_each_side + 1)
        if lo <= last_end:  # overlapping hits share one window
            continue
        out.append(" ".join(toks[lo:hi]))
        last_end = hi - 1
    return out
