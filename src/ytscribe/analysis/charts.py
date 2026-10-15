"""Matplotlib charts. Palette = the reference categorical slots 1-4 (fixed order), light surface, direct labels."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
BLUE = SERIES[0]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2})


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def short_event(e: str) -> str:
    y, _, s = e.partition(" ")
    return y if not s else f"{y[2:]}{s[0]}"


def direction_chart(ev: pd.DataFrame, topics: list[str], out: Path,
                    proj: pd.DataFrame | None = None, projected_event: str | None = None, ymax: float = 52,
                    title: str = "Topic direction over time") -> Path:
    """One line per topic (max 4: fixed palette order). Events are evenly spaced (order, not calendar). Events with
    too little material are hollow and unconnected; off-scale ones get a triangle. `proj` adds a diamond +
    p10-p90 bar at `projected_event`."""
    fig, ax = plt.subplots(figsize=(11.5, 6))
    x = list(range(len(ev)))
    thin = ev["thin"]
    solid = ~thin
    for k, name in enumerate(topics[:4]):
        color, y = SERIES[k], ev[f"{name} /10k"]
        xs, ys = [i for i in x if solid[i]], [y[i] for i in x if solid[i]]
        ax.plot(xs, ys, color=color, lw=2, zorder=2)
        ax.scatter(xs, ys, s=44, color=color, edgecolor=SURFACE, lw=2, zorder=3)
        for i in x:
            if thin[i]:
                if y[i] > ymax:
                    ax.annotate(f"{y[i]:.0f}", (i, ymax), xytext=(0, 6), textcoords="offset points", ha="center",
                                fontsize=8, color=INK2, annotation_clip=False)
                    ax.scatter([i], [ymax], marker="^", s=40, facecolor=SURFACE, edgecolor=color, lw=1.8, zorder=3,
                               clip_on=False)
                else:
                    ax.scatter([i], [y[i]], s=44, facecolor=SURFACE, edgecolor=color, lw=2, zorder=3)
    last = len(topics[:4])
    if proj is not None and projected_event in set(ev["event"]):
        xl = list(ev["event"]).index(projected_event)
        p = proj.set_index("topic")
        for k, name in enumerate(topics[:4]):
            dx = (k - (last - 1) / 2) * 0.09
            r = p.loc[name]
            ax.vlines(xl + 0.28 + dx, r["p10"], r["p90"], color=SERIES[k], lw=2.5)
            ax.scatter([xl + 0.28 + dx], [r["projected"]], marker="D", s=46, facecolor=SURFACE, edgecolor=SERIES[k],
                       lw=2, zorder=4)
            ax.annotate(name, (xl + 0.28, r["projected"]), xytext=(22, 0), textcoords="offset points", va="center",
                        fontsize=10, annotation_clip=False)
        ax.text(xl - 0.15, ymax - 1, f"{projected_event}: dot = videos up so far,\ndiamond = projected from the agenda",
                ha="left", va="top", fontsize=8, color=INK2)
    else:
        for name in topics[:4]:
            ax.annotate(name, (x[-1], ev[f"{name} /10k"].iloc[-1]), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=10, annotation_clip=False)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.5, len(ev) + 2.9)
    ax.set_ylabel("mentions per 10,000 words")
    ax.set_xticks(x, [f"{e}\n({n})" for e, n in zip(ev["event"], ev["talks"], strict=False)], fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.925, "Word mentions per 10k words, by event (talks in brackets). Hollow = 1-2 talks or slide-only; "
             "triangle = off-scale value.", fontsize=9, color=INK2, va="top")
    _style(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def small_multiples(ev: pd.DataFrame, topics: list[str], out: Path, title: str, ncols: int = 2,
                    labels: dict[str, str] | None = None, note: str = "") -> Path:
    """One bar panel per topic (one hue: identity is the panel title). Label above each bar = % of talks mentioning."""
    labels = labels or {}
    nrows = -(-len(topics) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 2.6 * nrows + 1), sharex=True, squeeze=False)
    for ax, name in zip(axes.flat, topics, strict=False):
        y = ev[f"{name} /10k"]
        ax.bar(range(len(ev)), y, color=BLUE, width=0.62)
        for i, (h, pct) in enumerate(zip(y, ev[f"{name} %talks"], strict=False)):
            if h > 0.05:
                ax.text(i, h, f"{pct:.0f}%", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_title(labels.get(name, name), loc="left", fontsize=10.5, color=INK)
        ax.set_ylim(0, max(y.max() * 1.18, 1))
        if y.sum() == 0:
            ax.text(0.5, 0.5, f"no mentions in any of the {int(ev['talks'].sum())} talks", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color=INK2)
        _style(ax)
    for ax in list(axes.flat)[len(topics):]:
        ax.axis("off")
    for ax in axes.flat:
        if ax.get_visible() and ax.axison:
            ax.set_xticks(range(len(ev)), [short_event(e) for e in ev["event"]], fontsize=8)
            ax.tick_params(labelbottom=True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.945, f"Mentions per 10k words (label = % of talks mentioning). S = Spring, F = Fall. {note}",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
