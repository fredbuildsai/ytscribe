"""Matplotlib charts. Palette = the reference categorical slots 1-4 (fixed order), light surface, direct labels."""

from pathlib import Path

import matplotlib
import matplotlib.patches  # noqa: F401

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
BLUE = SERIES[0]
CATEGORICAL8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # fixed order
NEUTRAL = "#9a9a96"  # "unclassified" / "unspecified" / "other": never a series colour
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2})


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def short_event(e: str) -> str:
    """'2026 H2' -> '26H2' (compact axis label)."""
    y, _, s = e.partition(" ")
    return y if not s else f"{y[2:]}{s}"


def stacked_chart(ev: pd.DataFrame, topics: list[str], out: Path, proj: pd.DataFrame | None = None,
                  projected_event: str | None = None, steps: pd.DataFrame | None = None, ymax: float | None = None,
                  title: str = "Topic direction over time") -> Path:
    """Portrait stack: one panel per topic, ALL panels on the same y scale (mentions per 10k words), so bar heights are
    comparable across topics. Thin events (1-2 talks, slide-only) are hollow; values above the scale get a triangle and
    their number. `proj` adds a diamond + p10-p90 bar at `projected_event`; `steps` (changepoints) draws the fitted
    before/after levels as dashed lines."""
    n = len(topics)
    if ymax is None:  # fit every normal event (and the projection's p90); thin outliers may still exceed it (triangle)
        top = max(ev.loc[~ev["thin"], [f"{t} /10k" for t in topics]].max().max(),
                  proj["p90"].max() if proj is not None else 0)
        ymax = -(-top * 1.04 // 5) * 5
    fig, axes = plt.subplots(n, 1, figsize=(8.2, 2.35 * n + 1.5), sharex=True, sharey=True)
    axes = [axes] if n == 1 else list(axes)
    x = list(range(len(ev)))
    events = list(ev["event"])
    thin = list(ev["thin"])
    solid_idx = [i for i in x if not thin[i]]
    pj = proj.set_index("topic") if proj is not None else None
    stp = steps.set_index("topic") if steps is not None else None
    for k, (ax, name) in enumerate(zip(axes, topics, strict=False)):
        color, y = SERIES[k % len(SERIES)], ev[f"{name} /10k"].to_numpy(float)
        for i in x:
            h = min(y[i], ymax)
            if thin[i]:
                ax.bar(i, h, width=0.62, facecolor=SURFACE, edgecolor=color, lw=1.6, hatch="////", zorder=2)
            else:
                ax.bar(i, h, width=0.62, color=color, zorder=2)
            if y[i] > ymax:
                ax.annotate(f"{y[i]:.0f}", (i, ymax), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=8, color=INK2)
                ax.scatter([i], [ymax], marker="^", s=34, facecolor=SURFACE, edgecolor=color, lw=1.6, zorder=3, clip_on=False)
        if stp is not None and name in stp.index and stp.loc[name, "first_event_after_step"] in events:
            r = stp.loc[name]
            k0 = events.index(r["first_event_after_step"])
            lo, hi = solid_idx[0], solid_idx[-1]
            ax.hlines(r["before"], lo - 0.4, k0 - 0.5, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=4)
            ax.hlines(r["after"], k0 - 0.5, hi + 0.4, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=4)
        if pj is not None and projected_event in events:
            xl = events.index(projected_event)
            r = pj.loc[name]
            ax.vlines(xl + 0.62, r["p10"], r["p90"], color=color, lw=2.5, zorder=4)
            ax.scatter([xl + 0.62], [r["projected"]], marker="D", s=44, facecolor=SURFACE, edgecolor=color, lw=2, zorder=5)
            ax.annotate(f"{r['projected']:.0f}", (xl + 0.62, r["projected"]), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)
        ax.set_title(name, loc="left", fontsize=11, color=INK, pad=4)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("per 10k words", fontsize=8.5)
        ax.axvline(0.5, color=GRID, lw=1, ls=(0, (4, 4)))
        _style(ax)
    axes[0].text(0.55, ymax * 0.97, "no recordings 2018-19", fontsize=7.5, color=INK2, va="top")
    axes[-1].set_xlim(-0.6, len(ev) + 0.3)
    axes[-1].set_xticks(x, ["\n".join([e.split(" ")[0], e.split(" ")[1] if " " in e else "", f"({t})"])
                            for e, t in zip(events, ev["talks"], strict=False)], fontsize=7.5)
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=13, color=INK)
    note = ("Same y scale in every panel. Hatched = 1-2 talks or slide-only. ^ = above scale (value shown). "
            "Dashed = best single step.")
    if proj is not None:
        note += f" Diamond = {projected_event} projected from the agenda (bar = p10 to p90)."
    fig.text(0.02, 0.972, note, fontsize=8, color=INK2, va="top", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out, dpi=170)
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
    fig.text(0.01, 0.945, f"Mentions per 10k words (label = % of talks mentioning). H1 / H2 = first / second-half workshop. {note}",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def composition_chart(panels: list[dict], ev: pd.DataFrame, out: Path, title: str, subtitle: str = "") -> Path:
    """Portrait stack of composition panels for ONE anchor topic. Each panel = {"title", "rates"} where rates is
    events x parts (mentions or windows per 10k words); bars are stacked, so a bar's height is the anchor's rate and the
    colours show what it is made of. All panels share one y scale. Parts beyond 8 are folded into 'Other'; neutral grey is
    reserved for unclassified / unspecified / other. Thin events are hatched."""
    events = list(ev["event"])
    thin = dict(zip(ev["event"], ev["thin"], strict=False))
    talks = dict(zip(ev["event"], ev["talks"], strict=False))
    ymax = max(p["rates"].reindex(events).fillna(0).sum(axis=1).max() for p in panels) * 1.12
    n = len(panels)
    H = 4.6 * n + 1.5
    fig, axes = plt.subplots(n, 1, figsize=(11.5, H), sharex=True, sharey=True)
    axes = [axes] if n == 1 else list(axes)
    for ax, panel in zip(axes, panels, strict=False):
        rates = panel["rates"].reindex(events).fillna(0.0)
        neutral = [c for c in rates.columns if c in ("Unclassified", "AI, unspecified") or c.startswith("Other")]
        colored = [c for c in rates.columns if c not in neutral]
        colored = sorted(colored, key=lambda c: -rates[c].sum())
        if len(colored) > 8:
            rates["Other (minor)"] = rates[colored[8:]].sum(axis=1)
            neutral.append("Other (minor)")
            colored = colored[:8]
        stack = colored + neutral
        color_of = {c: CATEGORICAL8[i] for i, c in enumerate(colored)} | {c: NEUTRAL for c in neutral}
        bottoms = pd.Series(0.0, index=events)
        for c in stack:
            for i, e in enumerate(events):
                h = rates.loc[e, c]
                if h <= 0:
                    continue
                if thin[e]:
                    ax.bar(i, h, 0.62, bottom=bottoms[e], facecolor=SURFACE, edgecolor=color_of[c], lw=1.2, hatch="////", zorder=2)
                else:
                    ax.bar(i, h, 0.62, bottom=bottoms[e], color=color_of[c], edgecolor=SURFACE, lw=1.0, zorder=2)
                bottoms[e] += h
        for i, e in enumerate(events):
            if bottoms[e] > 0:
                ax.text(i, bottoms[e] + ymax * 0.01, f"{bottoms[e]:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        handles = [matplotlib.patches.Patch(facecolor=color_of[c], label=c[:48]) for c in reversed(stack)]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8, handlelength=1.1,
                  labelcolor=INK)
        ax.set_title(panel["title"], loc="left", fontsize=11, color=INK, pad=4)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("per 10k words", fontsize=8.5)
        _style(ax)
    axes[-1].set_xticks(range(len(events)), ["\n".join([e.split(" ")[0], e.split(" ")[1] if " " in e else "", f"({talks[e]})"])
                                             for e in events], fontsize=7.5)
    fig.suptitle(title, x=0.02, y=1 - 0.12 / H, ha="left", fontsize=13, color=INK)
    fig.text(0.02, 1 - 0.5 / H, subtitle + " Same y scale in every panel; label above a bar = total. Hatched = 1-2 talks or "
             "slide-only.", fontsize=8, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 0.78, 1 - 0.85 / H))
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out
