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


def talk_share_chart(ev: pd.DataFrame, topics: list[str], out: Path, substantive: int = 3,
                     title: str = "Share of talks mentioning each topic") -> Path:
    """Portrait stack, one panel per topic, fixed 0-100% scale. Light bar = talks with ANY mention; solid bar = talks with at
    least `substantive` mentions. Label = any-mention % and (talks mentioning / talks). Thin events are hatched."""
    n = len(topics)
    H = 2.3 * n + 1.6
    fig, axes = plt.subplots(n, 1, figsize=(8.2, H), sharex=True, sharey=True)
    axes = [axes] if n == 1 else list(axes)
    events, thin, talks = list(ev["event"]), list(ev["thin"]), list(ev["talks"])
    for k, (ax, name) in enumerate(zip(axes, topics, strict=False)):
        color = SERIES[k % len(SERIES)]
        any_, sub, nt = ev[f"{name} %talks"], ev[f"{name} %talks_sub"], ev[f"{name} n_talks"]
        for i in range(len(events)):
            if thin[i]:
                ax.bar(i, any_[i], 0.62, facecolor=SURFACE, edgecolor=color, lw=1.5, hatch="////", zorder=2)
            else:
                ax.bar(i, any_[i], 0.62, color=color, alpha=0.38, zorder=2)
                ax.bar(i, sub[i], 0.62, color=color, zorder=3)
            if any_[i] > 0:
                ax.text(i, any_[i] + 1.5, f"{any_[i]:.0f}%", ha="center", va="bottom", fontsize=7.5, color=INK)
                ax.text(i, any_[i] + 13, f"{int(nt[i])}/{talks[i]}", ha="center", va="bottom", fontsize=6.3, color=INK2)
        ax.set_title(name, loc="left", fontsize=11, color=INK, pad=4)
        ax.set_ylim(0, 124)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        _style(ax)
    axes[-1].set_xticks(range(len(events)), ["\n".join([e.split(" ")[0], e.split(" ")[1] if " " in e else "", f"({t})"])
                                             for e, t in zip(events, talks, strict=False)], fontsize=7.5)
    fig.suptitle(title, x=0.02, y=1 - 0.12 / H, ha="left", fontsize=13, color=INK)
    fig.text(0.02, 1 - 0.5 / H, f"Light bar = any mention; solid = {substantive}+ mentions. Label = any-mention %, then talks mentioning / talks.\n"
             "Hatched = 1-2 talks or slide-only.", fontsize=8, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / H))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def corr_heatmaps(panels: list[dict], out: Path, title: str, subtitle: str = "") -> Path:
    """Portrait stack of square heatmaps. Each panel = {"title", "matrix", "q" (FDR-adjusted p matrix or None), "vmax"}.
    Diverging colour: blue (negative) <-> grey midpoint <-> red (positive), equal arms; * q<0.05, ** q<0.01; the number
    is the coefficient. Each panel has its own scale because the statistics differ (rho vs phi); the scale is stated."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    cmap.set_bad("#f3f2ee")  # masked cells (too little data): a flat pale cell, not a colour
    n = len(panels)
    size = max(len(panels[0]["matrix"]) * 0.74 + 2.6, 7.5)
    fig, axes = plt.subplots(n, 1, figsize=(size + 1.6, (size + 0.2) * n + 1.0))
    axes = [axes] if n == 1 else list(axes)
    for ax, panel in zip(axes, panels, strict=False):
        m = panel["matrix"]
        q = panel.get("q")
        vmax = panel.get("vmax", 1.0)
        names = list(m.index)
        labels = panel.get("labels", m)  # what is printed in a cell (may differ from the value that sets the colour)
        im = ax.imshow(m.to_numpy(float), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
        for i in range(len(names)):
            for j in range(len(names)):
                v = m.iloc[i, j]
                if i == j:
                    continue
                if pd.isna(v):
                    ax.text(j, i, "–", ha="center", va="center", fontsize=9, color=INK2)  # too few data to judge
                    continue
                star = ""
                if q is not None and not pd.isna(q.iloc[i, j]):
                    star = "**" if q.iloc[i, j] < 0.01 else "*" if q.iloc[i, j] < 0.05 else ""
                ax.text(j, i, f"{labels.iloc[i, j]:{panel.get('fmt', '.2f')}}{star}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 0.6 * vmax else INK, fontweight="bold" if star else "normal")
        ax.set_xticks(range(len(names)), [t[:22] for t in names], rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), [t[:22] for t in names], fontsize=8)
        ax.set_title(panel["title"], loc="left", fontsize=10.5, color=INK, pad=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(panel.get("scale_label", ""), fontsize=7.5)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK)
    fig.text(0.02, 0.955, subtitle, fontsize=8, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out
