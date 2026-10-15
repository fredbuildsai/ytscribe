"""Charts for experiment 01. Palette: reference categorical slots 1-4 (light surface), fixed order, direct labels."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "analysis"
ev = pd.read_csv(OUT / "topic_trends_by_event.csv")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2})
thin = ev["talks"] < 3  # 2022 Spring (1 talk); 2023 Fall is slide-only: shown hollow
slide_only = ev["event"] == "2023 Fall"

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

# ---- Figure 1: direction ----
# Events are evenly spaced (order, not calendar); there are no recordings for 2018-2019. Lines connect only events
# with enough material; hollow markers (1 talk / slide-only video) are shown unconnected, off-scale ones labelled.
fig, ax = plt.subplots(figsize=(11.5, 6))
x = list(range(len(ev)))
YMAX = 52
solid = ~(thin | slide_only)
series = [("ADF (data format)", BLUE, "Allotrope Data Format"), ("Ontologies / semantics", ORANGE, "Ontologies / semantics"),
          ("ASM (simple model)", AQUA, "ASM (simple model)"), ("AI / ML", YELLOW, "AI / ML")]
for col, color, label in series:
    y = ev[f"{col} /10k"]
    xs, ys = [i for i in x if solid[i]], [y[i] for i in x if solid[i]]
    ax.plot(xs, ys, color=color, lw=2, zorder=2)
    ax.scatter(xs, ys, s=44, color=color, edgecolor=SURFACE, lw=2, zorder=3)
    for i in x:
        if not solid[i]:
            if y[i] > YMAX:
                ax.annotate(f"{y[i]:.0f}", (i, YMAX), xytext=(0, 6), textcoords="offset points", ha="center",
                            fontsize=8, color=INK2, annotation_clip=False)
                ax.scatter([i], [YMAX], marker="^", s=40, facecolor=SURFACE, edgecolor=color, lw=1.8, zorder=3, clip_on=False)
            else:
                ax.scatter([i], [y[i]], s=44, facecolor=SURFACE, edgecolor=color, lw=2, zorder=3)
proj = pd.read_csv(OUT / "projection_2026_fall.csv").set_index("topic")
xl = len(ev) - 1
for k, (col, color, _) in enumerate(series):
    r = proj.loc[col]
    dx = (k - 1.5) * 0.09  # spread the four range bars so they stay distinguishable
    ax.vlines(xl + 0.28 + dx, r["p10"], r["p90"], color=color, lw=2.5)
    ax.scatter([xl + 0.28 + dx], [r["projected"]], marker="D", s=46, facecolor=SURFACE, edgecolor=color, lw=2, zorder=4)
for col, color, label in series:
    ax.annotate(label, (xl + 0.28, proj.loc[col, "projected"]), xytext=(22, 0), textcoords="offset points", va="center",
                color=INK, fontsize=10, annotation_clip=False)
ax.text(xl - 0.15, 51, "2026 Fall: dot = the 11 videos up so far,\ndiamond = projected incl. 9 sessions not yet up",
        ha="left", va="top", fontsize=8, color=INK2)
ax.axvline(0.5, color=GRID, lw=1, ls=(0, (4, 4)))
ax.text(0.5, YMAX * 1.0, " no recordings 2018-19", fontsize=8, color=INK2, va="top")
ax.set_ylim(0, YMAX)
ax.set_xlim(-0.5, len(ev) + 2.9)
ax.set_ylabel("mentions per 10,000 words")
ax.set_xticks(x, [f"{e}\n({n})" for e, n in zip(ev["event"], ev["talks"], strict=False)], fontsize=8)
fig.suptitle("Allotrope Connect: from the format, to ASM, to AI", x=0.01, ha="left", fontsize=13, color=INK)
fig.text(0.01, 0.925, "Word mentions per 10k words, by event (talks in brackets). Hollow = 1 talk or slide-only video; triangle = off-scale value.\n"
         "Diamond = 2026 Fall projected from the published agenda for talks not yet on YouTube (bar = p10 to p90 of the estimate).",
         fontsize=9, color=INK2, va="top", linespacing=1.5)
style(ax)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(OUT / "chart_direction.png", dpi=160)

# ---- Figure 2: partners, small multiples (one series each) ----
panels = [("Pistoia Alliance", "Pistoia Alliance"), ("OPC UA / LADS", "OPC UA / LADS"), ("IDMP", "IDMP (ISO medicinal product standard)")]
fig, axes = plt.subplots(len(panels), 1, figsize=(11, 7.6), sharex=True)
for ax, (col, title) in zip(axes, panels, strict=False):
    y = ev[f"{col} /10k"]
    bars = ax.bar(range(len(ev)), y, color=BLUE, width=0.6)
    for i, (h, pct) in enumerate(zip(y, ev[f"{col} %talks"], strict=False)):
        if h > 0.05:
            ax.text(i, h, f"{pct:.0f}%", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel("per 10k words", fontsize=9)
    style(ax)
axes[-1].set_xticks(range(len(ev)), [f"{e}\n({n})" for e, n in zip(ev["event"], ev["talks"], strict=False)], fontsize=8)
fig.suptitle("Partners and standards bodies: mentions per 10k words (label = % of talks mentioning)", x=0.01, ha="left",
             fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(OUT / "chart_partners.png", dpi=160)
print("ok")


# ---- Figure 3: AI and cloud companies (small multiples) ----
cos = [("Anthropic (company)", "Anthropic (company)"), ("Claude (AI)", "Claude (AI model)"),
       ("OpenAI (company)", "OpenAI (company)"), ("ChatGPT / GPT (AI)", "ChatGPT / GPT (AI model)"),
       ("AWS (cloud)", "AWS (cloud)"), ("Amazon (other)", "Amazon, other mentions"),
       ("Microsoft / Azure (cloud)", "Microsoft / Azure (cloud)"), ("Copilot (AI)", "Copilot (AI assistant)"),
       ("Google (company)", "Google (company, not necessarily AI)"), ("Gemini (AI)", "Gemini (AI model)"),
       ("MCP (protocol)", "MCP (Model Context Protocol)")]
short = {"2017": "2017", "2020 Fall": "20F", "2021 Spring": "21S", "2021 Fall": "21F", "2022 Spring": "22S", "2022 Fall": "22F",
         "2023 Spring": "23S", "2023 Fall": "23F", "2024 Spring": "24S", "2024 Fall": "24F", "2025 Fall": "25F",
         "2026 Spring": "26S", "2026 Fall": "26F"}
fig, axes = plt.subplots(4, 3, figsize=(13, 10), sharex=True)
for ax, (col, title) in zip(axes.flat, cos, strict=False):
    y = ev[f"{col} /10k"]
    ax.bar(range(len(ev)), y, color=BLUE, width=0.62)
    for i, (h, pct) in enumerate(zip(y, ev[f"{col} %talks"], strict=False)):
        if h > 0.05:
            ax.text(i, h, f"{pct:.0f}%", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_ylim(0, max(y.max() * 1.18, 1))
    if y.sum() == 0:
        ax.text(0.5, 0.5, "no mentions in any of the 145 talks", transform=ax.transAxes, ha="center", va="center", fontsize=10, color=INK2)
    style(ax)
for ax in list(axes[-1]) + [axes[-2][-1]]:
    ax.tick_params(labelbottom=True)
    ax.set_xticks(range(len(ev)), [short[e] for e in ev["event"]], fontsize=8)
fig.suptitle("AI and cloud companies named in the talks: mentions per 10k words (label = % of talks mentioning)", x=0.01,
             ha="left", fontsize=12, color=INK)
fig.text(0.01, 0.945, "Events: S = Spring, F = Fall. 2023 Fall is slide-only OCR text; 2022 Spring is one talk. Claude "
         "peak is mostly one 2026 Spring talk.", fontsize=8.5, color=INK2)
axes[-1][-1].axis("off")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "chart_ai_cloud.png", dpi=160)
print("ok3")
