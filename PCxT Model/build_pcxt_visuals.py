"""
Builds the visual set for the "Probabilistic Contextual Expected Threat
(PC-xT)" article.

Panel 1 (traditional-vs-contextual grid) reuses the real, league-wide xT
value grid fit on all 136 Ecuador 2026 matches (Cache/.xt_grid_cache.json,
the same value-iteration grid used by Scripts/xt_map.py). Everything else
in this script is a labelled conceptual/illustrative figure -- the repo's
event feed has no tracking data (no defender positions, no pressure
sensors), so per-action uncertainty, entropy, variance and causal-credit
numbers cannot be estimated from it yet. These figures exist to explain
what PC-xT would report once such data is available, not to claim it has
already been fit.

Usage: python3 "PCxT Model/build_pcxt_visuals.py"
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
XT_CACHE = os.path.join(ROOT, "Cache", ".xt_grid_cache.json")
OUT_DIR = os.path.join(HERE, "visuals")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- shared style, matched to the rest of the repo's dark visual set ----
BG = "#0d1117"
PANEL = "#11161f"
GRID_LINE = "#232b38"
TEXT_MAIN = "#e8ecf1"
TEXT_SUB = "#9aa4b2"
TEXT_FAINT = "#6b7684"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
C_GREEN = "#4fd1a5"
C_RED = "#ef5a6f"

XT_CMAP = LinearSegmentedColormap.from_list(
    "xt", [BG, "#1e3a5f", "#7b3fa0", "#c1447e", "#ff8a3d", C_AMBER]
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT_MAIN,
    "axes.edgecolor": GRID_LINE,
    "axes.labelcolor": TEXT_SUB,
    "xtick.color": TEXT_SUB,
    "ytick.color": TEXT_SUB,
})

RNG = np.random.default_rng(7)


def footer(fig, note):
    fig.text(0.985, 0.012, "Waltzing Analytics", fontsize=9, ha="right",
              color=TEXT_FAINT, style="italic")
    fig.text(0.015, 0.012, note, fontsize=7.3, color=TEXT_FAINT)


def title_block(fig, title, subtitle, y0=0.965, y1=0.934):
    fig.text(0.5, y0, title, fontsize=21, fontweight="bold", ha="center", color=TEXT_MAIN)
    fig.text(0.5, y1, subtitle, fontsize=11, ha="center", color=TEXT_SUB)


# ---------------------------------------------------------------------
# 01. Traditional value-only grid vs. a context-adjusted grid
# ---------------------------------------------------------------------
def panel_01():
    with open(XT_CACHE) as f:
        cached = json.load(f)
    xt = np.array(cached["grid"])  # (12 cols x 8 rows), real league grid

    # Illustrative context multiplier: same location, lower value when
    # pressure is high / lane is closed, higher when space and time are
    # available. This is a demonstration surface, not a fitted model --
    # it exists to show *why* a single static grid understates the range
    # of true situational value at a given cell.
    cols, rows = xt.shape
    cx, cy = np.meshgrid(np.linspace(-1, 1, rows), np.linspace(-1, 1, cols))
    pressure_field = 0.65 + 0.5 * np.exp(-((cx) ** 2 + (cy * 1.3) ** 2) * 1.8)
    contextual = xt * pressure_field

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.6))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(top=0.76, bottom=0.20, left=0.05, right=0.97, wspace=0.08)
    title_block(fig, "One Grid Cell, Many True Values",
                "Traditional xT (league-wide, location only) vs. a context-adjusted surface for the same cell",
                y0=0.955, y1=0.905)

    vmax = max(xt.max(), contextual.max())
    for ax, grid, label in zip(
        axes, [xt, contextual],
        ["Traditional xT\n(value iteration, 136 matches, location only)",
         "Same cell, contextual range\n(pressure + lane openness reshape the value)"]
    ):
        ax.set_facecolor(BG)
        im = ax.imshow(grid.T, origin="lower", cmap=XT_CMAP, vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(label, fontsize=10.5, color=TEXT_SUB, pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # pitch outline
        ax.add_patch(plt.Rectangle((-0.5, -0.5), grid.shape[0], grid.shape[1],
                                    fill=False, edgecolor=GRID_LINE, linewidth=1.2))
        ax.axvline(grid.shape[0] - 0.5 - 1.6, color=GRID_LINE, lw=0.8, ls="--", alpha=0.6)

    highlight_col, highlight_row = 9, 3
    for ax, grid in zip(axes, [xt, contextual]):
        rect = plt.Rectangle((highlight_col - 0.5, highlight_row - 0.5), 1, 1,
                              fill=False, edgecolor="white", linewidth=2.4, zorder=5)
        ax.add_patch(rect)
        ax.annotate(f"{grid[highlight_col, highlight_row]:.3f}",
                    (highlight_col, highlight_row), xytext=(highlight_col, highlight_row - 1.6),
                    ha="center", color="white", fontsize=9.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="white", lw=0.8, alpha=0.7))

    cbar_ax = fig.add_axes([0.30, 0.075, 0.4, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=XT_CMAP, norm=plt.Normalize(0, vmax)),
                       cax=cbar_ax, orientation="horizontal")
    cb.set_label("expected threat value", color=TEXT_SUB, fontsize=9)
    cb.ax.xaxis.set_tick_params(color=TEXT_SUB, labelsize=8)
    cb.outline.set_visible(False)
    plt.setp(cb.ax.get_xticklabels(), color=TEXT_SUB)

    fig.text(0.5, 0.135,
              "The highlighted cell keeps one traditional xT value all season. Its true value moves with pressure,\n"
              "passing-lane openness and time -- the premise the rest of PC-xT is built on.",
              fontsize=9.3, ha="center", color=TEXT_SUB)

    footer(fig, "Left panel: real value-iteration xT grid, Ecuador 2026 league, 136 matches (Cache/.xt_grid_cache.json). "
                 "Right panel: illustrative context surface for exposition.")
    fig.savefig(os.path.join(OUT_DIR, "01_traditional_vs_contextual_grid.png"), dpi=200, facecolor=BG)
    plt.close(fig)
    print("saved 01")


# ---------------------------------------------------------------------
# 02. Worked example waterfall
# ---------------------------------------------------------------------
def panel_02():
    steps = [
        ("Destination\nzone value", 0.080, "start"),
        ("x completion\nprobability (62%)", -0.0304, "down"),
        ("- turnover\ncost", -0.0190, "down"),
        ("+ timing\n(early lane)", 0.012, "up"),
        ("+ information\n(disguise)", 0.008, "up"),
        ("Contextual\nxT added", 0.0, "end"),
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    title_block(fig, "From 0.080 to 0.051: A Single Pass, Decomposed",
                "Worked example -- line-breaking pass into the left half-space", y0=0.955, y1=0.918)

    running = 0.0
    xs = np.arange(len(steps))
    bar_w = 0.62
    labels = []
    for i, (label, val, kind) in enumerate(steps):
        labels.append(label)
        if kind == "start":
            bottom, height = 0, val
            running = val
            color = C_AMBER
        elif kind == "end":
            bottom, height = 0, running
            color = C_GREEN
        else:
            bottom = running + val if val < 0 else running
            height = abs(val)
            running += val
            color = C_RED if val < 0 else C_GREEN
        ax.bar(i, height, bottom=bottom, width=bar_w, color=color, alpha=0.9, zorder=3,
               edgecolor=BG, linewidth=1.5)
        label_y = bottom + height + 0.004
        if kind in ("start", "end"):
            txt = f"{val if kind=='start' else running:.3f}"
        else:
            txt = f"{val:+.3f}"
        ax.text(i, label_y, txt, ha="center", va="bottom", color=TEXT_MAIN, fontsize=10.5, fontweight="bold")

    # connector lines
    cum = 0.0
    connector_ys = []
    for label, val, kind in steps:
        if kind == "start":
            cum = val
        elif kind == "end":
            pass
        else:
            cum += val
        connector_ys.append(cum)
    for i in range(len(steps) - 1):
        ax.plot([i + bar_w / 2, i + 1 - bar_w / 2], [connector_ys[i]] * 2,
                color=TEXT_FAINT, lw=1, ls="--", alpha=0.6, zorder=2)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.6, color=TEXT_SUB)
    ax.set_ylim(0, 0.10)
    ax.set_ylabel("expected threat (xT)", fontsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_LINE)
    ax.grid(axis="y", color=GRID_LINE, lw=0.6, alpha=0.6, zorder=0)
    ax.tick_params(axis="y", labelsize=9)

    ax.axhline(0.080, color=C_AMBER, lw=0.8, ls=":", alpha=0.5)
    ax.text(5.35, 0.080, "traditional xT = 0.080", color=C_AMBER, fontsize=8.7, va="center")

    footer(fig, "Illustrative worked example matching the PC-xT article. Values are the article's stated inputs, not fitted coefficients.")
    fig.savefig(os.path.join(OUT_DIR, "02_worked_example_waterfall.png"), dpi=200, facecolor=BG)
    plt.close(fig)
    print("saved 02")


# ---------------------------------------------------------------------
# 03. Uncertainty intervals (forest plot)
# ---------------------------------------------------------------------
def panel_03():
    actions = [
        ("Short lay-off, congested midfield", 0.018, 0.006, 420),
        ("Progressive pass, half-space", 0.051, 0.022, 95),
        ("Switch of play, low block", 0.034, 0.017, 140),
        ("Cutback, six-yard box", 0.091, 0.014, 260),
        ("Reverse pass, transition", 0.063, 0.031, 38),
        ("Rabona / disguised pass", 0.057, 0.045, 9),
    ]
    fig, ax = plt.subplots(figsize=(13, 6.8))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.82, bottom=0.13)
    ax.set_facecolor(BG)
    title_block(fig, "A Point Estimate Is Not the Whole Answer",
                "Contextual xT added, with 80% credible intervals -- interval width tracks how much comparable data exists",
                y0=0.955, y1=0.895)

    ys = np.arange(len(actions))[::-1]
    for y, (label, val, half_width, n) in zip(ys, actions):
        color = C_NAVY if half_width / max(val, 1e-6) < 0.5 else (C_CORAL if half_width / max(val, 1e-6) < 1.0 else C_PINK)
        ax.plot([val - half_width, val + half_width], [y, y], color=color, lw=3.2, alpha=0.85, solid_capstyle="round")
        ax.scatter([val], [y], color="white", s=70, zorder=5, edgecolor=color, linewidth=1.6)
        ax.text(val + half_width + 0.006, y, f"n={n} comparable situations", va="center", fontsize=8.4, color=TEXT_FAINT)

    ax.set_yticks(ys)
    ax.set_yticklabels([a[0] for a in actions], fontsize=10.2, color=TEXT_MAIN)
    ax.set_xlim(-0.02, 0.17)
    ax.set_ylim(-0.7, len(actions) - 0.3)
    ax.set_xlabel("contextual xT added (80% credible interval)", fontsize=10)
    ax.axvline(0, color=GRID_LINE, lw=0.9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID_LINE, lw=0.6, alpha=0.6)

    handles = [mpatches.Patch(color=C_NAVY, label="narrow: common, well-sampled situation"),
               mpatches.Patch(color=C_CORAL, label="moderate: fewer comparable cases"),
               mpatches.Patch(color=C_PINK, label="wide: rare action, low confidence")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.6, labelcolor=TEXT_SUB)

    footer(fig, "Illustrative -- interval widths shown here are for exposition; the repo's event feed does not yet carry the "
                 "tracking-derived situational counts needed to fit true credible intervals.")
    fig.savefig(os.path.join(OUT_DIR, "03_uncertainty_intervals.png"), dpi=200, facecolor=BG)
    plt.close(fig)
    print("saved 03")


# ---------------------------------------------------------------------
# 04. Variance / risk profiles -- two players, same mean
# ---------------------------------------------------------------------
def panel_04():
    a = RNG.gamma(shape=6.0, scale=0.008, size=4000)
    b = np.concatenate([
        RNG.gamma(shape=1.2, scale=0.006, size=3600),
        RNG.gamma(shape=2.0, scale=0.09, size=400),
    ])
    target_mean = 0.20
    a = a / a.mean() * target_mean
    b = b / b.mean() * target_mean

    fig, ax = plt.subplots(figsize=(12, 6.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    title_block(fig, "Same Average Threat, Different Risk Profiles",
                "Distribution of contextual xT added per possession -- both players average ~0.20 xT / 90",
                y0=0.955, y1=0.916)

    bins = np.linspace(0, 1.0, 60)
    ax.hist(a, bins=bins, color=C_NAVY, alpha=0.72, label="Player A -- consistent, low-variance carrier", density=True)
    ax.hist(b, bins=bins, color=C_PINK, alpha=0.62, label="Player B -- high-risk, high-reward carrier", density=True)
    ax.axvline(a.mean(), color=C_NAVY, lw=1.6, ls="--", alpha=0.9)
    ax.axvline(b.mean(), color=C_PINK, lw=1.6, ls="--", alpha=0.9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("xT added, single action (illustrative distribution)", fontsize=10)
    ax.set_ylabel("density", fontsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID_LINE, lw=0.6, alpha=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=TEXT_SUB)

    stats_text = (f"Player A   mean {a.mean():.3f}  ·  std {a.std():.3f}  ·  p95 {np.percentile(a,95):.3f}\n"
                  f"Player B   mean {b.mean():.3f}  ·  std {b.std():.3f}  ·  p95 {np.percentile(b,95):.3f}")
    ax.text(0.98, 0.70, stats_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=9.3, color=TEXT_SUB, family="monospace")

    footer(fig, "Synthetic distributions for exposition -- same mean, different variance and tail shape, illustrating why mean xT/90 alone can equate two very different profiles.")
    fig.savefig(os.path.join(OUT_DIR, "04_variance_risk_profiles.png"), dpi=200, facecolor=BG)
    plt.close(fig)
    print("saved 04")


# ---------------------------------------------------------------------
# 05. Entropy x option-quality quadrant
# ---------------------------------------------------------------------
def panel_05():
    players = [
        ("Y. Arce", 0.30, 0.78, C_NAVY),
        ("E. Mero", 0.82, 0.71, C_AMBER),
        ("G. Napa", 0.24, 0.31, C_INDIGO),
        ("J. Ayoví", 0.88, 0.26, C_RED),
        ("H. Piedra", 0.55, 0.62, C_PURPLE),
        ("A. Zova", 0.40, 0.52, C_CORAL),
        ("J. Quiñónez", 0.68, 0.44, C_PINK),
        ("R. Formento", 0.20, 0.55, C_GREEN),
        ("J. Morán", 0.72, 0.68, C_NAVY),
        ("P. Guzmán", 0.46, 0.20, C_INDIGO),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    title_block(fig, "Unpredictability Only Helps If the Options Are Good",
                "Action entropy vs. average option quality -- illustrative player placements",
                y0=0.958, y1=0.920)

    ax.axvline(0.5, color=GRID_LINE, lw=1)
    ax.axhline(0.5, color=GRID_LINE, lw=1)
    quad_labels = [
        (0.24, 0.94, "Predictable + Incisive\n(few options, but the right ones)"),
        (0.76, 0.94, "Unpredictable + Incisive\n(elite: several strong, varied options)"),
        (0.24, 0.06, "Predictable + Safe\n(low threat, low risk)"),
        (0.76, 0.06, "Unstructured Randomness\n(varied, but low-quality choices)"),
    ]
    for x, y, txt in quad_labels:
        ax.text(x, y, txt, transform=ax.transAxes, ha="center",
                va="top" if y > 0.5 else "bottom", fontsize=8.6, color=TEXT_FAINT, linespacing=1.5)

    for name, ent, qual, color in players:
        ax.scatter(ent, qual, s=260, color=color, alpha=0.88, edgecolor=BG, linewidth=1.4, zorder=5)
        ax.annotate(name, (ent, qual), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9.3, color=TEXT_MAIN, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("action entropy  →  how unpredictable the choice distribution is", fontsize=10)
    ax.set_ylabel("average option quality  →  how good the choices actually are", fontsize=10)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=0, length=0)

    footer(fig, "Player names are real Ecuador 2026 squad members (Aggregated/player_season_metrics.csv); "
                 "entropy/quality coordinates are illustrative placements, not fitted values.")
    fig.savefig(os.path.join(OUT_DIR, "05_entropy_quality_quadrant.png"), dpi=200, facecolor=BG)
    plt.close(fig)
    print("saved 05")


# ---------------------------------------------------------------------
# 06. Time decay of state value
# ---------------------------------------------------------------------
def panel_06():
    t = np.linspace(0, 4, 200)
    scenarios = [
        ("Open transition (defence scrambling)", 0.35, C_GREEN),
        ("Half-open (one recovering defender)", 0.75, C_AMBER),
        ("Set defence closing fast", 1.55, C_RED),
    ]
    v0 = 0.18

    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    title_block(fig, "Threat Decays While the Ball Is in Flight",
                r"$V(s,t) = V(s)\cdot e^{-\lambda t}$ -- value of the same starting state as time to execution grows",
                y0=0.955, y1=0.914)

    for label, lam, color in scenarios:
        v = v0 * np.exp(-lam * t)
        ax.plot(t, v, color=color, lw=2.6, label=f"{label}  (λ={lam})")
        ax.fill_between(t, 0, v, color=color, alpha=0.06)

    ax.scatter([0], [v0], color="white", zorder=6, s=50)
    ax.annotate(f"{v0:.2f} at the moment of reception", (0, v0), xytext=(0.35, v0 + 0.01),
                fontsize=9, color=TEXT_MAIN)

    for label, lam, color in scenarios:
        v_at_2 = v0 * np.exp(-lam * 2.0)
        ax.plot([2.0, 2.0], [0, v_at_2], color=color, lw=0.8, ls=":", alpha=0.6)
        ax.scatter([2.0], [v_at_2], color=color, s=40, zorder=5)

    ax.axvline(2.0, color=TEXT_FAINT, lw=0.8, ls="--", alpha=0.5)
    ax.text(2.02, 0.185, "2s delay", fontsize=8.5, color=TEXT_FAINT)

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 0.20)
    ax.set_xlabel("seconds elapsed before the pass is played", fontsize=10)
    ax.set_ylabel("value of the state, V(s,t)", fontsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID_LINE, lw=0.6, alpha=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=9.2, labelcolor=TEXT_SUB)

    footer(fig, "Illustrative decay curves -- lambda values are chosen for demonstration, not fit to tracked recovery speeds.")
    fig.savefig(os.path.join(OUT_DIR, "06_time_decay_value.png"), dpi=200, facecolor=BG)
    plt.close(fig)
    print("saved 06")


# ---------------------------------------------------------------------
# 07. Decision quality vs best alternative
# ---------------------------------------------------------------------
def panel_07():
    RNG2 = np.random.default_rng(3)
    n = 26
    chosen = RNG2.uniform(0.005, 0.09, n)
    best_alt = chosen + RNG2.exponential(0.025, n) * (RNG2.random(n) > 0.35)
    best_alt = np.maximum(best_alt, chosen)
    order = np.argsort(-(chosen - best_alt))
    chosen, best_alt = chosen[order], best_alt[order]
    ratio = chosen / best_alt

    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.07, right=0.97)
    ax.set_facecolor(BG)
    title_block(fig, "Was the Safe Pass Actually the Right Pass?",
                "Chosen action's xT vs. the best alternative available in the same moment -- sample of 26 on-ball decisions",
                y0=0.955, y1=0.895)

    xs = np.arange(n)
    ax.bar(xs, best_alt, width=0.62, color=GRID_LINE, alpha=0.9, label="best available alternative", zorder=2)
    colors = [C_GREEN if r > 0.85 else (C_AMBER if r > 0.5 else C_RED) for r in ratio]
    ax.bar(xs, chosen, width=0.38, color=colors, alpha=0.95, label="chosen action", zorder=3)

    ax.set_xlim(-0.8, n - 0.2)
    ax.set_xticks([])
    ax.set_ylabel("xT of action", fontsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID_LINE, lw=0.6, alpha=0.6)

    handles = [mpatches.Patch(color=GRID_LINE, label="best available alternative"),
               mpatches.Patch(color=C_GREEN, label="decision quality > 0.85 (chosen ~ best option)"),
               mpatches.Patch(color=C_AMBER, label="decision quality 0.5-0.85"),
               mpatches.Patch(color=C_RED, label="decision quality < 0.5 (a clearly better option existed)")]
    ax.set_ylim(0, 0.135)
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.4, labelcolor=TEXT_SUB, ncol=1)

    mean_dq = ratio.mean()
    ax.text(0.015, 0.965, f"Mean decision quality across sample: {mean_dq:.2f}\n"
                           f"(chosen xT ÷ max available xT, averaged over {n} decisions)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.3, color=TEXT_SUB)

    footer(fig, "Illustrative sample -- decision quality requires modelling every unplayed option at each moment, which needs full tracking data not present in this event feed.")
    fig.savefig(os.path.join(OUT_DIR, "07_decision_quality_alternatives.png"), dpi=200, facecolor=BG)
    plt.close(fig)
    print("saved 07")


# ---------------------------------------------------------------------
# 08. Five-layer framework diagram
# ---------------------------------------------------------------------
def panel_08():
    fig, ax = plt.subplots(figsize=(13, 6.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    title_block(fig, "PC-xT: Five Layers, One Action",
                "State -> Possibilities -> Probabilities -> Decision Quality -> Confidence", y0=0.955, y1=0.905)

    layers = [
        ("STATE", "location, pressure, shape,\ntime, match context", C_NAVY),
        ("POSSIBILITIES", "passes, carries, shots,\nretention options", C_INDIGO),
        ("PROBABILITIES", "success, turnover,\nfuture threat, recovery", C_PURPLE),
        ("DECISION\nQUALITY", "reward vs. risk vs.\nopportunity cost vs. timing", C_PINK),
        ("CONFIDENCE", "sample size, reliability,\nvariance, interval width", C_CORAL),
    ]
    n = len(layers)
    box_w, box_h = 1.62, 2.0
    xs = np.linspace(0.9, 9.1, n)
    y_c = 2.05

    for i, (x, (name, desc, color)) in enumerate(zip(xs, layers)):
        box = FancyBboxPatch((x - box_w / 2, y_c - box_h / 2), box_w, box_h,
                              boxstyle="round,pad=0.06,rounding_size=0.12",
                              facecolor=PANEL, edgecolor=color, linewidth=2.2, zorder=3)
        ax.add_patch(box)
        ax.text(x, y_c + 0.55, name, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=color, zorder=4, linespacing=1.3)
        ax.text(x, y_c - 0.35, desc, ha="center", va="center", fontsize=8.4,
                color=TEXT_SUB, zorder=4, linespacing=1.5)
        if i < n - 1:
            arrow = FancyArrowPatch((x + box_w / 2 + 0.03, y_c), (xs[i + 1] - box_w / 2 - 0.03, y_c),
                                     arrowstyle="-|>", mutation_scale=16, color=TEXT_FAINT, lw=1.6, zorder=2)
            ax.add_patch(arrow)

    ax.text(5.0, 0.35,
            "Threat is not a fixed property of pitch location -- it emerges from the interaction of\n"
            "space, time, information, pressure, choice and randomness.",
            ha="center", fontsize=10, color=TEXT_MAIN, style="italic", linespacing=1.6)

    footer(fig, "Conceptual framework diagram for the PC-xT model.")
    fig.savefig(os.path.join(OUT_DIR, "08_five_layer_framework.png"), dpi=200, facecolor=BG)
    plt.close(fig)
    print("saved 08")


if __name__ == "__main__":
    panel_01()
    panel_02()
    panel_03()
    panel_04()
    panel_05()
    panel_06()
    panel_07()
    panel_08()
    print("done ->", OUT_DIR)
