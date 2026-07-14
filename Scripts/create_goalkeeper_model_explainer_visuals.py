"""
More explanatory visuals for the Lamberts Goalkeeper Model: an
architecture overview infographic, the shape of each submodel's score
distribution, why the 450-minute ranking threshold was chosen, and how
much data backs each keeper's number.

Usage: python3 create_goalkeeper_model_explainer_visuals.py
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT / "Lamberts Goalkeeper Model"
VIS_DIR = MODEL_DIR / "visuals"

LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"

BG = "#0d1117"
PANEL_BG = "#101820"
GRID_COLOR = "#405160"
TEXT_MAIN = "#f8fafc"
TEXT_SUB = "#9aa4b2"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_AMBER = "#ffc247"
GREEN = "#7ee081"
RED = "#ff6b6b"

CATEGORY_COLOR = {
    "shot_stopping": C_NAVY,
    "command_sweeping": C_PURPLE,
    "distribution": C_INDIGO,
    "risk_availability": C_PINK,
}
CATEGORY_LABEL = {
    "shot_stopping": "Shot-Stopping",
    "command_sweeping": "Claiming & Sweeping",
    "distribution": "Distribution",
    "risk_availability": "Risk & Availability",
}
SUBMODEL_CATEGORY = {
    "shot_stopping_gpae": "shot_stopping",
    "save_difficulty_weighted": "shot_stopping",
    "big_chance_denial": "shot_stopping",
    "shot_stopping_reliability": "shot_stopping",
    "penalty_save_ability": "shot_stopping",
    "claiming_command": "command_sweeping",
    "sweeper_activity": "command_sweeping",
    "distribution_involvement": "distribution",
    "distribution_accuracy": "distribution",
    "progressive_distribution": "distribution",
    "error_risk": "risk_availability",
    "discipline_risk": "risk_availability",
    "availability": "risk_availability",
}
SUBMODEL_SHORT_LABEL = {
    "shot_stopping_gpae": "Goals Prevented",
    "save_difficulty_weighted": "Save Difficulty",
    "big_chance_denial": "Big-Chance Denial",
    "shot_stopping_reliability": "Reliability",
    "penalty_save_ability": "Penalty Saves",
    "claiming_command": "Claiming/Command",
    "sweeper_activity": "Sweeper Activity",
    "distribution_involvement": "Distribution Vol.",
    "distribution_accuracy": "Distribution Acc.",
    "progressive_distribution": "Progressive Value",
    "error_risk": "Ball Security",
    "discipline_risk": "Discipline",
    "availability": "Availability",
}

MIN_MINUTES_FOR_RANKING = 450.0


def add_logo(fig, width=0.12, margin=0.016):
    import matplotlib.image as mpimg
    try:
        img = mpimg.imread(LOGO_PATH)
    except FileNotFoundError:
        return
    fig_w, fig_h = fig.get_size_inches()
    img_h, img_w = img.shape[0], img.shape[1]
    width_in = width * fig_w
    height_in = width_in * (img_h / img_w)
    height = height_in / fig_h
    left = 1 - margin - width
    bottom = 1 - margin - height
    logo_ax = fig.add_axes([left, bottom, width, height], zorder=10)
    logo_ax.patch.set_alpha(0)
    logo_ax.set_xlim(0, img_w)
    logo_ax.set_ylim(img_h, 0)
    logo_ax.imshow(img)
    logo_ax.axis("off")


def style_axes(ax) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_SUB)
    ax.grid(color=GRID_COLOR, alpha=0.3, lw=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


# --- 1. Model architecture overview -------------------------------------------

def save_architecture_overview(season: pd.DataFrame, match_df: pd.DataFrame, defs: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 12), facecolor=BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.5, 0.965, "Lamberts Goalkeeper Model", ha="center", color=TEXT_MAIN, fontsize=28, fontweight="bold")
    fig.text(0.5, 0.935, "How the composite Goalkeeper Value Index is built", ha="center", color=TEXT_SUB, fontsize=13)

    def box(x, y, w, h, color, lw=1.6, fc=PANEL_BG, alpha=1.0):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                            linewidth=lw, edgecolor=color, facecolor=fc, alpha=alpha, transform=ax.transAxes, zorder=2)
        ax.add_patch(p)

    # Column 1: data sources
    sources = [
        ("player_match_metrics.csv", "Per-match GK actions, minutes,\npassing/xT (position_group == 'GK')"),
        ("Danger/*_danger_models.csv", "Trained per-shot xg / psxg / xgot\nmodel output"),
        ("team_match_metrics.csv", "Opponent cross volume\n(crosses faced)"),
        ("Event/*.json (typeId 51)", "Direct parse of tagged\n'error' events"),
    ]
    col1_x, col1_w = 0.03, 0.19
    col1_top = 0.80  # top edge of the first source box
    box_h = 0.15
    gap = 0.035
    for i, (title, desc) in enumerate(sources):
        y = col1_top - box_h - i * (box_h + gap)
        box(col1_x, y, col1_w, box_h, GRID_COLOR, fc=PANEL_BG)
        ax.text(col1_x + col1_w / 2, y + box_h - 0.028, title, ha="center", va="top", color=TEXT_MAIN,
                fontsize=10.5, fontweight="bold", transform=ax.transAxes)
        ax.text(col1_x + col1_w / 2, y + box_h / 2 - 0.02, desc, ha="center", va="center", color=TEXT_SUB,
                fontsize=8.7, transform=ax.transAxes)
    col1_bottom = col1_top - box_h - (len(sources) - 1) * (box_h + gap)
    ax.text(col1_x + col1_w / 2, col1_top + 0.03, "DATA SOURCES", ha="center", color=TEXT_SUB, fontsize=12, fontweight="bold", transform=ax.transAxes)

    # Column 2: 13 submodels grouped by category
    col2_x, col2_w = 0.29, 0.40
    col2_top = 0.83
    ax.text(col2_x + col2_w / 2, col2_top + 0.035, "13 SUBMODELS, 4 CATEGORIES", ha="center", color=TEXT_SUB, fontsize=12,
            fontweight="bold", transform=ax.transAxes)
    cats = ["shot_stopping", "command_sweeping", "distribution", "risk_availability"]
    item_h = 0.041
    cat_gap = 0.015
    y_cursor = col2_top
    for cat in cats:
        items = [(s, SUBMODEL_SHORT_LABEL[s]) for s, c in SUBMODEL_CATEGORY.items() if c == cat]
        n_items = len(items)
        cat_h = item_h + n_items * item_h
        box(col2_x, y_cursor - cat_h, col2_w, cat_h, CATEGORY_COLOR[cat], fc=PANEL_BG, lw=1.8)
        ax.text(col2_x + 0.012, y_cursor - 0.027, CATEGORY_LABEL[cat], ha="left", va="top", color=CATEGORY_COLOR[cat],
                fontsize=11, fontweight="bold", transform=ax.transAxes)
        for j, (subm, label) in enumerate(items):
            w = defs.loc[defs["submodel"] == subm, "composite_weight"].iloc[0]
            yi = y_cursor - item_h - 0.011 - j * item_h
            ax.text(col2_x + 0.03, yi, f"• {label}", ha="left", va="center", color=TEXT_MAIN, fontsize=9.3, transform=ax.transAxes)
            ax.text(col2_x + col2_w - 0.02, yi, f"w={w:.2f}", ha="right", va="center", color=TEXT_SUB, fontsize=8.7, transform=ax.transAxes)
        y_cursor -= cat_h + cat_gap
    col2_bottom = y_cursor + cat_gap

    # Column 3: composite index
    col3_x, col3_w = 0.75, 0.22
    box(col3_x, 0.55, col3_w, 0.30, C_AMBER, fc=PANEL_BG, lw=2.0)
    ax.text(col3_x + col3_w / 2, 0.82, "COMPOSITE INDEX", ha="center", color=C_AMBER, fontsize=12.5, fontweight="bold", transform=ax.transAxes)
    composite_text = (
        "goalkeeper_value_index\n= Σ (submodel percentile\n   × category weight)\n\n"
        "goalkeeper_value_index_zscore\n= Σ (submodel z-score\n   × category weight)"
    )
    ax.text(col3_x + col3_w / 2, 0.685, composite_text, ha="center", va="center", color=TEXT_MAIN, fontsize=9.3, transform=ax.transAxes)

    n_ranked = len(season)
    n_keepers_total = match_df["player"].nunique()
    n_matches = match_df["match_file"].nunique()
    stats_text = (
        f"n = {n_matches} matches\n"
        f"n = {n_keepers_total} distinct keepers observed\n"
        f"n = {n_ranked} keepers ranked (≥{int(MIN_MINUTES_FOR_RANKING)} min)\n"
        f"Ecuador 2026 season"
    )
    box(col3_x, 0.30, col3_w, 0.19, GRID_COLOR, fc=PANEL_BG, lw=1.4)
    ax.text(col3_x + col3_w / 2, 0.47, "SAMPLE", ha="center", color=TEXT_SUB, fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(col3_x + col3_w / 2, 0.385, stats_text, ha="center", va="center", color=TEXT_MAIN, fontsize=9, transform=ax.transAxes)

    # Flow bands: translucent trapezoids spanning the full height of each
    # column so the diagram reads as "all sources feed all submodels",
    # not as a single arrow pointing at one specific box.
    from matplotlib.patches import Polygon

    def flow_band(x0, top0, bottom0, x1, top1, bottom1, color):
        poly = Polygon(
            [(x0, top0), (x1, top1), (x1, bottom1), (x0, bottom0)],
            closed=True, facecolor=color, edgecolor="none", alpha=0.10, transform=ax.transAxes, zorder=0,
        )
        ax.add_patch(poly)
        mid_y0 = (top0 + bottom0) / 2
        mid_y1 = (top1 + bottom1) / 2
        arrow = FancyArrowPatch((x0, mid_y0), (x1, mid_y1), arrowstyle="-|>", mutation_scale=20,
                                 color=TEXT_SUB, lw=1.6, alpha=0.8, transform=ax.transAxes, zorder=1,
                                 shrinkA=0, shrinkB=0)
        ax.add_patch(arrow)

    flow_band(col1_x + col1_w + 0.006, col1_top, col1_bottom, col2_x - 0.006, col2_top, col2_bottom, TEXT_SUB)
    flow_band(col2_x + col2_w + 0.006, col2_top, col2_bottom, col3_x - 0.006, 0.85, 0.55, C_AMBER)

    fig.text(0.01, 0.02, "Every match has exactly one GK per team for the full 90 (no in-match keeper changes) -> all joins are exact match-level joins.",
              fontsize=9, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 2. Raw metric spread per submodel (before percentile/z-score squashing) --
#
# NOTE: percentile scores are rank transforms, so by construction they are
# always ~uniformly spread (sigma ~29 for n=15 regardless of the underlying
# data) and z-scores are always standardized to std=1 -- neither actually
# tells you whether a submodel's raw data separates keepers. This chart
# looks at the raw metric instead, min-max normalized per submodel purely
# so every row fits on one shared axis (units differ: rates, p90 counts,
# xT, etc. -- the normalization is for layout only, not a claim that the
# numbers are comparable across submodels).

RAW_METRIC_COL = {
    "shot_stopping_gpae": "gpae_p90",
    "save_difficulty_weighted": "weighted_save_value_p90",
    "big_chance_denial": "big_chance_save_rate",
    "shot_stopping_reliability": "gpae_volatility",
    "penalty_save_ability": "penalty_save_rate",
    "claiming_command": "claim_rate",
    "sweeper_activity": "sweeper_actions_p90",
    "distribution_involvement": "passes_p90",
    "distribution_accuracy": "pass_value_over_expected_p90",
    "progressive_distribution": "xt_p90",
    "error_risk": "risk_cost_p90",
    "discipline_risk": "discipline_cost_p90",
    "availability": "availability_pct",
}


def save_score_distributions(season: pd.DataFrame, out_path: pathlib.Path) -> None:
    submodels = list(SUBMODEL_CATEGORY.keys())
    fig, ax = plt.subplots(figsize=(13, 10), facecolor=BG)
    style_axes(ax)

    for i, subm in enumerate(submodels):
        y = len(submodels) - 1 - i
        col = RAW_METRIC_COL[subm]
        raw = season[col].values.astype(float)
        color = CATEGORY_COLOR[SUBMODEL_CATEGORY[subm]]
        lo, hi = raw.min(), raw.max()
        norm = (raw - lo) / (hi - lo) if hi > lo else np.full_like(raw, 0.5)

        ax.hlines(y, 0, 1, color=GRID_COLOR, lw=1.0, alpha=0.5, zorder=1)
        ax.scatter(norm, np.full_like(norm, y), s=60, color=color, alpha=0.8, edgecolors=BG, linewidths=0.8, zorder=3)
        ax.text(-0.02, y, f"{lo:.2f}", ha="right", va="center", fontsize=8, color=TEXT_SUB)
        ax.text(1.02, y, f"{hi:.2f}", ha="left", va="center", fontsize=8, color=TEXT_SUB)

    ax.set_yticks(range(len(submodels)))
    ax.set_yticklabels([SUBMODEL_SHORT_LABEL[s] for s in reversed(submodels)], color=TEXT_MAIN, fontsize=10)
    ax.set_xlim(-0.12, 1.12)
    ax.set_xticks([])
    ax.set_xlabel("Each keeper's raw value, scaled to that submodel's own min-max range", color=TEXT_SUB, fontsize=10)
    fig.suptitle("Raw Metric Spread Behind Each Submodel", color=TEXT_MAIN, fontsize=17, fontweight="bold", y=1.01)
    fig.text(0.5, 0.965, "Clustered dots = keepers are close together on that raw metric; spread-out dots = real separation exists before percentile ranking",
              ha="center", color=TEXT_SUB, fontsize=10)

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=CATEGORY_LABEL[k]) for k, c in CATEGORY_COLOR.items()]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=9.5, labelcolor=TEXT_MAIN)

    fig.text(0.01, 0.02, "Numbers at each end = actual min/max raw value (units vary by submodel) | Lamberts Goalkeeper Model", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 3. Minutes threshold sensitivity ------------------------------------------

def save_threshold_sensitivity(match_df: pd.DataFrame, out_path: pathlib.Path) -> None:
    season_minutes = match_df.groupby(["season", "team", "player_id", "player"], as_index=False)["minutes"].sum()
    thresholds = np.arange(0, 1250, 50)
    counts = [(season_minutes["minutes"] >= t).sum() for t in thresholds]

    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=BG)
    style_axes(ax)
    ax.plot(thresholds, counts, color=C_NAVY, lw=2.4, marker="o", markersize=4)
    ax.axvline(MIN_MINUTES_FOR_RANKING, color=C_AMBER, lw=1.8, ls="--", alpha=0.9)
    chosen_count = (season_minutes["minutes"] >= MIN_MINUTES_FOR_RANKING).sum()
    ax.scatter([MIN_MINUTES_FOR_RANKING], [chosen_count], color=C_AMBER, s=140, zorder=5, edgecolors=BG, linewidths=1.2)
    ax.annotate(f"chosen cutoff: {int(MIN_MINUTES_FOR_RANKING)} min\n-> {chosen_count} keepers ranked",
                (MIN_MINUTES_FOR_RANKING, chosen_count), xytext=(40, 30), textcoords="offset points",
                color=TEXT_MAIN, fontsize=10.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=TEXT_SUB, lw=1.2))

    ax.set_xlabel("Minimum minutes played this season", color=TEXT_SUB, fontsize=11)
    ax.set_ylabel("Keepers meeting the threshold", color=TEXT_SUB, fontsize=11)
    fig.suptitle("Why 450 Minutes as the Ranking Cutoff", color=TEXT_MAIN, fontsize=16, fontweight="bold", y=1.04)
    fig.text(0.5, 0.985, "Too low -> single-match small-sample noise dominates rankings. Too high -> too few keepers left to compare.",
              ha="center", color=TEXT_SUB, fontsize=10)

    fig.text(0.01, -0.01, f"n={len(season_minutes)} keeper-team-season entries | Lamberts Goalkeeper Model", fontsize=8.5, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 4. Sample-size confidence --------------------------------------------------

def save_sample_size_confidence(season: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.5), facecolor=BG)
    style_axes(ax)

    x = season["shots_faced"]
    y = season["goalkeeper_value_index"]
    sizes = 80 + season["matches"] / season["matches"].max() * 300
    colors = np.where(season["shots_faced"] >= season["shots_faced"].median(), C_NAVY, C_PINK)
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.85, edgecolors=BG, linewidths=1.2, zorder=3)
    for _, r in season.iterrows():
        ax.annotate(r["player"], (r["shots_faced"], r["goalkeeper_value_index"]), xytext=(7, 5),
                    textcoords="offset points", fontsize=8.5, color=TEXT_MAIN)

    median_shots = season["shots_faced"].median()
    ax.axvline(median_shots, color=TEXT_SUB, lw=1.2, ls="--", alpha=0.6)
    ax.text(median_shots, ax.get_ylim()[1] * 0.02 + ax.get_ylim()[0], " median shots faced", color=TEXT_SUB, fontsize=8.5, rotation=90, va="bottom")

    ax.set_xlabel("Shots faced this season (sample size behind shot-stopping submodels)", color=TEXT_SUB, fontsize=10.5)
    ax.set_ylabel("Goalkeeper Value Index", color=TEXT_SUB, fontsize=11)
    ax.set_title("How Much Data Backs Each Keeper's Number", color=TEXT_MAIN, fontsize=15.5, fontweight="bold", pad=12)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_NAVY, markersize=10, label="Above median shots faced"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_PINK, markersize=10, label="Below median shots faced"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", frameon=False, labelcolor=TEXT_MAIN, fontsize=9.5)

    fig.text(0.01, 0.01, "Marker size = matches played. Keepers on the left (fewer shots faced) have noisier shot-stopping submodel scores.",
              fontsize=8.5, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    season = pd.read_csv(MODEL_DIR / "goalkeeper_season_value_model.csv")
    match_df = pd.read_csv(MODEL_DIR / "goalkeeper_match_value.csv")
    defs = pd.read_csv(MODEL_DIR / "submodel_definitions.csv")

    save_architecture_overview(season, match_df, defs, VIS_DIR / "model_architecture_overview.png")
    save_score_distributions(season, VIS_DIR / "submodel_score_distributions.png")
    save_threshold_sensitivity(match_df, VIS_DIR / "minutes_threshold_sensitivity.png")
    save_sample_size_confidence(season, VIS_DIR / "sample_size_confidence.png")

    for name in ["model_architecture_overview", "submodel_score_distributions",
                 "minutes_threshold_sensitivity", "sample_size_confidence"]:
        print("Saved:", VIS_DIR / f"{name}.png")


if __name__ == "__main__":
    main()
