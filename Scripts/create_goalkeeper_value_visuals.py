"""
Visuals for the Lamberts Goalkeeper Model: a league-wide value ranking,
a submodel weight breakdown, and a per-keeper pizza chart of the 13
submodel percentile scores.

Reads Lamberts Goalkeeper Model/goalkeeper_season_value_model.csv and
submodel_definitions.csv (built by build_goalkeeper_value_model.py).

Usage: python3 create_goalkeeper_value_visuals.py
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import PyPizza

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT / "Lamberts Goalkeeper Model"
VIS_DIR = MODEL_DIR / "visuals"
PLAYER_VIS_DIR = MODEL_DIR / "player_visuals"

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
GREEN = "#7ee081"
RED = "#ff6b6b"

CATEGORY_COLOR = {
    "shot_stopping": C_NAVY,
    "command_sweeping": C_PURPLE,
    "distribution": C_INDIGO,
    "risk_availability": C_PINK,
}

# submodel -> (short label, category, raw metric column, is_rate)
SUBMODEL_INFO = {
    "shot_stopping_gpae": ("Goals Prevented", "shot_stopping", "gpae_p90", False),
    "save_difficulty_weighted": ("Save Difficulty", "shot_stopping", "weighted_save_value_p90", False),
    "big_chance_denial": ("Big-Chance Denial", "shot_stopping", "big_chance_save_rate", True),
    "shot_stopping_reliability": ("Reliability", "shot_stopping", "gpae_volatility", False),
    "penalty_save_ability": ("Penalty Saves", "shot_stopping", "penalty_save_rate", True),
    "claiming_command": ("Claiming/Command", "command_sweeping", "claim_rate", True),
    "sweeper_activity": ("Sweeper Activity", "command_sweeping", "sweeper_actions_p90", False),
    "distribution_involvement": ("Distribution Vol.", "distribution", "passes_p90", False),
    "distribution_accuracy": ("Distribution Acc.", "distribution", "pass_value_over_expected_p90", False),
    "progressive_distribution": ("Progressive Value", "distribution", "xt_p90", False),
    "error_risk": ("Ball Security", "risk_availability", "risk_cost_p90", False),
    "discipline_risk": ("Discipline", "risk_availability", "discipline_cost_p90", False),
    "availability": ("Availability", "risk_availability", "availability_pct", True),
}

CATEGORY_LABEL = {
    "shot_stopping": "Shot-Stopping",
    "command_sweeping": "Claiming & Sweeping",
    "distribution": "Distribution",
    "risk_availability": "Risk & Availability",
}


def add_logo(fig, width=0.14, margin=0.016):
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


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def style_axes(ax) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(axis="x", colors=TEXT_SUB)
    ax.grid(axis="x", color=GRID_COLOR, alpha=0.35, lw=0.7)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def save_value_rankings(season: pd.DataFrame, out_path: pathlib.Path) -> None:
    ranked = season.sort_values("goalkeeper_value_index", ascending=True).copy()
    mean_idx = ranked["goalkeeper_value_index"].mean()
    colors = np.where(ranked["goalkeeper_value_index"] >= mean_idx, GREEN, RED)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), facecolor=BG)
    labels = [f"{r.player} ({r.team})" for r in ranked.itertuples()]
    y = np.arange(len(ranked))

    ax1.barh(y, ranked["goalkeeper_value_index"], color=colors, alpha=0.9)
    ax1.axvline(mean_idx, color=TEXT_MAIN, lw=1.0, ls="--", alpha=0.6, label="Pool mean")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, color=TEXT_MAIN, fontsize=9)
    ax1.set_title("Goalkeeper Value Index\n(percentile-blended composite)", color=TEXT_MAIN, fontsize=13, fontweight="bold")
    ax1.set_xlabel("Goalkeeper Value Index (0-100)", color=TEXT_SUB)
    style_axes(ax1)

    zcolors = np.where(ranked["goalkeeper_value_index_zscore"] >= 0, GREEN, RED)
    ax2.barh(y, ranked["goalkeeper_value_index_zscore"], color=zcolors, alpha=0.9)
    ax2.axvline(0, color=TEXT_MAIN, lw=1.0, ls="--", alpha=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(ranked))
    ax2.set_title("Goalkeeper Value Index (z-score)\n(standardized composite)", color=TEXT_MAIN, fontsize=13, fontweight="bold")
    ax2.set_xlabel("Weighted composite z-score", color=TEXT_SUB)
    style_axes(ax2)

    fig.suptitle("Lamberts Goalkeeper Model | Ecuador 2026 | min 450 minutes played", color=TEXT_MAIN, fontsize=15, fontweight="bold", y=1.01)
    fig.text(0.01, -0.01, "Data via Opta | Lamberts Goalkeeper Model", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_submodel_weights(defs: pd.DataFrame, out_path: pathlib.Path) -> None:
    ranked = defs.sort_values("composite_weight", ascending=True).copy()
    ranked["category"] = ranked["submodel"].map(lambda s: SUBMODEL_INFO[s][1])
    colors = ranked["category"].map(CATEGORY_COLOR)

    fig, ax = plt.subplots(figsize=(11, 8), facecolor=BG)
    y = np.arange(len(ranked))
    ax.barh(y, ranked["composite_weight"], color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(ranked["name"], color=TEXT_MAIN, fontsize=10)
    ax.set_xlabel("Weight in Goalkeeper Value Index", color=TEXT_SUB)
    ax.set_title("Submodel Weights", color=TEXT_MAIN, fontsize=15, fontweight="bold", pad=12)
    style_axes(ax)

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=CATEGORY_LABEL[k]) for k, c in CATEGORY_COLOR.items()]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9.5, labelcolor=TEXT_MAIN)

    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def format_alt(value: float, is_rate: bool) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.0f}%" if is_rate else f"{value:.2f}"


def save_player_pizza(row: pd.Series, n_pool: int, out_path: pathlib.Path) -> None:
    params, values, alt_texts, slice_colors = [], [], [], []
    for submodel, (label, category, raw_col, is_rate) in SUBMODEL_INFO.items():
        params.append(label)
        values.append(round(float(row[f"{submodel}_score"]), 1))
        alt_texts.append(format_alt(row.get(raw_col, np.nan), is_rate))
        slice_colors.append(CATEGORY_COLOR[category])

    baker = PyPizza(
        params=params, min_range=None, max_range=None,
        background_color=BG, straight_line_color="#2c3540", straight_line_lw=1,
        last_circle_color="#6b7684", last_circle_lw=1.5,
        other_circle_color="#2c3540", other_circle_lw=1, other_circle_ls="--",
    )

    fig, ax = baker.make_pizza(
        values, alt_text_values=alt_texts, figsize=(12, 13.5),
        param_location=112, slice_colors=slice_colors,
        value_bck_colors=slice_colors, value_colors=["#ffffff"] * len(params),
        blank_alpha=0.35,
        kwargs_slices=dict(edgecolor=BG, linewidth=2, zorder=2),
        kwargs_params=dict(color="#f0f2f5", fontsize=10, fontweight="bold", va="center"),
        kwargs_values=dict(color="#ffffff", fontsize=10, fontweight="bold", zorder=3,
                            bbox=dict(edgecolor="none", boxstyle="round,pad=0.25", lw=1)),
    )
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.975, row["player"], fontsize=24, fontweight="bold", ha="center", color="#ffffff")
    fig.text(0.5, 0.945, f"{row['team']} · Ecuador 2026", fontsize=12.5, ha="center", color=TEXT_SUB)
    fig.text(0.5, 0.920, f"{row['minutes']:.0f} minutes · {int(row['matches'])} matches", fontsize=10, ha="center", color="#7b8794")
    fig.text(0.5, 0.895, f"GOALKEEPER VALUE INDEX: {row['goalkeeper_value_index']:.1f} ({row['goalkeeper_value_index_pctile']:.0f}th pctile)",
             fontsize=11.5, fontweight="bold", ha="center", color=C_NAVY)

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=CATEGORY_LABEL[k]) for k, c in CATEGORY_COLOR.items()]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.012), fontsize=9.5)

    fig.text(0.02, -0.005, "Data via Opta", fontsize=8, color="#7b8794")
    fig.text(0.985, -0.005, f"vs {n_pool} keepers (>=450 min) | Lamberts Goalkeeper Model", fontsize=8, color="#7b8794", ha="right")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    season = pd.read_csv(MODEL_DIR / "goalkeeper_season_value_model.csv")
    defs = pd.read_csv(MODEL_DIR / "submodel_definitions.csv")

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    PLAYER_VIS_DIR.mkdir(parents=True, exist_ok=True)

    save_value_rankings(season, VIS_DIR / "goalkeeper_value_rankings.png")
    save_submodel_weights(defs, VIS_DIR / "submodel_weights.png")

    n_pool = len(season)
    for _, row in season.iterrows():
        out_path = PLAYER_VIS_DIR / f"{safe_name(row['player'])}_pizza.png"
        save_player_pizza(row, n_pool, out_path)
        print("Saved:", out_path)

    print("Saved:", VIS_DIR / "goalkeeper_value_rankings.png")
    print("Saved:", VIS_DIR / "submodel_weights.png")


if __name__ == "__main__":
    main()
