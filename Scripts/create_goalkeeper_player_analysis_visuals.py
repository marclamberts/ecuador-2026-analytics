"""
Visuals for player_analysis.md: charts that illustrate the specific
findings in the written analysis (strength/weakness per keeper, the
Napa distribution-volume-vs-quality gap, the "busy but leaky" shot-
stopping-vs-secondary-skills split, and the submodel specialist board),
as opposed to create_goalkeeper_value_visuals.py (profiles) or
create_goalkeeper_model_diagnostics.py / _explainer_visuals.py
(methodology).

Usage: python3 create_goalkeeper_player_analysis_visuals.py
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

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
SUBMODELS = list(SUBMODEL_CATEGORY.keys())


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


# --- 1. Strength / weakness board ----------------------------------------------

def save_strength_weakness_board(season: pd.DataFrame, out_path: pathlib.Path) -> None:
    ranked = season.sort_values("goalkeeper_value_index", ascending=False).reset_index(drop=True)
    n = len(ranked)
    row_h = 1.0

    fig, ax = plt.subplots(figsize=(15, 0.62 * n + 1.6), facecolor=BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n * row_h)
    ax.axis("off")

    headers = ["#", "Player", "Team", "Index", "Strength", "Weakness"]
    col_x = [0.15, 0.7, 2.7, 4.9, 5.9, 8.0]
    for hx, htext in zip(col_x, headers):
        ax.text(hx, n * row_h + 0.15, htext, color=TEXT_SUB, fontsize=10.5, fontweight="bold", va="bottom")

    def pill(x, y, w, text, color):
        p = FancyBboxPatch((x, y - 0.32), w, 0.6, boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=1.2, edgecolor=color, facecolor=PANEL_BG, alpha=1.0, zorder=2)
        ax.add_patch(p)
        ax.text(x + w / 2, y, text, ha="center", va="center", color=color, fontsize=8.8, fontweight="bold", zorder=3)

    for i, row in ranked.iterrows():
        y = (n - 1 - i) * row_h + row_h / 2
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y - row_h / 2), 10, row_h, facecolor=PANEL_BG, alpha=0.4, zorder=0))

        score_vals = row[[f"{s}_score" for s in SUBMODELS]]
        best_sub = score_vals.idxmax().replace("_score", "")
        worst_sub = score_vals.idxmin().replace("_score", "")
        best_val = score_vals.max()
        worst_val = score_vals.min()

        ax.text(col_x[0], y, f"{i + 1}", color=TEXT_MAIN, fontsize=10, va="center")
        ax.text(col_x[1], y, row["player"], color=TEXT_MAIN, fontsize=10.5, fontweight="bold", va="center")
        ax.text(col_x[2], y, row["team"], color=TEXT_SUB, fontsize=9, va="center")
        ax.text(col_x[3], y, f"{row['goalkeeper_value_index']:.1f}", color=C_AMBER, fontsize=10.5, fontweight="bold", va="center")

        pill(col_x[4], y, 1.9, f"{SUBMODEL_SHORT_LABEL[best_sub]} ({best_val:.0f})", GREEN)
        pill(col_x[5], y, 1.9, f"{SUBMODEL_SHORT_LABEL[worst_sub]} ({worst_val:.0f})", RED)

    ax.axhline(n * row_h + 0.05, color=GRID_COLOR, lw=1.2)
    fig.suptitle("Every Ranked Keeper's Best and Worst Submodel", color=TEXT_MAIN, fontsize=17, fontweight="bold", y=1.0)
    fig.text(0.5, 0.965, "Green = highest-scoring submodel (0-100 percentile) for that keeper; red = lowest-scoring", ha="center", color=TEXT_SUB, fontsize=10)
    fig.text(0.01, 0.01, f"Lamberts Goalkeeper Model | n={n} ranked keepers", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 2. Distribution volume vs quality (the Napa chart) -----------------------

def save_distribution_volume_vs_quality(season: pd.DataFrame, out_path: pathlib.Path) -> None:
    from adjustText import adjust_text

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
    style_axes(ax)

    x = season["distribution_involvement_score"]
    y = season["distribution_accuracy_score"]
    ax.axvline(50, color=GRID_COLOR, lw=1.2, ls="--", alpha=0.7)
    ax.axhline(50, color=GRID_COLOR, lw=1.2, ls="--", alpha=0.7)

    # Data-driven outlier: highest involvement paired with lowest accuracy,
    # among keepers who are at least above-median on involvement (so we
    # don't flag a low-volume keeper who also happens to have poor accuracy).
    above_median_volume = season[season["distribution_involvement_score"] >= season["distribution_involvement_score"].median()]
    outlier_player = above_median_volume.loc[above_median_volume["distribution_accuracy_score"].idxmin(), "player"]
    outlier_row = season[season["player"] == outlier_player].iloc[0]
    outlier_raw = outlier_row["pass_value_over_expected_p90"]
    worst_raw = season["pass_value_over_expected_p90"].min()
    is_dataset_worst = np.isclose(outlier_raw, worst_raw)

    is_outlier = season["player"] == outlier_player
    colors = np.where(is_outlier, C_AMBER, C_INDIGO)
    sizes = np.where(is_outlier, 260, 110)
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.9, edgecolors=BG, linewidths=1.2, zorder=3)

    texts = []
    for _, r in season.iterrows():
        fw = "bold" if r["player"] == outlier_player else "normal"
        texts.append(ax.text(r["distribution_involvement_score"], r["distribution_accuracy_score"], r["player"],
                              fontsize=9, color=TEXT_MAIN, fontweight=fw, zorder=4))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=TEXT_SUB, lw=0.6, alpha=0.6))

    worst_note = "the single worst number in the whole dataset" if is_dataset_worst else "one of the weakest in the pool"
    ax.annotate(f"Most involved passer among high-volume keepers,\nweakest pass value over expected\n({outlier_raw:+.2f}, {worst_note})",
                xy=(float(outlier_row["distribution_involvement_score"]), float(outlier_row["distribution_accuracy_score"])),
                xycoords="data", xytext=(0.30, 0.32), textcoords="axes fraction",
                fontsize=9.3, color=C_AMBER, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.3))

    ax.text(0.97, 0.06, "high volume\nlow accuracy", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=TEXT_SUB)
    ax.text(0.03, 0.94, "low volume\nhigh accuracy", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=TEXT_SUB)
    ax.text(0.97, 0.94, "high volume\nhigh accuracy", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=TEXT_SUB)

    ax.set_xlim(-6, 106)
    ax.set_ylim(-6, 106)
    ax.set_xlabel("Distribution Involvement score (pass volume, percentile)", color=TEXT_SUB, fontsize=10.5)
    ax.set_ylabel("Distribution Accuracy score (pass value over expected, percentile)", color=TEXT_SUB, fontsize=10.5)
    ax.set_title("Distribution Volume vs. Quality", color=TEXT_MAIN, fontsize=16, fontweight="bold", pad=12)

    fig.text(0.01, 0.01, "Lamberts Goalkeeper Model | Being heavily involved in build-up doesn't guarantee the passing adds value.", fontsize=8.5, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 3. Shot-stopping vs everything else (the "busy but leaky" chart) ---------

def save_shot_stopping_vs_secondary(season: pd.DataFrame, out_path: pathlib.Path) -> None:
    from adjustText import adjust_text

    shot_cols = [f"{s}_score" for s, c in SUBMODEL_CATEGORY.items() if c == "shot_stopping"]
    other_cols = [f"{s}_score" for s, c in SUBMODEL_CATEGORY.items() if c != "shot_stopping"]
    df = season.copy()
    df["shot_stopping_avg"] = df[shot_cols].mean(axis=1)
    df["secondary_skills_avg"] = df[other_cols].mean(axis=1)

    # Data-driven highlight selection instead of hardcoded names, so this
    # stays correct if the underlying data changes: "busy but leaky" =
    # weakest shot-stopping among keepers who are still above-average
    # elsewhere; "elite all-round" = best of both simultaneously (highest
    # min of the two axes, among keepers above 50 on both).
    both_strong = df[(df["shot_stopping_avg"] >= 50) & (df["secondary_skills_avg"] >= 50)]
    elite_allround = both_strong.loc[
        both_strong[["shot_stopping_avg", "secondary_skills_avg"]].min(axis=1).idxmax(), "player"
    ] if not both_strong.empty else None
    leaky_candidates = df[(df["shot_stopping_avg"] < 50) & (df["secondary_skills_avg"] >= 50)]
    busy_but_leaky = leaky_candidates.nsmallest(2, "shot_stopping_avg")["player"].tolist()

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
    style_axes(ax)
    ax.axvline(50, color=GRID_COLOR, lw=1.2, ls="--", alpha=0.7)
    ax.axhline(50, color=GRID_COLOR, lw=1.2, ls="--", alpha=0.7)

    highlight_names = set(busy_but_leaky) | ({elite_allround} if elite_allround else set())
    is_leaky = df["player"].isin(busy_but_leaky)
    is_elite = df["player"] == elite_allround
    colors = np.where(is_elite, GREEN, np.where(is_leaky, C_AMBER, C_NAVY))
    sizes = np.where(is_leaky | is_elite, 220, 100)
    ax.scatter(df["shot_stopping_avg"], df["secondary_skills_avg"], s=sizes, c=colors, alpha=0.9, edgecolors=BG, linewidths=1.2, zorder=3)

    texts = []
    for _, r in df.iterrows():
        fw = "bold" if r["player"] in highlight_names else "normal"
        texts.append(ax.text(r["shot_stopping_avg"], r["secondary_skills_avg"], r["player"],
                              fontsize=9, color=TEXT_MAIN, fontweight=fw, zorder=4))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=TEXT_SUB, lw=0.6, alpha=0.6))

    ax.text(0.02, 0.02, "weak shot-stopping\nweak elsewhere", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=TEXT_SUB)
    ax.text(0.02, 0.98, "weak shot-stopping\nstrong elsewhere\n(\"busy but leaky\")", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=C_AMBER)
    ax.text(0.98, 0.98, "strong shot-stopping\nstrong elsewhere\n(elite all-round)", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=GREEN)
    ax.text(0.98, 0.02, "strong shot-stopping\nweak elsewhere", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=TEXT_SUB)

    ax.set_xlim(-6, 106)
    ax.set_ylim(-6, 106)
    ax.set_xlabel("Average shot-stopping submodel score (5 submodels)", color=TEXT_SUB, fontsize=10.5)
    ax.set_ylabel("Average of the other 8 submodel scores", color=TEXT_SUB, fontsize=10.5)
    fig.suptitle('Shot-Stopping vs. Everything Else', color=TEXT_MAIN, fontsize=16, fontweight="bold", y=1.04)
    leaky_txt = " and ".join(busy_but_leaky) if busy_but_leaky else "No keeper this season"
    fig.text(0.5, 0.985, f"{leaky_txt} {'both have' if len(busy_but_leaky) > 1 else 'has'} a real standout secondary skill outweighed by the model's largest single weight: shot-stopping",
              ha="center", color=TEXT_SUB, fontsize=9.7)

    fig.text(0.01, 0.01, "Lamberts Goalkeeper Model | Both axes are simple (unweighted) averages of percentile scores within the category.", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 4. Submodel specialists board ---------------------------------------------

def save_specialists_board(season: pd.DataFrame, out_path: pathlib.Path) -> None:
    rows = []
    for subm in SUBMODELS:
        col = f"{subm}_score"
        leader = season.loc[season[col].idxmax()]
        rows.append((subm, leader["player"], leader[col]))
    board = pd.DataFrame(rows, columns=["submodel", "leader", "score"])
    win_counts = board["leader"].value_counts()

    fig, ax = plt.subplots(figsize=(11.5, 9), facecolor=BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(board))
    ax.axis("off")

    top_count = win_counts.max()
    top_leaders = win_counts[win_counts == top_count].index.tolist()
    multi_leaders = win_counts[win_counts > 1]
    if top_count <= 1:
        leader_txt = "Every submodel is led by a different keeper -- no repeat specialists this season"
    elif len(top_leaders) == 1:
        leader_txt = f"{top_leaders[0]} leads the most submodels ({top_count}); {len(multi_leaders) - 1} other keeper(s) lead 2 apiece" if len(multi_leaders) > 1 else f"{top_leaders[0]} is the only keeper to lead more than one submodel ({top_count})"
    else:
        leader_txt = f"{', '.join(top_leaders)} tie for the most submodels led ({top_count} each)"
    fig.suptitle("Submodel Specialists — Who Leads Each Category?", color=TEXT_MAIN, fontsize=17, fontweight="bold", y=1.0)
    fig.text(0.5, 0.955, leader_txt, ha="center", color=TEXT_SUB, fontsize=10)

    for i, r in board.iterrows():
        y = len(board) - 1 - i + 0.5
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y - 0.5), 10, 1.0, facecolor=PANEL_BG, alpha=0.4, zorder=0))
        color = CATEGORY_COLOR[SUBMODEL_CATEGORY[r["submodel"]]]
        ax.add_patch(plt.Rectangle((0.05, y - 0.32), 0.12, 0.64, facecolor=color, edgecolor="none", zorder=2))
        ax.text(0.35, y, SUBMODEL_SHORT_LABEL[r["submodel"]], color=TEXT_MAIN, fontsize=11, va="center", zorder=2)
        multi = win_counts[r["leader"]] > 1
        leader_color = GREEN if multi else TEXT_MAIN
        star = "  ★" if multi else ""
        ax.text(6.3, y, f"{r['leader']}{star}", color=leader_color, fontsize=11, fontweight="bold", va="center", zorder=2)
        ax.text(9.6, y, f"{r['score']:.0f}", color=TEXT_SUB, fontsize=10, va="center", ha="right", zorder=2)

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=CATEGORY_LABEL[k]) for k, c in CATEGORY_COLOR.items()]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9.5, labelcolor=TEXT_MAIN)

    fig.text(0.01, 0.02, "★ = leads more than one submodel | score = that submodel's 0-100 percentile score | Lamberts Goalkeeper Model", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    season = pd.read_csv(MODEL_DIR / "goalkeeper_season_value_model.csv")

    save_strength_weakness_board(season, VIS_DIR / "player_strength_weakness_board.png")
    save_distribution_volume_vs_quality(season, VIS_DIR / "distribution_volume_vs_quality.png")
    save_shot_stopping_vs_secondary(season, VIS_DIR / "shot_stopping_vs_secondary_skills.png")
    save_specialists_board(season, VIS_DIR / "submodel_specialists_board.png")

    for name in ["player_strength_weakness_board", "distribution_volume_vs_quality",
                 "shot_stopping_vs_secondary_skills", "submodel_specialists_board"]:
        print("Saved:", VIS_DIR / f"{name}.png")


if __name__ == "__main__":
    main()
