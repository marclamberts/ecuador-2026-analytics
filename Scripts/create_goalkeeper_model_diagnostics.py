"""
Methodology / diagnostic visuals for the Lamberts Goalkeeper Model —
these explain how the model works and whether its inputs are sound,
as opposed to create_goalkeeper_value_visuals.py which profiles players.

1. shot_stopping_diagnostic.png   - PSxG faced vs goals conceded (what
                                     shot_stopping_gpae is actually measuring)
2. shot_model_calibration.png     - reliability curve for the trained PSxG
                                     model the shot-stopping submodels sit on
3. submodel_correlation_heatmap.png - do the 13 submodels measure different
                                     things, or are several redundant?
4. bayesian_shrinkage.png         - how big_chance_denial / penalty_save_ability
                                     pull small samples toward the league mean
5. composite_score_decomposition.png - each keeper's index broken into the
                                     4 category contributions that built it
6. percentile_vs_zscore_agreement.png - where the two composite methods
                                     (percentile-blend vs z-score-blend) agree
                                     and disagree on rank

Usage: python3 create_goalkeeper_model_diagnostics.py
"""

from __future__ import annotations

import glob
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT / "Lamberts Goalkeeper Model"
DANGER_DIR = ROOT / "Danger"
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


def add_logo(fig, width=0.13, margin=0.016):
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


def load_shots() -> pd.DataFrame:
    files = [f for f in glob.glob(str(DANGER_DIR / "*_danger_models.csv")) if "eredivisie" not in f]
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["match_file"] = pathlib.Path(f).name.replace("_danger_models.csv", ".json")
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# --- 1. Shot-stopping diagnostic: PSxG faced vs goals conceded ---------------

def save_shot_stopping_diagnostic(season: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 9), facecolor=BG)
    style_axes(ax)

    x = season["psxg_faced"]
    y = season["goals_conceded"]
    lim = max(x.max(), y.max()) * 1.1
    ax.plot([0, lim], [0, lim], color=TEXT_SUB, lw=1.3, ls="--", alpha=0.7, zorder=1)

    overperform = y < x
    colors = np.where(overperform, GREEN, RED)
    sizes = 90 + season["minutes"] / season["minutes"].max() * 260
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.85, edgecolors=BG, linewidths=1.2, zorder=3)

    for _, r in season.iterrows():
        ax.annotate(r["player"], (r["psxg_faced"], r["goals_conceded"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=TEXT_MAIN)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("PSxG faced (on-target shots, season total)", color=TEXT_SUB, fontsize=11)
    ax.set_ylabel("Goals conceded (season total)", color=TEXT_SUB, fontsize=11)
    ax.set_title("What shot_stopping_gpae Measures", color=TEXT_MAIN, fontsize=16, fontweight="bold", pad=10)
    fig.text(0.5, 0.925, "Below the dashed line = conceded fewer goals than the shot quality faced predicted (overperforming)",
              ha="center", color=TEXT_SUB, fontsize=10)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markersize=10, label="Overperforming PSxG"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markersize=10, label="Underperforming PSxG"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", frameon=False, labelcolor=TEXT_MAIN, fontsize=9.5)

    fig.text(0.01, 0.01, "Marker size = minutes played | Data via Opta", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 2. Shot model calibration ------------------------------------------------

def save_shot_model_calibration(shots: pd.DataFrame, out_path: pathlib.Path) -> None:
    on_target = shots[shots["is_on_target"] == 1].copy()
    on_target["bin"] = pd.qcut(on_target["psxg"], q=8, duplicates="drop")
    calib = on_target.groupby("bin", observed=True).agg(
        predicted=("psxg", "mean"), actual=("is_goal", "mean"), n=("is_goal", "size")
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), facecolor=BG, gridspec_kw={"width_ratios": [3, 2]})
    style_axes(ax1)
    style_axes(ax2)

    ax1.plot([0, 1], [0, 1], color=TEXT_SUB, lw=1.3, ls="--", alpha=0.7, label="Perfect calibration")
    ax1.plot(calib["predicted"], calib["actual"], color=C_NAVY, lw=2.2, marker="o", markersize=8, zorder=3)
    for i, (_, r) in enumerate(calib.iterrows()):
        offset = (10, 10) if i % 2 == 0 else (10, -14)
        ax1.annotate(f"n={int(r['n'])}", (r["predicted"], r["actual"]), xytext=offset,
                     textcoords="offset points", fontsize=8, color=TEXT_SUB, ha="left")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Mean predicted PSxG (bin)", color=TEXT_SUB, fontsize=11)
    ax1.set_ylabel("Actual goal rate (bin)", color=TEXT_SUB, fontsize=11)
    ax1.set_title("PSxG Calibration (on-target shots, 8 quantile bins)", color=TEXT_MAIN, fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", frameon=False, labelcolor=TEXT_MAIN, fontsize=9.5)

    ax2.hist(on_target["psxg"], bins=25, color=C_INDIGO, alpha=0.85)
    ax2.set_xlabel("PSxG (on-target shots)", color=TEXT_SUB, fontsize=11)
    ax2.set_ylabel("Shot count", color=TEXT_SUB, fontsize=11)
    ax2.set_title(f"PSxG Distribution\n(n={len(on_target)} on-target shots)", color=TEXT_MAIN, fontsize=13, fontweight="bold")

    fig.suptitle("The Shot Model Behind Shot-Stopping Submodels", color=TEXT_MAIN, fontsize=16, fontweight="bold", y=1.03)
    fig.text(0.01, -0.02, "Every shot-stopping submodel (goals prevented, save difficulty, big-chance denial) inherits this model's calibration.",
              fontsize=9, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 3. Submodel correlation heatmap ------------------------------------------

def save_correlation_heatmap(season: pd.DataFrame, out_path: pathlib.Path) -> None:
    submodels = list(SUBMODEL_CATEGORY.keys())
    score_cols = [f"{s}_score" for s in submodels]
    corr = season[score_cols].corr(method="spearman")
    labels = [SUBMODEL_SHORT_LABEL[s] for s in submodels]

    fig, ax = plt.subplots(figsize=(12, 10.5), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("gk_div", [RED, "#232a35", GREEN])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, color=TEXT_MAIN, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(labels, color=TEXT_MAIN, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            txt_color = "#0d1117" if abs(v) > 0.55 else TEXT_MAIN
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.yaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB)
    cbar.set_label("Spearman rank correlation", color=TEXT_SUB, fontsize=10)
    cbar.outline.set_edgecolor(GRID_COLOR)

    ax.set_title("Do the 13 Submodels Measure Different Things?", color=TEXT_MAIN, fontsize=16, fontweight="bold", pad=14)
    fig.text(0.5, 0.965, "Low off-diagonal correlation = submodels capture distinct skills, not the same signal repeated",
              ha="center", color=TEXT_SUB, fontsize=10)
    fig.text(0.01, 0.01, f"n={len(season)} ranked keepers | Lamberts Goalkeeper Model", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 4. Bayesian shrinkage -----------------------------------------------------

def save_bayesian_shrinkage(season: pd.DataFrame, out_path: pathlib.Path) -> None:
    season = season.copy()
    season["big_chance_saves"] = season["big_chances_faced"] - season["big_chance_goals_conceded"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG)

    def shrinkage_panel(ax, attempts_col, raw_num_col, shrunk_col, prior_weight, title, xlabel):
        style_axes(ax)
        attempts = season[attempts_col]
        raw = (season[raw_num_col] / attempts.replace(0, np.nan)).fillna(season[shrunk_col])
        shrunk = season[shrunk_col]
        order = np.argsort(attempts.values)
        for idx in order:
            ax.plot([attempts.iloc[idx], attempts.iloc[idx]], [raw.iloc[idx], shrunk.iloc[idx]],
                    color=TEXT_SUB, lw=1.0, alpha=0.6, zorder=1)
        ax.scatter(attempts, raw, s=70, color=C_AMBER, alpha=0.9, label="Raw rate", zorder=3, edgecolors=BG, linewidths=0.8)
        ax.scatter(attempts, shrunk, s=70, color=C_NAVY, alpha=0.9, label="Shrunk rate", zorder=3, edgecolors=BG, linewidths=0.8)
        league_mean = shrunk.mean()
        ax.axhline(league_mean, color=TEXT_SUB, lw=1.2, ls="--", alpha=0.6)
        ax.text(attempts.max() * 0.98, league_mean, "  league mean", color=TEXT_SUB, fontsize=8.5, va="bottom", ha="right")
        ax.set_xlabel(xlabel, color=TEXT_SUB, fontsize=11)
        ax.set_ylabel("Save rate", color=TEXT_SUB, fontsize=11)
        ax.set_title(title, color=TEXT_MAIN, fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", frameon=False, labelcolor=TEXT_MAIN, fontsize=9.5)
        ax.text(0.02, 0.02, f"prior weight = {prior_weight:.0f} attempts at the league rate", transform=ax.transAxes,
                fontsize=8.5, color=TEXT_SUB)

    shrinkage_panel(ax1, "big_chances_faced", "big_chance_saves", "big_chance_save_rate", 6.0,
                     "Big-Chance Denial", "Big chances faced (xG >= 0.30, season total)")
    shrinkage_panel(ax2, "penalties_faced_shots", "penalty_saves", "penalty_save_rate", 8.0,
                     "Penalty Save Ability", "Penalties faced (season total)")

    fig.suptitle("Bayesian Shrinkage: Small Samples Pulled Toward the League Rate", color=TEXT_MAIN, fontsize=16, fontweight="bold", y=1.02)
    fig.text(0.01, -0.02, "Fewer attempts -> raw rate pulled further toward league mean, so one lucky/unlucky penalty doesn't swing the ranking.",
              fontsize=9, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 5. Composite score decomposition -----------------------------------------

def save_score_decomposition(season: pd.DataFrame, weights: dict, out_path: pathlib.Path) -> None:
    ranked = season.sort_values("goalkeeper_value_index", ascending=True).copy()
    cats = ["shot_stopping", "command_sweeping", "distribution", "risk_availability"]
    for cat in cats:
        cols = [f"{s}_score" for s, c in SUBMODEL_CATEGORY.items() if c == cat]
        w = [weights[s] for s, c in SUBMODEL_CATEGORY.items() if c == cat]
        ranked[f"contrib_{cat}"] = sum(ranked[col] * wi for col, wi in zip(cols, w))

    fig, ax = plt.subplots(figsize=(12, 8.5), facecolor=BG)
    style_axes(ax)
    y = np.arange(len(ranked))
    left = np.zeros(len(ranked))
    for cat in cats:
        vals = ranked[f"contrib_{cat}"].values
        ax.barh(y, vals, left=left, color=CATEGORY_COLOR[cat], alpha=0.9, label=CATEGORY_LABEL[cat],
                edgecolor=BG, linewidth=1.0)
        left += vals

    for yi, total in zip(y, ranked["goalkeeper_value_index"]):
        ax.text(total + 0.6, yi, f"{total:.1f}", va="center", fontsize=8.5, color=TEXT_MAIN, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.player} ({r.team})" for r in ranked.itertuples()], color=TEXT_MAIN, fontsize=9)
    ax.set_xlabel("Goalkeeper Value Index, decomposed by category", color=TEXT_SUB, fontsize=11)
    ax.set_title("What Builds Each Keeper's Value Index", color=TEXT_MAIN, fontsize=16, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=False, labelcolor=TEXT_MAIN, fontsize=9.5, ncol=1)

    fig.text(0.01, -0.01, "Each segment = (sum of that category's submodel percentile scores x their composite weights) | Lamberts Goalkeeper Model",
              fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 6. Percentile vs z-score agreement ---------------------------------------

def save_percentile_vs_zscore(season: pd.DataFrame, out_path: pathlib.Path) -> None:
    ranked = season.copy()
    ranked["pctile_rank"] = ranked["goalkeeper_value_index"].rank(ascending=False, method="min")
    ranked["zscore_rank"] = ranked["goalkeeper_value_index_zscore"].rank(ascending=False, method="min")

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG)
    style_axes(ax)
    n = len(ranked)
    ax.plot([1, n], [1, n], color=TEXT_SUB, lw=1.3, ls="--", alpha=0.7, zorder=1)

    diff = (ranked["pctile_rank"] - ranked["zscore_rank"]).abs()
    colors = np.where(diff <= 1, GREEN, np.where(diff <= 3, C_AMBER, RED))
    ax.scatter(ranked["pctile_rank"], ranked["zscore_rank"], s=140, c=colors, alpha=0.9, edgecolors=BG, linewidths=1.2, zorder=3)
    for _, r in ranked.iterrows():
        ax.annotate(r["player"], (r["pctile_rank"], r["zscore_rank"]), xytext=(7, 5),
                    textcoords="offset points", fontsize=8.5, color=TEXT_MAIN)

    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.5, n + 0.5)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Rank by percentile-blended index (1 = best)", color=TEXT_SUB, fontsize=11)
    ax.set_ylabel("Rank by z-score-blended index (1 = best)", color=TEXT_SUB, fontsize=11)
    ax.set_title("Do the Two Scoring Methods Agree?", color=TEXT_MAIN, fontsize=15, fontweight="bold", pad=10)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markersize=10, label="Rank differs by <=1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_AMBER, markersize=10, label="Rank differs by 2-3"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markersize=10, label="Rank differs by >3"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", frameon=False, labelcolor=TEXT_MAIN, fontsize=9)

    fig.text(0.01, 0.01, "Points off the diagonal = keepers whose ranking depends on which composite method you trust.",
              fontsize=8.5, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    season = pd.read_csv(MODEL_DIR / "goalkeeper_season_value_model.csv")
    defs = pd.read_csv(MODEL_DIR / "submodel_definitions.csv")
    weights = dict(zip(defs["submodel"], defs["composite_weight"]))
    shots = load_shots()

    save_shot_stopping_diagnostic(season, VIS_DIR / "shot_stopping_diagnostic.png")
    save_shot_model_calibration(shots, VIS_DIR / "shot_model_calibration.png")
    save_correlation_heatmap(season, VIS_DIR / "submodel_correlation_heatmap.png")
    save_bayesian_shrinkage(season, VIS_DIR / "bayesian_shrinkage.png")
    save_score_decomposition(season, weights, VIS_DIR / "composite_score_decomposition.png")
    save_percentile_vs_zscore(season, VIS_DIR / "percentile_vs_zscore_agreement.png")

    for name in ["shot_stopping_diagnostic", "shot_model_calibration", "submodel_correlation_heatmap",
                 "bayesian_shrinkage", "composite_score_decomposition", "percentile_vs_zscore_agreement"]:
        print("Saved:", VIS_DIR / f"{name}.png")


if __name__ == "__main__":
    main()
