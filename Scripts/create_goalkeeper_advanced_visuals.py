"""
Advanced/innovative visuals for the Lamberts Goalkeeper Model: shot
location pitch maps, rolling form over the season, a league-wide shot
outcome funnel, a bump chart of cumulative shot-stopping rank, and
waffle grids for small-sample submodels.

Usage: python3 create_goalkeeper_advanced_visuals.py
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from mplsoccer import VerticalPitch

from _gk_shared import (
    BG, C_AMBER, C_INDIGO, C_NAVY, C_PINK, C_PURPLE, GREEN, GRID_COLOR,
    MIN_MINUTES_FOR_RANKING, PANEL_BG, RED, TEXT_MAIN, TEXT_SUB, VIS_DIR,
    add_logo, load_match, load_season, load_shots_faced_by_keeper,
    safe_name, style_axes,
)

TOP_N_PITCH_MAPS = 6
BIG_CHANCE_XG = 0.30


# --- 1. Shot-stopping pitch maps -----------------------------------------------

def save_pitch_maps(season: pd.DataFrame, faced: pd.DataFrame, out_path: pathlib.Path) -> None:
    top = season.sort_values("goalkeeper_value_index", ascending=False).head(TOP_N_PITCH_MAPS)

    fig, axes = plt.subplots(2, 3, figsize=(16, 12), facecolor=BG)
    axes_flat = axes.flatten()
    pitch = VerticalPitch(pitch_type="opta", half=True, pitch_color=PANEL_BG, line_color=GRID_COLOR, linewidth=1.2, pad_bottom=2)

    for ax, (_, row) in zip(axes_flat, top.iterrows()):
        pitch.draw(ax=ax)
        ks = faced[faced["keeper"] == row["player"]]
        goal = ks[ks["is_goal"] == 1]
        saved = ks[(ks["is_on_target"] == 1) & (ks["is_goal"] == 0)]
        off = ks[ks["is_on_target"] == 0]

        sizes_off = 25 + np.sqrt(off["psxg"].clip(lower=0)) * 220
        sizes_saved = 35 + np.sqrt(saved["psxg"].clip(lower=0)) * 260
        sizes_goal = 50 + np.sqrt(goal["psxg"].clip(lower=0)) * 300

        pitch.scatter(off["x"], off["y"], ax=ax, s=sizes_off, color=TEXT_SUB, alpha=0.45, edgecolors="none", zorder=2)
        pitch.scatter(saved["x"], saved["y"], ax=ax, s=sizes_saved, color=GREEN, alpha=0.85, edgecolors=BG, linewidths=0.8, zorder=3)
        pitch.scatter(goal["x"], goal["y"], ax=ax, s=sizes_goal, color=RED, alpha=0.9, edgecolors=BG, linewidths=0.8, zorder=4)

        save_pct = len(saved) / max(len(saved) + len(goal), 1) * 100
        ax.set_title(
            f"{row['player']} ({row['team']})\n{int(len(saved)+len(goal))} on target -- {save_pct:.0f}% saved -- GPAE {row['gpae']:+.1f}",
            color=TEXT_MAIN, fontsize=10.5, fontweight="bold", pad=8,
        )

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markersize=11, label="Goal"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markersize=11, label="Saved (on target)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEXT_SUB, markersize=9, label="Off target / blocked"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=10.5, labelcolor=TEXT_MAIN)

    fig.suptitle("Shot-Stopping Pitch Maps -- Top 6 by Goalkeeper Value Index", color=TEXT_MAIN, fontsize=17, fontweight="bold", y=1.01)
    fig.text(0.5, 0.975, "Marker size = PSxG (danger of the shot) -- direction of play is upward, goal at the top", ha="center", color=TEXT_SUB, fontsize=10)
    fig.text(0.01, -0.03, "Data via Opta | Lamberts Goalkeeper Model", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 2. Rolling form / momentum ------------------------------------------------

def save_rolling_form(match_df: pd.DataFrame, season: pd.DataFrame, out_path: pathlib.Path, window: int = 4) -> None:
    top = season.sort_values("goalkeeper_value_index", ascending=False).head(6)
    palette = [C_NAVY, C_PURPLE, C_INDIGO, C_PINK, C_AMBER, GREEN]

    m = match_df.copy()
    m["date"] = pd.to_datetime(m["date"])
    m["gpae_p90"] = (m["gpae"] / m["minutes"].replace(0, np.nan) * 90.0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    style_axes(ax)
    ax.axhline(0, color=TEXT_SUB, lw=1.2, ls="--", alpha=0.6)

    for color, (_, row) in zip(palette, top.iterrows()):
        km = m[m["player"] == row["player"]].sort_values("date")
        if len(km) < 2:
            continue
        rolling = km["gpae_p90"].rolling(window=window, min_periods=2).mean()
        ax.plot(km["date"], rolling, color=color, lw=2.4, marker="o", markersize=4, label=row["player"], alpha=0.95)

    ax.set_ylabel(f"Rolling {window}-match average GPAE per 90", color=TEXT_SUB, fontsize=11)
    ax.set_xlabel("Match date", color=TEXT_SUB, fontsize=11)
    fig.suptitle("Rolling Shot-Stopping Form Over the Season", color=TEXT_MAIN, fontsize=16, fontweight="bold", y=1.02)
    fig.text(0.5, 0.975, "Top 6 keepers by Goalkeeper Value Index -- above the line = outperforming PSxG in that stretch", ha="center", color=TEXT_SUB, fontsize=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT_MAIN, fontsize=9.5, ncol=2)
    fig.autofmt_xdate()

    fig.text(0.01, 0.01, "Lamberts Goalkeeper Model | A single bad or good match swings a short rolling window -- read direction, not the exact number.", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 3. League-wide shot outcome funnel ----------------------------------------

def _funnel_tier(ax, segments, y, bar_h, total_width, center_x=0.0, gap_frac=0.02):
    """Draw one tier of the funnel as horizontal bars centered on center_x.
    segments: list of (label, count, color). Returns list of (x0, x1, color)
    for connecting to the next tier."""
    total = sum(c for _, c, _ in segments)
    gap = total_width * gap_frac
    widths = [total_width * (c / total) for _, c, _ in segments]
    total_w_with_gaps = sum(widths) + gap * (len(segments) - 1)
    x = center_x - total_w_with_gaps / 2
    spans = []
    for (label, count, color), w in zip(segments, widths):
        rect = plt.Rectangle((x, y), w, bar_h, facecolor=color, edgecolor=BG, linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        pct = count / total * 100
        ax.text(x + w / 2, y + bar_h / 2, f"{label}\n{count} ({pct:.0f}%)", ha="center", va="center",
                 color=TEXT_MAIN, fontsize=10, fontweight="bold", zorder=4)
        spans.append((x, x + w, color))
        x += w + gap
    return spans


def _connect_tiers(ax, upper_spans, lower_spans, y_top, y_bottom, mapping):
    """mapping: list of (upper_index, lower_index) pairs to connect with a
    translucent trapezoid in the lower segment's color."""
    upper_cursor = {i: upper_spans[i][0] for i in range(len(upper_spans))}
    for upper_i, lower_i in mapping:
        lx0, lx1, lcolor = lower_spans[lower_i]
        ux0 = upper_cursor[upper_i]
        width = lx1 - lx0
        ux1 = ux0 + width
        upper_cursor[upper_i] = ux1
        poly = Polygon(
            [(ux0, y_top), (ux1, y_top), (lx1, y_bottom), (lx0, y_bottom)],
            closed=True, facecolor=lcolor, edgecolor="none", alpha=0.25, zorder=1,
        )
        ax.add_patch(poly)


def save_shot_funnel(faced: pd.DataFrame, out_path: pathlib.Path) -> None:
    total = len(faced)
    on_target = int((faced["is_on_target"] == 1).sum())
    off_target = total - on_target
    goals = int((faced["is_goal"] == 1).sum())
    saved = on_target - goals

    fig, ax = plt.subplots(figsize=(12, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    bar_h = 0.8
    width = 20.0
    tier1 = _funnel_tier(ax, [("Shots Faced", total, C_NAVY)], y=6, bar_h=bar_h, total_width=width)
    tier2 = _funnel_tier(ax, [("On Target", on_target, C_NAVY), ("Off Target / Blocked", off_target, TEXT_SUB)],
                          y=3, bar_h=bar_h, total_width=width)
    # tier3 sits under the "On Target" segment of tier2 specifically, so
    # center it under that segment's midpoint rather than under x=0.
    on_target_center = (tier2[0][0] + tier2[0][1]) / 2
    tier3_on = _funnel_tier(ax, [("Saved", saved, GREEN), ("Goal", goals, RED)],
                             y=0, bar_h=bar_h, total_width=width * (on_target / total), center_x=on_target_center)

    _connect_tiers(ax, tier1, tier2, y_top=6, y_bottom=3 + bar_h, mapping=[(0, 0), (0, 1)])
    _connect_tiers(ax, tier2, tier3_on, y_top=3, y_bottom=0 + bar_h, mapping=[(0, 0), (0, 1)])

    ax.set_xlim(-width / 2 - 1, width / 2 + 1)
    ax.set_ylim(-0.5, 6 + bar_h + 0.5)
    ax.set_aspect("auto")

    save_pct = saved / max(on_target, 1) * 100
    n_keepers = faced["keeper"].nunique()
    fig.suptitle("League-Wide Shot Outcome Funnel", color=TEXT_MAIN, fontsize=17, fontweight="bold", y=1.03)
    fig.text(0.5, 0.98, f"{save_pct:.0f}% of on-target shots faced across the league were saved | n = {total} shots, all {n_keepers} ranked keepers",
              ha="center", color=TEXT_SUB, fontsize=10.5)
    fig.text(0.01, 0.02, "Data via Opta | Lamberts Goalkeeper Model", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 4. Bump chart of cumulative shot-stopping rank ----------------------------

def save_rank_bump_chart(match_df: pd.DataFrame, season: pd.DataFrame, out_path: pathlib.Path) -> None:
    ranked_players = season.sort_values("goalkeeper_value_index", ascending=False)["player"].tolist()
    m = match_df[match_df["player"].isin(ranked_players)].copy()
    m["date"] = pd.to_datetime(m["date"])
    m["month"] = m["date"].dt.to_period("M")

    cum_rows = []
    for month in sorted(m["month"].unique()):
        to_date = m[m["month"] <= month]
        agg = to_date.groupby("player").agg(minutes=("minutes", "sum"), gpae=("gpae", "sum")).reset_index()
        agg = agg[agg["minutes"] >= 180]  # need at least 2 full matches of cumulative sample
        if agg.empty:
            continue
        agg["gpae_p90"] = agg["gpae"] / agg["minutes"] * 90.0
        agg["rank"] = agg["gpae_p90"].rank(ascending=False, method="min")
        agg["month"] = str(month)
        cum_rows.append(agg[["month", "player", "rank", "gpae_p90"]])
    cum = pd.concat(cum_rows, ignore_index=True)

    months = sorted(cum["month"].unique())
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
    style_axes(ax)

    palette = [C_NAVY, C_PURPLE, C_INDIGO, C_PINK, C_AMBER, GREEN, RED,
               "#5fb9e0", "#e096d4", "#a3a8f0", "#ffb3c9", "#ffe08a", "#a8f0b0", "#ff9d8a",
               "#4a90a4", "#b088d8", "#88b0e0", "#e0889c", "#c9a24a", "#6fbf8a"]
    final_month = months[-1]
    final_order = cum[cum["month"] == final_month].sort_values("rank")["player"].tolist()

    for i, player in enumerate(final_order):
        pdata = cum[cum["player"] == player].sort_values("month")
        color = palette[i % len(palette)]
        ax.plot(pdata["month"], pdata["rank"], color=color, lw=2.2, marker="o", markersize=6, alpha=0.9, zorder=3)
        last = pdata.iloc[-1]
        ax.annotate(player, (last["month"], last["rank"]), xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color=color, va="center", fontweight="bold")

    ax.set_ylim(cum["rank"].max() + 1, 0)
    ax.set_xticks(months)
    ax.set_xticklabels(months, rotation=30, ha="right")
    ax.set_ylabel("Cumulative shot-stopping rank (GPAE per 90, to date)", color=TEXT_SUB, fontsize=11)
    fig.suptitle("Rank Movement Over the Season", color=TEXT_MAIN, fontsize=17, fontweight="bold", y=1.02)
    fig.text(0.5, 0.975, "Cumulative-to-date shot-stopping rank only (not the full 13-submodel composite -- monthly samples are too thin for that)",
              ha="center", color=TEXT_SUB, fontsize=9.7)

    fig.text(0.01, 0.01, "Requires >=180 cumulative minutes to appear in a given month | Lamberts Goalkeeper Model", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- 5. Waffle / icon grid for small-sample submodels --------------------------

def save_waffle_grids(season: pd.DataFrame, out_path: pathlib.Path) -> None:
    ranked = season.sort_values("goalkeeper_value_index", ascending=False)
    keepers_with_big_chances = ranked[ranked["big_chances_faced"] > 0]

    n_keepers = len(keepers_with_big_chances)
    grid_cols = 5
    grid_rows = -(-n_keepers // grid_cols)  # ceil division, sized to the pool instead of a fixed count
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(18, 3.7 * grid_rows), facecolor=BG)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax in axes_flat:
        ax.axis("off")

    for ax, (_, row) in zip(axes_flat, keepers_with_big_chances.iterrows()):
        n = int(row["big_chances_faced"])
        goals = int(row["big_chance_goals_conceded"])
        saved = n - goals
        cols = min(n, 6)
        for i in range(n):
            gx, gy = i % cols, i // cols
            color = RED if i < goals else GREEN
            ax.add_patch(plt.Rectangle((gx, -gy), 0.82, 0.82, facecolor=color, edgecolor=BG, linewidth=1.5))
        ax.set_xlim(-0.3, cols + 0.3)
        rows_n = (n - 1) // cols + 1
        ax.set_ylim(-rows_n + 0.3, 1.1)
        ax.set_aspect("equal")
        ax.set_title(f"{row['player']}\n{saved}/{n} big chances saved", color=TEXT_MAIN, fontsize=10, fontweight="bold", pad=6)

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=GREEN, label="Saved"), Patch(facecolor=RED, label="Goal")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=10.5, labelcolor=TEXT_MAIN)

    fig.suptitle("Big-Chance Outcomes, One Icon Per Shot (xG >= 0.30)", color=TEXT_MAIN, fontsize=17, fontweight="bold", y=1.01)
    fig.text(0.5, 0.965, "Making the small-sample caveat visible: some of these rates are decided by 3-5 shots all season",
              ha="center", color=TEXT_SUB, fontsize=10)
    fig.text(0.01, -0.02, "Lamberts Goalkeeper Model", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    season = load_season()
    match_df = load_match()
    faced = load_shots_faced_by_keeper()
    faced_ranked = faced[faced["keeper"].isin(season["player"])]

    save_pitch_maps(season, faced, VIS_DIR / "shot_stopping_pitch_maps.png")
    save_rolling_form(match_df, season, VIS_DIR / "rolling_form_momentum.png")
    save_shot_funnel(faced_ranked, VIS_DIR / "shot_outcome_funnel.png")
    save_rank_bump_chart(match_df, season, VIS_DIR / "rank_bump_chart.png")
    save_waffle_grids(season, VIS_DIR / "big_chance_waffle_grid.png")

    for name in ["shot_stopping_pitch_maps", "rolling_form_momentum", "shot_outcome_funnel",
                 "rank_bump_chart", "big_chance_waffle_grid"]:
        print("Saved:", VIS_DIR / f"{name}.png")


if __name__ == "__main__":
    main()
