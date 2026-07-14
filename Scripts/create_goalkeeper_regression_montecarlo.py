"""
Statistical layer on top of the Lamberts Goalkeeper Model: linear
regression of shot-stopping form over the season (is a keeper's trend
real or noise?), a Monte Carlo simulation of shot-by-shot luck (how much
of GPAE is skill vs. variance in the shots a keeper happened to face),
and a Monte Carlo bootstrap of the season-long ranking itself (how
stable is each keeper's rank to which matches they happened to have a
good or bad game in?).

Usage: python3 create_goalkeeper_regression_montecarlo.py
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from lamberts_goalkeeper_model.model import _build_season_table, _score_submodels

from _gk_shared import (
    BG, C_AMBER, C_INDIGO, C_NAVY, C_PINK, C_PURPLE, GREEN, GRID_COLOR,
    MIN_MINUTES_FOR_RANKING, MODEL_DIR, PANEL_BG, RED, TEXT_MAIN, TEXT_SUB,
    VIS_DIR, add_logo, load_match, load_season, load_shots_faced_by_keeper,
    safe_name, style_axes,
)

N_BOOTSTRAP = 1000
N_LUCK_SIM = 10000
RNG_SEED = 42
SIG_LEVEL = 0.10
MIN_MATCHES_FOR_REGRESSION = 5


# =====================================================================
# 1. Regression trend analysis
# =====================================================================

def compute_trends(match_df: pd.DataFrame, season: pd.DataFrame) -> pd.DataFrame:
    m = match_df[match_df["player"].isin(season["player"])].copy()
    m["date"] = pd.to_datetime(m["date"])
    m["gpae_p90"] = (m["gpae"] / m["minutes"].replace(0, np.nan) * 90.0).fillna(0.0)

    rows = []
    for player, g in m.groupby("player"):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n < MIN_MATCHES_FOR_REGRESSION:
            continue
        x = np.arange(1, n + 1)
        y = g["gpae_p90"].values
        result = stats.linregress(x, y)
        significant = result.pvalue < SIG_LEVEL
        if significant:
            trend = "improving" if result.slope > 0 else "declining"
        else:
            trend = "no clear trend"
        rows.append({
            "player": player, "team": g["team"].iloc[0], "n_matches": n,
            "slope_per_match": result.slope, "slope_per_10_matches": result.slope * 10,
            "intercept": result.intercept, "r_squared": result.rvalue ** 2,
            "p_value": result.pvalue, "std_err": result.stderr,
            "significant": significant, "trend": trend,
        })
    return pd.DataFrame(rows)


def save_trend_leaderboard(trends: pd.DataFrame, out_path: pathlib.Path) -> None:
    ranked = trends.sort_values("slope_per_10_matches", ascending=True)
    colors = np.where(
        ~ranked["significant"], TEXT_SUB,
        np.where(ranked["slope_per_10_matches"] > 0, GREEN, RED),
    )

    fig, ax = plt.subplots(figsize=(12, 9), facecolor=BG)
    style_axes(ax)
    y = np.arange(len(ranked))
    ax.barh(y, ranked["slope_per_10_matches"], color=colors, alpha=0.9)
    ax.axvline(0, color=TEXT_MAIN, lw=1.0, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.player} (n={r.n_matches})" for r in ranked.itertuples()], color=TEXT_MAIN, fontsize=9)
    ax.set_xlabel("Change in GPAE per 90, per 10 matches (regression slope x10)", color=TEXT_SUB, fontsize=10.5)
    fig.suptitle("Shot-Stopping Trend: Regression Slope Over the Season", color=TEXT_MAIN, fontsize=16, fontweight="bold", y=1.02)
    fig.text(0.5, 0.975, f"Green/red = statistically significant trend (p < {SIG_LEVEL:.2f}); gray = not distinguishable from a flat line",
              ha="center", color=TEXT_SUB, fontsize=10)

    fig.text(0.01, 0.01, f"Linear regression of match-level GPAE per 90 vs. matchday index, min {MIN_MATCHES_FOR_REGRESSION} matches | Lamberts Goalkeeper Model",
              fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_trend_detail(match_df: pd.DataFrame, trends: pd.DataFrame, out_path: pathlib.Path, n_panels: int = 6) -> None:
    sig = trends[trends["significant"]].copy()
    sig["abs_slope"] = sig["slope_per_10_matches"].abs()
    picks = sig.nlargest(n_panels, "abs_slope")
    if picks.empty:
        picks = trends.nlargest(n_panels, "r_squared")

    m = match_df[match_df["player"].isin(picks["player"])].copy()
    m["date"] = pd.to_datetime(m["date"])
    m["gpae_p90"] = (m["gpae"] / m["minutes"].replace(0, np.nan) * 90.0).fillna(0.0)

    n_cols = 3
    n_rows = -(-len(picks) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.3 * n_rows), facecolor=BG)
    axes_flat = np.atleast_1d(axes).flatten()
    for ax in axes_flat:
        ax.axis("off")

    for ax, (_, row) in zip(axes_flat, picks.iterrows()):
        ax.axis("on")
        style_axes(ax)
        g = m[m["player"] == row["player"]].sort_values("date").reset_index(drop=True)
        n = len(g)
        x = np.arange(1, n + 1)
        y = g["gpae_p90"].values
        color = GREEN if row["slope_per_10_matches"] > 0 else RED

        ax.scatter(x, y, color=color, s=45, alpha=0.85, zorder=3)
        xfit = np.linspace(1, n, 50)
        yfit = row["intercept"] + row["slope_per_match"] * xfit
        ax.plot(xfit, yfit, color=color, lw=2, alpha=0.9, zorder=2)

        residual_std = np.sqrt(np.sum((y - (row["intercept"] + row["slope_per_match"] * x)) ** 2) / max(n - 2, 1))
        xbar = x.mean()
        se_fit = residual_std * np.sqrt(1 / n + (xfit - xbar) ** 2 / max(np.sum((x - xbar) ** 2), 1e-9))
        tval = stats.t.ppf(0.975, max(n - 2, 1))
        ax.fill_between(xfit, yfit - tval * se_fit, yfit + tval * se_fit, color=color, alpha=0.15, zorder=1)

        ax.axhline(0, color=TEXT_SUB, lw=1.0, ls="--", alpha=0.5)
        ax.set_title(f"{row['player']} ({row['team']})", color=TEXT_MAIN, fontsize=10.5, fontweight="bold", pad=6)
        ax.text(0.03, 0.05, f"slope={row['slope_per_10_matches']:+.2f}/10mtc, R²={row['r_squared']:.2f}, p={row['p_value']:.3f}",
                transform=ax.transAxes, fontsize=8, color=TEXT_SUB, va="bottom")
        ax.set_xlabel("Matchday", color=TEXT_SUB, fontsize=9)
        ax.set_ylabel("GPAE / 90", color=TEXT_SUB, fontsize=9)

    fig.suptitle("Statistically Significant Trends, With 95% Confidence Bands", color=TEXT_MAIN, fontsize=16, fontweight="bold", y=1.01)
    fig.text(0.01, -0.01, "Lamberts Goalkeeper Model | Shaded band = 95% confidence interval on the regression line itself", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 2. Monte Carlo shot-luck simulation
# =====================================================================

def run_shot_luck_simulation(faced: pd.DataFrame, season: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for player, g in faced.groupby("keeper"):
        on_target = g[g["is_on_target"] == 1]
        p = on_target["psxg"].clip(0, 1).values
        actual_goals = int(on_target["is_goal"].sum())
        if len(p) == 0:
            continue
        draws = rng.random((N_LUCK_SIM, len(p))) < p
        sim_goals = draws.sum(axis=1)
        mean_sim, std_sim = sim_goals.mean(), sim_goals.std()
        pctile_actual = float((sim_goals >= actual_goals).mean()) * 100  # % of sims that conceded >= actual (low = overperformed)
        z = (mean_sim - actual_goals) / std_sim if std_sim > 0 else 0.0
        rows.append({
            "player": player, "shots_on_target": len(p), "psxg_sum": p.sum(),
            "actual_goals": actual_goals, "sim_mean_goals": mean_sim, "sim_std_goals": std_sim,
            "sim_p05_goals": np.percentile(sim_goals, 5), "sim_p95_goals": np.percentile(sim_goals, 95),
            "pctile_of_actual": pctile_actual, "luck_z_score": z,
        })
    df = pd.DataFrame(rows)
    return df.merge(season[["player", "team", "goalkeeper_value_index"]], on="player", how="left")


def save_shot_luck_chart(luck: pd.DataFrame, out_path: pathlib.Path) -> None:
    ranked = luck.sort_values("luck_z_score", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 9), facecolor=BG)
    style_axes(ax)
    y = np.arange(len(ranked))

    lower_err = ranked["sim_mean_goals"] - ranked["sim_p05_goals"]
    upper_err = ranked["sim_p95_goals"] - ranked["sim_mean_goals"]
    ax.errorbar(ranked["sim_mean_goals"], y, xerr=[lower_err, upper_err], fmt="none",
                ecolor=GRID_COLOR, elinewidth=6, alpha=0.5, capsize=0, zorder=1)
    ax.scatter(ranked["sim_mean_goals"], y, color=TEXT_SUB, s=40, zorder=2, label="Simulated mean (90% band)")
    colors = np.where(ranked["luck_z_score"] >= 1.0, GREEN, np.where(ranked["luck_z_score"] <= -1.0, RED, C_AMBER))
    ax.scatter(ranked["actual_goals"], y, color=colors, s=110, zorder=3, edgecolors=BG, linewidths=1, label="Actual goals conceded")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.player} (z={r.luck_z_score:+.1f})" for r in ranked.itertuples()], color=TEXT_MAIN, fontsize=9)
    ax.set_xlabel("Goals conceded (on-target shots faced)", color=TEXT_SUB, fontsize=10.5)
    fig.suptitle("Monte Carlo Shot-Luck: Actual vs. 10,000 Simulated Seasons", color=TEXT_MAIN, fontsize=16, fontweight="bold", y=1.02)
    fig.text(0.5, 0.975, "Each shot faced simulated as a coin flip at its own PSxG; green = actual beats the simulated 90% band, red = underperforms it",
              ha="center", color=TEXT_SUB, fontsize=9.5)
    ax.legend(loc="lower right", frameon=False, labelcolor=TEXT_MAIN, fontsize=9)

    fig.text(0.01, 0.01, f"n={N_LUCK_SIM:,} simulations per keeper | Lamberts Goalkeeper Model", fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 3. Monte Carlo bootstrap ranking simulation
# =====================================================================

def run_bootstrap_ranking(match_df: pd.DataFrame, season: pd.DataFrame) -> pd.DataFrame:
    ranked_players = set(season["player"])
    mdf = match_df[match_df["player"].isin(ranked_players)].reset_index(drop=True)
    idx_by_player = mdf.groupby("player").indices
    rng = np.random.default_rng(RNG_SEED)

    rank_records = {p: [] for p in idx_by_player}
    for _ in range(N_BOOTSTRAP):
        sampled_idx = np.concatenate([rng.choice(idxs, size=len(idxs), replace=True) for idxs in idx_by_player.values()])
        resampled = mdf.iloc[sampled_idx]
        season_raw = _build_season_table(resampled)
        scored = _score_submodels(season_raw, min_minutes=0)
        scored = scored.sort_values("goalkeeper_value_index", ascending=False).reset_index(drop=True)
        scored["draw_rank"] = np.arange(1, len(scored) + 1)
        for _, row in scored.iterrows():
            if row["player"] in rank_records:
                rank_records[row["player"]].append(row["draw_rank"])

    rows = []
    actual_rank = season.sort_values("goalkeeper_value_index", ascending=False).reset_index(drop=True)
    actual_rank["actual_rank"] = np.arange(1, len(actual_rank) + 1)
    actual_rank_map = dict(zip(actual_rank["player"], actual_rank["actual_rank"]))

    for player, ranks in rank_records.items():
        ranks = np.array(ranks)
        rows.append({
            "player": player, "actual_rank": actual_rank_map.get(player),
            "median_rank": np.median(ranks), "rank_p05": np.percentile(ranks, 5),
            "rank_p95": np.percentile(ranks, 95), "prob_rank1": float((ranks == 1).mean()),
            "prob_top3": float((ranks <= 3).mean()), "n_draws": len(ranks),
        })
    return pd.DataFrame(rows).merge(season[["player", "team"]], on="player", how="left")


def save_bootstrap_chart(boot: pd.DataFrame, out_path: pathlib.Path) -> None:
    ranked = boot.sort_values("actual_rank")
    fig, ax = plt.subplots(figsize=(12, 9), facecolor=BG)
    style_axes(ax)
    y = np.arange(len(ranked))[::-1]

    lower = ranked["median_rank"] - ranked["rank_p05"]
    upper = ranked["rank_p95"] - ranked["median_rank"]
    ax.errorbar(ranked["median_rank"], y, xerr=[lower, upper], fmt="none",
                ecolor=GRID_COLOR, elinewidth=4, capsize=3, alpha=0.9, zorder=2)
    ax.scatter(ranked["actual_rank"], y, color=C_AMBER, s=90, zorder=3, marker="D", label="Actual observed rank")
    # Drawn last, on top and smaller, so the median marker stays visible as
    # a dot inside the diamond even on rows where median == actual exactly.
    ax.scatter(ranked["median_rank"], y, color=C_NAVY, s=28, zorder=4, edgecolors=BG, linewidths=0.8, label="Median rank (90% CI)")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.player}" for r in ranked.itertuples()], color=TEXT_MAIN, fontsize=9)
    ax.set_xlabel(f"Rank (1 = best) across {N_BOOTSTRAP:,} resampled seasons", color=TEXT_SUB, fontsize=10.5)
    ax.invert_xaxis()
    fig.suptitle("How Stable Is Each Keeper's Rank?", color=TEXT_MAIN, fontsize=16, fontweight="bold", y=1.02)
    fig.text(0.5, 0.975, "Bootstrap: each keeper's own matches resampled with replacement, full model rescored, 1,000 times",
              ha="center", color=TEXT_SUB, fontsize=9.7)
    ax.legend(loc="lower left", frameon=False, labelcolor=TEXT_MAIN, fontsize=9)

    fig.text(0.01, 0.01, "Wide bars = rank depends heavily on which matches went well; narrow bars = robust ranking | Lamberts Goalkeeper Model",
              fontsize=8, color=TEXT_SUB)
    add_logo(fig)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    season = load_season()
    match_df = load_match()
    faced = load_shots_faced_by_keeper()
    faced_ranked = faced[faced["keeper"].isin(season["player"])]

    print("Computing regression trends...")
    trends = compute_trends(match_df, season)
    trends.to_csv(MODEL_DIR / "goalkeeper_trend_regression.csv", index=False)
    save_trend_leaderboard(trends, VIS_DIR / "regression_trend_leaderboard.png")
    save_trend_detail(match_df, trends, VIS_DIR / "regression_trend_detail.png")
    print(f"  {trends['significant'].sum()} / {len(trends)} keepers show a statistically significant trend (p<{SIG_LEVEL})")

    print("Running Monte Carlo shot-luck simulation...")
    luck = run_shot_luck_simulation(faced_ranked, season)
    luck.to_csv(MODEL_DIR / "goalkeeper_montecarlo_shot_luck.csv", index=False)
    save_shot_luck_chart(luck, VIS_DIR / "montecarlo_shot_luck.png")

    print(f"Running Monte Carlo bootstrap ranking ({N_BOOTSTRAP} draws)...")
    boot = run_bootstrap_ranking(match_df, season)
    boot.to_csv(MODEL_DIR / "goalkeeper_montecarlo_ranking.csv", index=False)
    save_bootstrap_chart(boot, VIS_DIR / "montecarlo_rank_stability.png")

    for name in ["regression_trend_leaderboard", "regression_trend_detail", "montecarlo_shot_luck", "montecarlo_rank_stability"]:
        print("Saved:", VIS_DIR / f"{name}.png")


if __name__ == "__main__":
    main()
