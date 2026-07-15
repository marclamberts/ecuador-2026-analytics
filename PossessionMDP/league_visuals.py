"""
League-wide visuals for the possession-clock MDP work (README.md):

  1. possession_clock_hazard.png  - how the outcome of the next action
     changes as the possession clock runs on (the empirical nonstationarity
     the whole framework is built to capture).
  2. zone_heatmaps.png            - shot-attempt rate and goal-conversion
     rate by pitch zone, league-wide.
  3. league_team_comparison.png   - simulated on-policy goal/shot rate per
     possession, ranked across every Liga Pro team.
  4. policy_comparison_<team>.png - the on-policy vs altered-policy chart
     (run_analysis.py) for the most data-rich team, as a worked example.

Usage: python3 league_visuals.py [--out-dir DIR] [--n-worlds N]
"""
from __future__ import annotations

import argparse
import pathlib
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import VerticalPitch

import mdp_model as mm
from hierarchy import (
    aggregate_counts,
    hierarchical_policy,
    hierarchical_transition_alpha,
    lineup_exposure_weights,
    team_average_alpha,
)
from run_analysis import BG, C_CORAL, C_NAVY, INK, MUTED, add_source_line, plot_comparison
from simulator import apply_directness_policy, simulate_policy, start_zone_distribution, summarize

MUTED_FILL = "#2a3341"
C_AMBER = "#ffc247"

CLOCK_LABELS = ["0-5s", "5-10s", "10-15s", "15-25s", "25s+"]


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="League-wide possession-MDP visuals.")
    p.add_argument("--data-dir", type=pathlib.Path, default=here.parent / "Event")
    p.add_argument("--out-dir", type=pathlib.Path, default=here)
    p.add_argument("--n-worlds", type=int, default=50,
                    help="Posterior draws per team; low values make the 2.5/97.5 percentile whiskers noisy.")
    p.add_argument("--episodes-per-world", type=int, default=250)
    p.add_argument("--min-episodes", type=int, default=200, help="Skip teams with fewer possessions than this.")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def possession_clock_hazard(transition_counts) -> np.ndarray:
    """Returns a (N_CLOCK, 4) array of league-wide proportions
    [continue, turnover, shot_nogoal, goal] by possession-clock bucket,
    aggregated over every team, lineup, zone and action."""
    totals = np.zeros((mm.N_CLOCK, 4))
    cols = {mm.TURNOVER: 1, mm.NOGOAL_SHOT: 2, mm.GOAL: 3}
    for (cid, lineup), by_s in transition_counts.items():
        for s, by_a in by_s.items():
            clk = s % mm.N_CLOCK
            for a, ctr in by_a.items():
                for outcome, w in ctr.items():
                    col = cols.get(outcome, 0)  # 0 = continue (destination is another state)
                    totals[clk, col] += w
    row_sums = totals.sum(axis=1, keepdims=True)
    return np.divide(totals, row_sums, out=np.zeros_like(totals), where=row_sums > 0)


def plot_possession_clock_hazard(props: np.ndarray, out_path: pathlib.Path) -> None:
    labels = ["Continue possession", "Turnover", "Shot (no goal)", "Goal"]
    colors = [MUTED_FILL, C_CORAL, C_NAVY, C_AMBER]
    order = [0, 1, 2, 3]  # continue, turnover, shot_nogoal, goal (stack bottom to top)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.set_facecolor(BG)
    x = np.arange(len(CLOCK_LABELS))
    bottom = np.zeros(len(CLOCK_LABELS))
    for idx in order:
        vals = props[:, idx] * 100
        ax.bar(x, vals, bottom=bottom, width=0.62, color=colors[idx], label=labels[idx], zorder=3)
        if idx in (1, 2, 3):  # annotate the eventful slivers, not the majority "continue" segment
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0.4:
                    ax.text(xi, b + v / 2, f"{v:.1f}%", ha="center", va="center", color=INK, fontsize=8)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(CLOCK_LABELS, color=INK)
    ax.set_xlabel("Possession clock (seconds since gaining the ball)", color=MUTED)
    ax.set_ylabel("Share of actions taken from this clock bucket", color=MUTED)
    ax.set_ylim(0, 100)
    ax.tick_params(colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.set_title("Liga Pro: possession outcome by possession-clock bucket", color=INK, fontsize=13)
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], lbls[::-1], frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5), labelcolor=INK)
    fig.tight_layout()
    add_source_line(fig)
    fig.savefig(out_path, dpi=160, facecolor=BG)
    plt.close(fig)


MIN_SHOTS_FOR_CONVERSION = 15.0  # below this, a zone's goal-conversion rate is noise, not signal


def zone_shot_and_goal_rates(transition_counts):
    """Returns (shot_rate, goal_rate, shots) as (N_COLS, N_ROWS) arrays:
    shot_rate = P(a visit to this zone attempts a shot); goal_rate =
    P(goal | shot attempted from this zone), NaN where the zone has too few
    shots for that rate to be meaningful; shots = the (weighted) shot count
    used for that masking."""
    visits = np.zeros(mm.N_ZONES)
    shots = np.zeros(mm.N_ZONES)
    goals = np.zeros(mm.N_ZONES)
    for (cid, lineup), by_s in transition_counts.items():
        for s, by_a in by_s.items():
            zone = s // mm.N_CLOCK
            for a, ctr in by_a.items():
                n = sum(ctr.values())
                visits[zone] += n
                if a == "shot":
                    shots[zone] += n
                    goals[zone] += ctr.get(mm.GOAL, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shot_rate = np.where(visits > 0, shots / visits, 0)
        goal_rate = np.where(shots >= MIN_SHOTS_FOR_CONVERSION, goals / np.maximum(shots, 1), np.nan)
    shape = (mm.N_COLS, mm.N_ROWS)
    return shot_rate.reshape(shape), goal_rate.reshape(shape), shots.reshape(shape)


def plot_zone_heatmaps(shot_rate, goal_rate, shots, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.5), facecolor=BG)
    cmap = LinearSegmentedColormap.from_list("mdp", [BG, "#1e3a5f", "#7b3fa0", "#c1447e", "#ff8a3d", C_AMBER])
    cmap.set_bad(color=MUTED_FILL)

    titles = ["Shot-attempt rate by zone", "Goal conversion rate by zone (given a shot)"]
    grids = [shot_rate, goal_rate]
    for ax, grid, title in zip(axes, grids, titles):
        pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color="#3a4658", linewidth=1.0, half=False)
        pitch.draw(ax=ax)
        bins = pitch.bin_statistic(np.array([50.0]), np.array([50.0]), statistic="count",
                                    bins=(mm.N_COLS, mm.N_ROWS))
        bins["statistic"] = grid.T * 100
        pitch.heatmap(bins, ax=ax, cmap=cmap, alpha=0.9, zorder=0.5, edgecolors=BG, linewidth=1.0)

        # label_heatmap (not plain ax.text) is required here: it routes
        # through pitch.text(), which applies mplsoccer's vertical-pitch
        # coordinate flip. Two passes so masked (too-few-shots) cells read
        # "n/a" instead of a spurious rate; "n/a".format(x) just returns
        # "n/a" since the format string has no placeholder.
        mask = np.isnan(bins["statistic"])
        numeric_bins = dict(bins, statistic=np.where(mask, np.nan, bins["statistic"]))
        pitch.label_heatmap(numeric_bins, ax=ax, str_format="{:.1f}%", exclude_nan=True,
                             color=INK, fontsize=8, zorder=3, ha="center", va="center")
        na_bins = dict(bins, statistic=np.where(mask, 1.0, np.nan))
        pitch.label_heatmap(na_bins, ax=ax, str_format="n/a", exclude_nan=True,
                             color=MUTED, fontsize=8, zorder=3, ha="center", va="center")
        ax.set_title(title, color=INK, fontsize=11, pad=10)

    fig.suptitle("Liga Pro: possession-MDP zone tendencies (whole league, attacking direction normalized)",
                  color=INK, fontsize=13)
    fig.text(0.5, 0.03,
              "Pitch runs bottom (own goal) to top (opponent goal); grid = the MDP's zone discretization. "
              f"\"n/a\" = fewer than {int(MIN_SHOTS_FOR_CONVERSION)} shots recorded from that zone.",
              color=MUTED, fontsize=9, ha="center")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    add_source_line(fig, y=0.006)
    fig.savefig(out_path, dpi=160, facecolor=BG)
    plt.close(fig)


def plot_league_comparison(records: list[dict], exemplar_name: str, out_path: pathlib.Path) -> None:
    records = sorted(records, key=lambda r: r["goal_mean"])
    names = [r["name"] for r in records]
    means = np.array([r["goal_mean"] for r in records]) * 100
    los = np.array([r["goal_mean"] - r["goal_lo"] for r in records]) * 100
    his = np.array([r["goal_hi"] - r["goal_mean"] for r in records]) * 100
    colors = [C_CORAL if r["name"] == exemplar_name else C_NAVY for r in records]

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(records) + 1.5), facecolor=BG)
    ax.set_facecolor(BG)
    y = np.arange(len(records))
    ax.barh(y, means, xerr=[los, his], color=colors, height=0.62,
            error_kw=dict(ecolor=INK, elinewidth=1, capsize=3), zorder=3)
    xmax = max(m + h for m, h in zip(means, his))
    pad = xmax * 0.02
    for yi, m, h in zip(y, means, his):
        ax.text(m + h + pad, yi, f"{m:.2f}%", va="center", color=INK, fontsize=8.5)

    ax.set_xlim(0, xmax * 1.14)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK, fontsize=9)
    ax.set_xlabel("Simulated on-policy goal rate per possession", color=MUTED)
    ax.tick_params(colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.set_title("Liga Pro on-policy possession simulation, ranked by goal rate", color=INK, fontsize=12)
    fig.text(0.01, 0.005, f"Highlighted: {exemplar_name} (Figure 4 worked example)", color=C_CORAL,
              fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    add_source_line(fig)
    fig.savefig(out_path, dpi=160, facecolor=BG)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading matches from {args.data_dir} ...")
    episodes, id_to_name = mm.load_all_episodes(args.data_dir)
    print(f"  {len(episodes)} possession episodes, {len(id_to_name)} teams.")

    transition_counts, policy_counts = aggregate_counts(episodes)
    _, team_alpha, lineup_alpha = hierarchical_transition_alpha(transition_counts)
    team_policy = hierarchical_policy(policy_counts)
    weights = lineup_exposure_weights(transition_counts)
    team_avg_alpha = team_average_alpha(transition_counts, lineup_alpha, team_alpha, weights)

    print("Figure 1/4: possession-clock hazard ...")
    props = possession_clock_hazard(transition_counts)
    plot_possession_clock_hazard(props, args.out_dir / "possession_clock_hazard.png")

    print("Figure 2/4: zone heatmaps ...")
    shot_rate, goal_rate, shots = zone_shot_and_goal_rates(transition_counts)
    plot_zone_heatmaps(shot_rate, goal_rate, shots, args.out_dir / "zone_heatmaps.png")

    print("Figure 3/4: league-wide team comparison (simulating every team) ...")
    episode_counts = Counter(ep.team for ep in episodes)
    records = []
    for cid, name in id_to_name.items():
        if episode_counts.get(cid, 0) < args.min_episodes:
            continue
        start_dist = start_zone_distribution(episodes, cid)
        stats = simulate_policy(cid, team_avg_alpha, team_policy, start_dist, rng,
                                 n_worlds=args.n_worlds, episodes_per_world=args.episodes_per_world)
        g_mean, g_lo, g_hi = summarize(stats["goal_rate"])
        s_mean, s_lo, s_hi = summarize(stats["shot_rate"])
        records.append({"cid": cid, "name": name, "n_episodes": episode_counts[cid],
                         "goal_mean": g_mean, "goal_lo": g_lo, "goal_hi": g_hi,
                         "shot_mean": s_mean})
        print(f"    {name:40s} goal_rate={g_mean*100:5.2f}%  shot_rate={s_mean*100:5.2f}%")

    exemplar = max(records, key=lambda r: r["n_episodes"])
    plot_league_comparison(records, exemplar["name"], args.out_dir / "league_team_comparison.png")

    print(f"Figure 4/4: on-policy vs altered-policy worked example ({exemplar['name']}) ...")
    cid = exemplar["cid"]
    start_dist = start_zone_distribution(episodes, cid)
    on_stats = simulate_policy(cid, team_avg_alpha, team_policy, start_dist, rng,
                                n_worlds=args.n_worlds, episodes_per_world=args.episodes_per_world)
    altered_policy = apply_directness_policy(team_policy, cid)
    alt_stats = simulate_policy(cid, team_avg_alpha, altered_policy, start_dist, rng,
                                 n_worlds=args.n_worlds, episodes_per_world=args.episodes_per_world)
    plot_comparison(on_stats, alt_stats, exemplar["name"],
                     args.out_dir / f"policy_comparison_{exemplar['name'].replace(' ', '_')}.png")

    print(f"\nAll figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
