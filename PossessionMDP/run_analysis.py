"""
CLI entry point: fits the possession-clock MDPs to Ecuadorian Liga Pro event
data and runs the on-policy vs altered-policy ("directness") comparison
described in README.md Sections 6-7.

Usage:
  python3 run_analysis.py --team "Independiente del Valle"
  python3 run_analysis.py --team "Barcelona SC" --limit 40 --n-worlds 20
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

import mdp_model as mm
from hierarchy import (
    aggregate_counts,
    hierarchical_policy,
    hierarchical_transition_alpha,
    lineup_exposure_weights,
    team_average_alpha,
)
from simulator import apply_directness_policy, simulate_policy, start_zone_distribution, summarize

BG = "#0d1117"
C_NAVY = "#2f8fd1"
C_CORAL = "#ff8a75"
INK = "#e6edf3"
MUTED = "#8b949e"


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Fit and simulate football possession MDPs.")
    p.add_argument("--data-dir", type=pathlib.Path, default=here.parent / "Event")
    p.add_argument("--out-dir", type=pathlib.Path, default=here)
    p.add_argument("--team", type=str, default=None, help="Team display name (substring match).")
    p.add_argument("--limit", type=int, default=None, help="Only load the first N match files.")
    p.add_argument("--n-worlds", type=int, default=25, help="Posterior draws to simulate.")
    p.add_argument("--episodes-per-world", type=int, default=300)
    p.add_argument("--directness-shift", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def pick_team(id_to_name: dict, episodes: list[mm.Episode], wanted: str | None) -> str:
    if wanted:
        matches = [cid for cid, name in id_to_name.items() if wanted.lower() in name.lower()]
        if matches:
            return matches[0]
        print(f"No team name matched '{wanted}'; falling back to most-active team.", file=sys.stderr)
    counts: dict[str, int] = {}
    for ep in episodes:
        counts[ep.team] = counts.get(ep.team, 0) + 1
    return max(counts, key=counts.get)


def print_summary(label: str, stats: dict) -> None:
    print(f"\n{label}")
    for key in ("goal_rate", "shot_rate", "turnover_rate", "mean_steps"):
        mean, lo, hi = summarize(stats[key])
        unit = "" if key == "mean_steps" else "%"
        scale = 1.0 if key == "mean_steps" else 100.0
        print(f"  {key:16s} {mean*scale:6.2f}{unit}   (95% CI {lo*scale:.2f}{unit} - {hi*scale:.2f}{unit})")


def plot_comparison(on_policy: dict, altered: dict, team_name: str, out_path: pathlib.Path) -> None:
    # Small multiples: goal/shot/turnover rate live on very different scales
    # (turnover ~90%, goal ~0.3%), so a shared axis would bury the two
    # metrics that matter most. Each panel gets its own scale instead.
    metrics = [("goal_rate", "Goal rate"), ("shot_rate", "Shot rate"), ("turnover_rate", "Turnover rate")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.5), facecolor=BG)

    bar_w = 0.5
    x = np.array([0.0, 1.0])
    policies = [(on_policy, C_NAVY, "On-policy"), (altered, C_CORAL, "Altered (directness)")]

    for ax, (key, title) in zip(axes, metrics):
        ax.set_facecolor(BG)
        means, los, his = [], [], []
        for stats, _, _ in policies:
            m, lo, hi = summarize(stats[key])
            means.append(m * 100)
            los.append((m - lo) * 100)
            his.append((hi - m) * 100)
        colors = [c for _, c, _ in policies]
        ax.bar(x, means, width=bar_w, color=colors, zorder=3)
        ax.errorbar(x, means, yerr=[los, his], fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
        top = max(m + h for m, h in zip(means, his))
        pad = top * 0.05 + 0.05
        for xi, m, h in zip(x, means, his):
            ax.text(xi, m + h + pad, f"{m:.2f}%", ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_ylim(0, top * 1.2 + pad)
        ax.set_xticks(x)
        ax.set_xticklabels(["On-policy", "Altered"], color=MUTED, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11)
        ax.tick_params(colors=MUTED, left=True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)

    fig.suptitle(f"{team_name}: on-policy vs altered-policy possession simulation", color=INK, fontsize=13)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in policies]
    labels = [lbl for _, _, lbl in policies]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out_path, dpi=160, facecolor=BG)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading matches from {args.data_dir} ...")
    episodes, id_to_name = mm.load_all_episodes(args.data_dir, limit=args.limit)
    print(f"  {len(episodes)} possession episodes extracted.")

    cid = pick_team(id_to_name, episodes, args.team)
    team_name = id_to_name.get(cid, cid)
    print(f"Team: {team_name}")

    transition_counts, policy_counts = aggregate_counts(episodes)
    n_lineups = len({lineup for (c, lineup) in transition_counts if c == cid})
    print(f"  {n_lineups} distinct lineups observed for {team_name}.")

    _, team_alpha, lineup_alpha = hierarchical_transition_alpha(transition_counts)
    team_policy = hierarchical_policy(policy_counts)
    weights = lineup_exposure_weights(transition_counts)
    team_avg_alpha = team_average_alpha(transition_counts, lineup_alpha, team_alpha, weights)

    start_dist = start_zone_distribution(episodes, cid)
    if not start_dist:
        print(f"No possessions found for team id {cid}; nothing to simulate.", file=sys.stderr)
        sys.exit(1)

    on_policy_stats = simulate_policy(
        cid, team_avg_alpha, team_policy, start_dist, rng,
        n_worlds=args.n_worlds, episodes_per_world=args.episodes_per_world,
    )
    altered_policy = apply_directness_policy(team_policy, cid, shift=args.directness_shift)
    altered_stats = simulate_policy(
        cid, team_avg_alpha, altered_policy, start_dist, rng,
        n_worlds=args.n_worlds, episodes_per_world=args.episodes_per_world,
    )

    print_summary("On-policy simulation", on_policy_stats)
    print_summary(f"Altered policy simulation (+{args.directness_shift:.0%} directness)", altered_stats)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / f"policy_comparison_{team_name.replace(' ', '_')}.png"
    plot_comparison(on_policy_stats, altered_stats, team_name, out_png)
    print(f"\nSaved comparison chart to {out_png}")


if __name__ == "__main__":
    main()
