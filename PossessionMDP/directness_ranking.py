"""
Which Liga Pro teams would gain or lose the most simulated goal rate under
the "directness" altered policy (README.md Section 6)? Runs both the
on-policy and altered-policy simulation for every team, and ranks the
league by the change in simulated on-policy goal rate per possession.

Usage: python3 directness_ranking.py [--out PATH]
"""
from __future__ import annotations

import argparse
import pathlib
from collections import Counter

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
from run_analysis import BG, INK, MUTED
from simulator import apply_directness_policy, mean_ci, simulate_policy_pair, start_zone_distribution, summarize

C_GAIN = "#57c785"
C_LOSS = "#e0555f"


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Rank teams by simulated benefit from a more direct policy.")
    p.add_argument("--data-dir", type=pathlib.Path, default=here.parent / "Event")
    p.add_argument("--out", type=pathlib.Path, default=here / "directness_ranking.png")
    p.add_argument("--n-worlds", type=int, default=40)
    p.add_argument("--episodes-per-world", type=int, default=250)
    p.add_argument("--min-episodes", type=int, default=200)
    p.add_argument("--directness-shift", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading matches from {args.data_dir} ...")
    episodes, id_to_name = mm.load_all_episodes(args.data_dir)
    print(f"  {len(episodes)} possession episodes, {len(id_to_name)} teams.")

    transition_counts, policy_counts = aggregate_counts(episodes)
    _, team_alpha, lineup_alpha = hierarchical_transition_alpha(transition_counts)
    team_policy = hierarchical_policy(policy_counts)
    weights = lineup_exposure_weights(transition_counts)
    team_avg_alpha = team_average_alpha(transition_counts, lineup_alpha, team_alpha, weights)

    episode_counts = Counter(ep.team for ep in episodes)
    rows = []
    for cid, name in id_to_name.items():
        if episode_counts.get(cid, 0) < args.min_episodes:
            continue
        start_dist = start_zone_distribution(episodes, cid)
        altered_policy = apply_directness_policy(team_policy, cid, shift=args.directness_shift)
        paired = simulate_policy_pair(cid, team_avg_alpha, team_policy, altered_policy, start_dist, rng,
                                       n_worlds=args.n_worlds, episodes_per_world=args.episodes_per_world)
        on_mean, _, _ = summarize(paired["a"]["goal_rate"])
        alt_mean, _, _ = summarize(paired["b"]["goal_rate"])
        delta_mean, delta_lo, delta_hi = mean_ci(paired["delta_goal_rate"])
        rows.append({"name": name, "on": on_mean, "alt": alt_mean, "delta": delta_mean,
                      "delta_lo": delta_lo, "delta_hi": delta_hi})
        sign = "+" if delta_mean >= 0 else ""
        print(f"  {name:40s} on={on_mean*100:5.2f}%  altered={alt_mean*100:5.2f}%  "
              f"delta={sign}{delta_mean*100:.2f}pp  (95% CI {delta_lo*100:+.2f} to {delta_hi*100:+.2f}pp)")

    rows.sort(key=lambda r: r["delta"])

    fig, ax = plt.subplots(figsize=(9.5, 0.46 * len(rows) + 2), facecolor=BG)
    ax.set_facecolor(BG)
    y = np.arange(len(rows))
    deltas = np.array([r["delta"] for r in rows]) * 100
    los = np.array([r["delta"] - r["delta_lo"] for r in rows]) * 100
    his = np.array([r["delta_hi"] - r["delta"] for r in rows]) * 100
    colors = [C_GAIN if d >= 0 else C_LOSS for d in deltas]

    ax.barh(y, deltas, color=colors, height=0.62, zorder=3)
    ax.errorbar(deltas, y, xerr=[los, his], fmt="none", ecolor=INK, elinewidth=1, capsize=3,
                alpha=0.8, zorder=4)
    ax.axvline(0, color=MUTED, linewidth=1.2, zorder=2)

    reach = np.maximum(np.abs(deltas - los), np.abs(deltas + his))
    span = reach.max()
    for yi, d, h, l in zip(y, deltas, his, los):
        pad = span * 0.035
        ha = "left" if d >= 0 else "right"
        x = d + h + pad if d >= 0 else d - l - pad
        ax.text(x, yi, f"{d:+.2f}pp", va="center", ha=ha, color=INK, fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], color=INK, fontsize=9)
    ax.set_xlim(-span * 1.3, span * 1.3)
    ax.set_xlabel("Change in simulated on-policy goal rate per possession\n"
                  "(altered \"more direct\" policy minus current on-policy behavior)", color=MUTED)
    ax.tick_params(colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)

    fig.suptitle("Liga Pro: who would gain from playing more directly?", color=INK, fontsize=15, y=0.998)
    fig.text(0.5, 0.965,
              f"Simulated effect of a +{args.directness_shift:.0%} directness shift in the middle and "
              "attacking thirds, dynamics held fixed (README.md §6-7)",
              color=MUTED, fontsize=9.8, ha="center")
    fig.text(0.5, 0.012, "Reflects the immediate, fixed-defense upside of the change, not a steady-state "
                         "prediction once opponents adapt — see README.md §7.",
              color=MUTED, fontsize=8.6, ha="center")

    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(args.out, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
