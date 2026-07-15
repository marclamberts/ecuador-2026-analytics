"""
A ternary comparison of every Liga Pro team's build-up style, fit from the
possession-MDP's action-choice policy (README.md Section 5): for each team,
its revealed mix of forward (advance + cross), sideways, and backward
passing/carrying, aggregated across every zone and possession-clock state it
was actually in, weighted by how often it was in each. Shot attempts are
excluded from the three axes (they are rare enough to collapse to an edge)
and shown instead as marker size, using the same on-policy simulated goal
rate as league_visuals.py's league-wide comparison.

These three axes are a much better fit for a ternary plot than the
{goal, shot, turnover} possession-outcome split: that split is dominated by
turnover (~80-90%) for every team, so every point would crowd into one
corner. Forward/sideways/backward shares are comparably sized, so teams
actually spread across the triangle.

Usage: python3 team_style_ternary.py [--out PATH]
"""
from __future__ import annotations

import argparse
import pathlib
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

import mdp_model as mm
from hierarchy import (
    aggregate_counts,
    hierarchical_policy,
    hierarchical_transition_alpha,
    lineup_exposure_weights,
    team_average_alpha,
)
from run_analysis import BG, INK, MUTED
from simulator import simulate_policy, start_zone_distribution, summarize

MUTED_2 = "#4a5568"
GRID_COLOR = "#2a3341"

V_TOP = np.array([0.5, np.sqrt(3) / 2])   # Forward (advance + cross)
V_RIGHT = np.array([1.0, 0.0])            # Sideways
V_LEFT = np.array([0.0, 0.0])             # Backward

TEAM_COLORS = [
    "#4da3e8", "#ff9179", "#f2c14e", "#9b8ce0", "#57c785", "#e069a6",
    "#5ec8d8", "#e0904a", "#a8d65e", "#c179d1", "#e0555f", "#4fb8a0",
    "#f0a13c", "#7b93e0", "#c9a13c", "#6fd1a0",
]


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Ternary comparison of team build-up style.")
    p.add_argument("--data-dir", type=pathlib.Path, default=here.parent / "Event")
    p.add_argument("--out", type=pathlib.Path, default=here / "team_style_ternary.png")
    p.add_argument("--n-worlds", type=int, default=25)
    p.add_argument("--episodes-per-world", type=int, default=250)
    p.add_argument("--min-episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def bary_to_cart(f: float, s: float, b: float) -> np.ndarray:
    return f * V_TOP + s * V_RIGHT + b * V_LEFT


def team_style(policy_counts, cid: str) -> tuple[float, float, float, float]:
    """Weighted (forward, sideways, backward, shot_share) for one team,
    aggregated across every lineup, zone and clock bucket it was observed
    in. forward/sideways/backward are renormalized to sum to 1 with shot
    excluded, since shot is rare enough to otherwise collapse every team
    toward one edge of the triangle."""
    totals: Counter = Counter()
    for (c, lineup), by_s in policy_counts.items():
        if c != cid:
            continue
        for s, ctr in by_s.items():
            totals.update(ctr)
    shot_share = totals.get("shot", 0.0) / sum(totals.values()) if totals else 0.0
    ex_shot = {a: v for a, v in totals.items() if a != "shot"}
    denom = sum(ex_shot.values())
    forward = (ex_shot.get("advance", 0.0) + ex_shot.get("cross", 0.0)) / denom
    sideways = ex_shot.get("sideways", 0.0) / denom
    backward = ex_shot.get("back", 0.0) / denom
    return forward, sideways, backward, shot_share


def draw_ternary_frame(ax) -> None:
    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.12, np.sqrt(3) / 2 + 0.12)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.plot(*zip(V_TOP, V_RIGHT, V_LEFT, V_TOP), color=MUTED, linewidth=1.4, zorder=2)

    for level in (0.2, 0.4, 0.6, 0.8):
        f_line = (bary_to_cart(level, 0, 1 - level), bary_to_cart(level, 1 - level, 0))
        s_line = (bary_to_cart(1 - level, level, 0), bary_to_cart(0, level, 1 - level))
        b_line = (bary_to_cart(1 - level, 0, level), bary_to_cart(0, 1 - level, level))
        for p0, p1 in (f_line, s_line, b_line):
            ax.plot(*zip(p0, p1), color=GRID_COLOR, linewidth=0.8, zorder=1)

    label_kw = dict(ha="center", va="center", color=INK, fontsize=12.5)
    ax.text(V_TOP[0], V_TOP[1] + 0.075, "Forward\n(advance + cross)", **label_kw)
    ax.text(V_RIGHT[0] + 0.075, V_RIGHT[1] - 0.02, "Sideways", **label_kw)
    ax.text(V_LEFT[0] - 0.075, V_LEFT[1] - 0.02, "Backward", **label_kw)

    for level in (0.2, 0.4, 0.6, 0.8):
        p = bary_to_cart(1 - level, level, 0) + np.array([0.035, -0.02])
        ax.text(*p, f"{int(level*100)}%", color=MUTED_2, fontsize=7.5, ha="left", va="center")


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
        forward, sideways, backward, shot_share = team_style(policy_counts, cid)
        start_dist = start_zone_distribution(episodes, cid)
        stats = simulate_policy(cid, team_avg_alpha, team_policy, start_dist, rng,
                                 n_worlds=args.n_worlds, episodes_per_world=args.episodes_per_world)
        goal_mean, _, _ = summarize(stats["goal_rate"])
        rows.append({"name": name, "forward": forward, "sideways": sideways,
                      "backward": backward, "shot_share": shot_share, "goal_rate": goal_mean})
        print(f"  {name:40s} fwd={forward*100:5.1f}%  side={sideways*100:5.1f}%  "
              f"back={backward*100:5.1f}%  goal_rate={goal_mean*100:.2f}%")

    goal_rates = np.array([r["goal_rate"] for r in rows])
    g_min, g_max = goal_rates.min(), goal_rates.max()
    sizes = 90 + (goal_rates - g_min) / max(g_max - g_min, 1e-9) * 420

    fig, ax = plt.subplots(figsize=(10.5, 10), facecolor=BG)
    ax.set_facecolor(BG)
    draw_ternary_frame(ax)

    texts = []
    for i, r in enumerate(rows):
        xy = bary_to_cart(r["forward"], r["sideways"], r["backward"])
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        ax.scatter(*xy, s=sizes[i], color=color, edgecolor=BG, linewidth=1.2, zorder=5, alpha=0.92)
        texts.append(ax.text(xy[0], xy[1], r["name"], color=color, fontsize=9.5, zorder=6,
                              fontweight="bold"))

    adjust_text(texts, ax=ax, expand=(1.3, 1.6),
                arrowprops=dict(arrowstyle="-", color=MUTED_2, lw=0.6, alpha=0.7))

    fig.suptitle("Liga Pro: build-up style by team", color=INK, fontsize=18, y=0.985)
    ax.set_title("Forward / sideways / backward share of every non-shot action attempted\n"
                  "(marker size = simulated on-policy goal rate per possession)",
                  color=MUTED, fontsize=10.5, pad=2)

    legend_sizes = [g_min, (g_min + g_max) / 2, g_max]
    legend_handles = [
        plt.scatter([], [], s=90 + (v - g_min) / max(g_max - g_min, 1e-9) * 420,
                    color=MUTED_2, edgecolor=BG, linewidth=1.2, alpha=0.92)
        for v in legend_sizes
    ]
    legend_labels = [f"{v*100:.1f}% goal rate" for v in legend_sizes]
    ax.legend(legend_handles, legend_labels, frameon=False, labelcolor=INK,
              loc="lower right", bbox_to_anchor=(1.05, -0.06), fontsize=8.5, handletextpad=1.2,
              labelspacing=1.4, borderpad=0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
