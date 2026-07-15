"""
The possession MDP's value function, made visible: V(s) is the probability
that continuing to play out a possession from state s = (zone, possession
clock), under the team's own fitted policy and dynamics, eventually ends in
a goal. This is the quantity a Bellman equation solves for, and it is the
one piece of "what is an MDP" that every other chart in this folder has
used implicitly (it's exactly what the simulator estimates by rollout)
without ever drawing directly.

Solved by value iteration on the fitted team-average MDP:

    V(s) = sum_a pi(a|s) * sum_{s'} P(s'|s,a) * r(s')

where r(GOAL) = 1, r(SHOT_NOGOAL) = r(TURNOVER) = 0, and r(s') = V(s') for
any non-terminal next state -- i.e. exactly the xT-style value-iteration
recursion already used for the (separate, direct-from-events) xT grid
elsewhere in this repo, but computed here from the possession-MDP's own
fitted policy and Dirichlet-posterior-mean dynamics instead of raw pass
completions, and faceted by possession-clock bucket rather than collapsed
across it.

Usage: python3 value_function.py [--team NAME] [--out PATH]
"""
from __future__ import annotations

import argparse
import pathlib

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
from league_visuals import CLOCK_LABELS
from run_analysis import BG, INK, MUTED

C_AMBER = "#ffc247"


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Solve and plot the possession-MDP value function.")
    p.add_argument("--data-dir", type=pathlib.Path, default=here.parent / "Event")
    p.add_argument("--team", type=str, default="Independiente del Valle")
    p.add_argument("--out", type=pathlib.Path, default=here / "value_function.png")
    p.add_argument("--iterations", type=int, default=150)
    return p.parse_args()


def solve_value_function(cid: str, team_policy: dict, team_avg_alpha: dict, iterations: int) -> np.ndarray:
    V = np.zeros(mm.N_STATES)
    for _ in range(iterations):
        V_new = np.zeros(mm.N_STATES)
        for s in range(mm.N_STATES):
            pol = team_policy.get((cid, s))
            if not pol:
                continue
            total = 0.0
            for a, pa in pol.items():
                alpha = team_avg_alpha.get((cid, s, a))
                if not alpha:
                    continue
                denom = sum(alpha.values())
                if denom <= 0:
                    continue
                for outcome, w in alpha.items():
                    p = w / denom
                    if outcome == mm.GOAL:
                        total += pa * p * 1.0
                    elif outcome in (mm.TURNOVER, mm.NOGOAL_SHOT):
                        continue  # r = 0
                    else:
                        total += pa * p * V[outcome]
            V_new[s] = total
        V = V_new
    return V


def main() -> None:
    args = parse_args()
    episodes, id_to_name = mm.load_all_episodes(args.data_dir)
    name_to_cid = {v: k for k, v in id_to_name.items()}
    cid = name_to_cid.get(args.team)
    if cid is None:
        raise SystemExit(f"Team '{args.team}' not found. Known teams: {sorted(name_to_cid)}")

    transition_counts, policy_counts = aggregate_counts(episodes)
    _, team_alpha, lineup_alpha = hierarchical_transition_alpha(transition_counts)
    team_policy = hierarchical_policy(policy_counts)
    weights = lineup_exposure_weights(transition_counts)
    team_avg_alpha = team_average_alpha(transition_counts, lineup_alpha, team_alpha, weights)

    print(f"Solving value function for {args.team} ({args.iterations} sweeps) ...")
    V = solve_value_function(cid, team_policy, team_avg_alpha, args.iterations)

    grids = []
    for clk in range(mm.N_CLOCK):
        grid = np.zeros((mm.N_COLS, mm.N_ROWS))
        for zone in range(mm.N_ZONES):
            col, row = zone // mm.N_ROWS, zone % mm.N_ROWS
            grid[col, row] = V[mm.state_idx(zone, clk)]
        grids.append(grid)

    vmax = max(g.max() for g in grids)
    cmap = LinearSegmentedColormap.from_list("value", [BG, "#1e3a5f", "#7b3fa0", "#c1447e", "#ff8a3d", C_AMBER])

    fig, axes = plt.subplots(1, mm.N_CLOCK, figsize=(4.2 * mm.N_CLOCK, 6.6), facecolor=BG)
    for ax, grid, label in zip(axes, grids, CLOCK_LABELS):
        pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color="#3a4658", linewidth=1.0, half=False)
        pitch.draw(ax=ax)
        bins = pitch.bin_statistic(np.array([50.0]), np.array([50.0]), statistic="count",
                                    bins=(mm.N_COLS, mm.N_ROWS))
        bins["statistic"] = grid.T
        pitch.heatmap(bins, ax=ax, cmap=cmap, vmin=0, vmax=vmax, alpha=0.92, zorder=0.5,
                      edgecolors=BG, linewidth=1.0)
        # label_heatmap (not plain ax.text) is required: it routes through
        # pitch.text(), which applies mplsoccer's vertical-pitch coordinate
        # flip. Plain ax.text(cx, cy, ...) places labels using the raw,
        # unflipped bin centers, so the printed numbers land on the wrong
        # cells even though the heatmap colors underneath are correct.
        bins["statistic"] = bins["statistic"] * 100
        pitch.label_heatmap(bins, ax=ax, str_format="{:.1f}", color=INK, fontsize=8, zorder=3,
                             ha="center", va="center")
        ax.set_title(label, color=INK, fontsize=12.5, pad=8)

    fig.suptitle(f"{args.team}: possession value function V(s)", color=INK, fontsize=16, y=0.99)
    fig.text(0.5, 0.945, "P(this possession eventually ends in a goal), by pitch zone and possession clock "
                          "-- solved by value iteration on the fitted MDP",
              color=MUTED, fontsize=10.5, ha="center")
    fig.text(0.5, 0.02, "Pitch runs bottom (own goal) to top (opponent goal). Numbers are V(s) x 100 "
                        "for readability, e.g. \"2.1\" = a 2.1% chance this possession ends in a goal.",
              color=MUTED, fontsize=9, ha="center")

    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(args.out, dpi=155, facecolor=BG)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
