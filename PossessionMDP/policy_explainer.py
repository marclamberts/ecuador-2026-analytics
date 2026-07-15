"""
The on-policy vs altered-policy split (README.md Section 5, "Policy vs.
dynamics") made concrete: for one real, well-populated state, shows the
fitted model's own on-policy and altered-policy action-choice
probabilities side by side. The conceptual (state -> policy -> action ->
dynamics -> outcome) schematic itself is published separately as an HTML
explainer, since a boxes-and-arrows diagram renders far more cleanly there
than as hand-placed matplotlib patches.

Usage: python3 policy_explainer.py [--team NAME] [--out PATH]
"""
from __future__ import annotations

import argparse
import pathlib
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

import mdp_model as mm
from hierarchy import aggregate_counts, hierarchical_policy
from run_analysis import BG, C_CORAL, C_NAVY, INK, MUTED
from simulator import apply_directness_policy


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="On-policy vs altered-policy worked example.")
    p.add_argument("--data-dir", type=pathlib.Path, default=here.parent / "Event")
    p.add_argument("--team", type=str, default="Independiente del Valle")
    p.add_argument("--out", type=pathlib.Path, default=here / "policy_explainer.png")
    p.add_argument("--directness-shift", type=float, default=0.15)
    return p.parse_args()


def pick_example_state(transition_counts, policy, altered_policy, cid: str):
    """Highest-volume wide, attacking-third state, so the example is both
    visually clean (crosses are a live option there) and well-supported."""
    totals: Counter = Counter()
    for (c, lineup), by_s in transition_counts.items():
        if c != cid:
            continue
        for s, by_a in by_s.items():
            zone = s // mm.N_CLOCK
            col, row = zone // mm.N_ROWS, zone % mm.N_ROWS
            if col >= 4 and row in (0, mm.N_ROWS - 1):
                totals[s] += sum(w for ctr in by_a.values() for w in ctr.values())
    for s, _ in totals.most_common():
        if (cid, s) in policy and (cid, s) in altered_policy:
            return s
    return None


def describe_state(s: int) -> str:
    zone, clk = s // mm.N_CLOCK, s % mm.N_CLOCK
    col, row = zone // mm.N_ROWS, zone % mm.N_ROWS
    third = ["defensive", "defensive-mid", "attacking-mid", "attacking"][min(col * 4 // mm.N_COLS, 3)]
    channel = "wide" if row in (0, mm.N_ROWS - 1) else "central"
    from league_visuals import CLOCK_LABELS
    return f"{third} third, {channel} channel, {CLOCK_LABELS[clk]} into the possession"


def plot_example(on_dist: dict, alt_dist: dict, state_desc: str, team_name: str, out_path: pathlib.Path) -> None:
    actions = mm.ACTIONS
    on_vals = np.array([on_dist.get(a, 0.0) for a in actions]) * 100
    alt_vals = np.array([alt_dist.get(a, 0.0) for a in actions]) * 100

    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=BG)
    ax.set_facecolor(BG)

    x = np.arange(len(actions))
    w = 0.36
    ax.bar(x - w / 2, on_vals, width=w, color=C_NAVY, label="On-policy", zorder=3)
    ax.bar(x + w / 2, alt_vals, width=w, color=C_CORAL, label="Altered (+15% directness)", zorder=3)
    for xi, v in zip(x - w / 2, on_vals):
        ax.text(xi, v + 1.3, f"{v:.0f}%", ha="center", color=INK, fontsize=9)
    for xi, v in zip(x + w / 2, alt_vals):
        ax.text(xi, v + 1.3, f"{v:.0f}%", ha="center", color=INK, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in actions], color=INK)
    ax.set_ylabel("P(action chosen)", color=MUTED)
    ax.set_ylim(0, max(on_vals.max(), alt_vals.max()) * 1.3)
    ax.tick_params(colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.set_title(f"{team_name}: on-policy vs altered-policy action mix\n{state_desc}", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK, fontsize=9)
    fig.text(0.5, 0.02, "Same fitted dynamics both ways — only the mix of attempted actions moves\n"
                        "(mass shifts from sideways/back toward advance/cross).",
              color=MUTED, fontsize=8.8, ha="center", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=160, facecolor=BG)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    episodes, id_to_name = mm.load_all_episodes(args.data_dir)
    name_to_cid = {v: k for k, v in id_to_name.items()}
    cid = name_to_cid.get(args.team)
    if cid is None:
        raise SystemExit(f"Team '{args.team}' not found. Known teams: {sorted(name_to_cid)}")

    transition_counts, policy_counts = aggregate_counts(episodes)
    team_policy = hierarchical_policy(policy_counts)
    altered_policy = apply_directness_policy(team_policy, cid, shift=args.directness_shift)

    s = pick_example_state(transition_counts, team_policy, altered_policy, cid)
    if s is None:
        raise SystemExit("No well-populated example state found for this team.")

    plot_example(team_policy[(cid, s)], altered_policy[(cid, s)], describe_state(s), args.team, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
