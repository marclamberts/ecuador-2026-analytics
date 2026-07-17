"""
A conceptual schematic showing how possessions are segmented out of the
raw chronological event stream (paper Section 3 / README.md Data
section): a new episode begins on a change of ball control and ends in
a goal, a non-scoring shot, a failed pass/take-on, a defensive
interception/tackle/clearance, a foul conceded, or the end of a period.

This is a diagram, not a data chart -- it takes no arguments and reads
no event data -- so it lives standalone rather than inside run_analysis.py.

Usage: python3 segmentation_schematic.py [--out PATH]
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from run_analysis import BG, C_CORAL, C_NAVY, INK, MUTED, add_source_line

C_GOAL = "#f2c14e"


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Possession segmentation concept schematic.")
    p.add_argument("--out", type=pathlib.Path, default=here / "segmentation_schematic.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    fig = plt.figure(figsize=(15.6, 6.6), facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 6.8)
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(7.8, 6.45, "How a Possession Is Carved Out of the Raw Event Stream",
            ha="center", va="center", color=INK, fontsize=18, fontweight="bold")
    ax.text(7.8, 6.0,
            "Each dot is one raw Opta event, in chronological order, colored by which team has the ball.",
            ha="center", va="center", color=MUTED, fontsize=10.6)

    # -- The event timeline ------------------------------------------------
    timeline_y = 4.2
    # (x, team) team: 0 = navy, 1 = coral, 2 = goal
    events = [
        (0.9, 0), (1.7, 0), (2.5, 0), (3.3, 0),          # possession 1 (navy) build-up
        (4.5, 1), (5.3, 1), (6.1, 1),                     # possession 2 (coral) short
        (7.3, 0), (8.1, 0), (8.9, 0), (9.7, 0), (10.5, 0),  # possession 3 (navy) longer, ends in shot
        (11.7, 1), (12.5, 1), (13.3, 1),                 # possession 4 (coral) ends in goal
    ]
    boundary_xs = [3.9, 6.7, 11.1, 13.9]
    boundary_labels = [
        "Failed pass\n→ TURNOVER\n(possession ends)",
        "Loose-ball recovery\n→ new possession\nSTARTS",
        "Shot saved\n→ SHOT (no goal)\n(possession ends)",
        "Goal scored\n→ GOAL\n(possession ends)",
    ]

    for x, team in events:
        color = C_NAVY if team == 0 else C_CORAL
        ax.scatter([x], [timeline_y], s=190, color=color, zorder=4, edgecolor=BG, linewidth=1.2)
    # mark the final event as a goal (gold ring)
    gx, gteam = events[-1]
    ax.scatter([gx], [timeline_y], s=340, facecolor="none", edgecolor=C_GOAL, linewidth=2.6, zorder=5)

    ax.plot([0.5, 13.7], [timeline_y, timeline_y], color=MUTED, linewidth=1.2, zorder=1)
    ax.annotate("", xy=(14.3, timeline_y), xytext=(13.9, timeline_y),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    ax.text(14.65, timeline_y, "time", ha="left", va="center", color=MUTED, fontsize=10, style="italic")

    for x, label in zip(boundary_xs, boundary_labels):
        ax.plot([x, x], [timeline_y - 0.55, timeline_y + 0.55], color=INK, linewidth=1.6,
                linestyle=(0, (4, 2)), zorder=3)
        ax.text(x, timeline_y + 1.35, label, ha="center", va="center", color=INK, fontsize=9.1,
                linespacing=1.5, zorder=4)

    # -- Possession span brackets -----------------------------------------
    spans = [(0.6, 3.7, "Possession 1"), (4.2, 6.4, "Possession 2"),
             (7.0, 10.8, "Possession 3"), (11.4, 13.6, "Possession 4")]
    bracket_y = timeline_y - 1.0
    for x0, x1, label in spans:
        ax.plot([x0, x1], [bracket_y, bracket_y], color=MUTED, linewidth=1.4, zorder=2)
        ax.plot([x0, x0], [bracket_y - 0.08, bracket_y + 0.08], color=MUTED, linewidth=1.4, zorder=2)
        ax.plot([x1, x1], [bracket_y - 0.08, bracket_y + 0.08], color=MUTED, linewidth=1.4, zorder=2)
        ax.text((x0 + x1) / 2, bracket_y - 0.35, label, ha="center", va="center",
                color=MUTED, fontsize=9.3, style="italic")

    # -- Legend --------------------------------------------------------------
    leg_y = 0.95
    ax.scatter([3.2], [leg_y], s=170, color=C_NAVY, zorder=4)
    ax.text(3.55, leg_y, "Team A has the ball", ha="left", va="center", color=INK, fontsize=10)
    ax.scatter([7.4], [leg_y], s=170, color=C_CORAL, zorder=4)
    ax.text(7.75, leg_y, "Team B has the ball", ha="left", va="center", color=INK, fontsize=10)
    ax.scatter([11.6], [leg_y], s=280, facecolor="none", edgecolor=C_GOAL, linewidth=2.4, zorder=4)
    ax.text(12.0, leg_y, "terminal event (goal / shot / turnover)", ha="left", va="center", color=INK, fontsize=10)

    ax.text(7.8, 0.35,
            "A possession starts on any change of ball control (completed pass, take-on, aerial win, recovery) "
            "and ends in a goal, a shot, a turnover, or the whistle.",
            ha="center", va="center", color=MUTED, fontsize=9.2, style="italic")

    add_source_line(fig, y=0.012)
    fig.savefig(args.out, dpi=165, facecolor=BG)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
