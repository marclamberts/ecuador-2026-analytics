"""
A conceptual (not data-driven) landscape placing this project's
possession-clock MDP against existing football possession/action-value
literature along two axes: how much of a possession gets valued at once,
and how explicitly time/clock enters the model. Positions are editorial
judgment, not measurements -- the point is the relative layout, not
precise coordinates.

This is a diagram, not a data chart -- it takes no arguments and reads
no event data -- so it lives standalone rather than inside run_analysis.py.

Usage: python3 related_work_schematic.py [--out PATH]
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from run_analysis import BG, C_CORAL, C_NAVY, INK, MUTED, add_source_line


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Related-work landscape schematic.")
    p.add_argument("--out", type=pathlib.Path, default=here / "related_work_schematic.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    fig = plt.figure(figsize=(13.6, 10.4), facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 10.6)
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(6.8, 10.3, "Where the Possession-Clock MDP Sits", ha="center", va="center",
            color=INK, fontsize=18, fontweight="bold")
    ax.text(6.8, 9.85,
            "A rough, editorial landscape, not a measurement -- positions reflect how each approach treats\n"
            "scope and time, not a ranking of quality.",
            ha="center", va="center", color=MUTED, fontsize=10.2, linespacing=1.6)

    # -- Plot area ------------------------------------------------------------
    x0, x1 = 1.6, 12.2
    y0, y1 = 1.6, 8.9
    ax.plot([x0, x1], [y0, y0], color=MUTED, linewidth=1.4)
    ax.plot([x0, x0], [y0, y1], color=MUTED, linewidth=1.4)
    ax.annotate("", xy=(x1 + 0.35, y0), xytext=(x1, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    ax.annotate("", xy=(x0, y1 + 0.35), xytext=(x0, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))

    ax.text((x0 + x1) / 2, 0.75, "scope of what gets valued", ha="center", va="center",
            color=INK, fontsize=11.5, fontweight="bold")
    ax.text(x0 + 0.1, 1.05, "a single action", ha="left", va="center", color=MUTED, fontsize=9.5)
    ax.text(x1 - 0.1, 1.05, "a whole possession", ha="right", va="center", color=MUTED, fontsize=9.5)

    ax.text(0.75, (y0 + y1) / 2, "how explicitly time enters the model", ha="center", va="center",
            color=INK, fontsize=11.5, fontweight="bold", rotation=90)
    ax.text(1.05, y0 + 0.15, "time-stationary", ha="left", va="bottom", color=MUTED, fontsize=9.5, rotation=90)
    ax.text(1.05, y1 - 0.15, "explicit clock / continuous time", ha="left", va="top", color=MUTED, fontsize=9.5,
            rotation=90)

    # -- Points -----------------------------------------------------------------
    points = [
        (3.2, 2.6, "VAEP", "Decroos et al. (2019)\nvalues individual actions", C_TERMINAL := "#6b7684"),
        (4.9, 3.1, "Expected Threat (xT)", "zone-to-zone Markov chain,\nstationary transition matrix", C_TERMINAL),
        (9.2, 6.7, "Expected Possession\nValue (EPV)", "Fernández, Bornn & Cervone (2019)\ntracking-based, per-moment value", C_NAVY),
        (11.1, 8.3, "Semi-Markov\npossessions", "Le Coz, Boustila & Imbach (2026)\ncontinuous sojourn-time model", C_NAVY),
        (9.7, 4.9, "THIS WORK", "possession-clock MDP\nzone x 5-bucket clock discretization", C_CORAL),
    ]
    for x, y, label, sub, color in points:
        is_this = label == "THIS WORK"
        ax.scatter([x], [y], s=420 if is_this else 260, color=color, zorder=5,
                   edgecolor=INK if is_this else "none", linewidth=2.2 if is_this else 0)
        ax.text(x, y + (0.62 if not is_this else 0.68), label, ha="center", va="bottom", color=INK,
                fontsize=11.5 if is_this else 10.3, fontweight="bold", zorder=6, linespacing=1.3)
        ax.text(x, y - (0.42 if not is_this else 0.5), sub, ha="center", va="top", color=MUTED,
                fontsize=8.8, zorder=6, linespacing=1.4)

    # -- Side note on match-outcome models -----------------------------------------
    note_xy = (6.8, 0.0)
    note = ("Not shown: Dixon-Coles and pi-ratings (Constantinou & Fenton) model match scorelines, not "
            "possessions -- a different axis entirely. This project borrows their shrinkage logic, not their scope.")
    box = FancyBboxPatch((1.6, -0.85), 10.6, 0.85, boxstyle="round,pad=0.25,rounding_size=0.12",
                          facecolor=BG, edgecolor=MUTED, linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(6.9, -0.42, note, ha="center", va="center", color=MUTED, fontsize=9.2, style="italic",
            linespacing=1.5, wrap=True)

    ax.set_ylim(-1.4, 10.6)
    add_source_line(fig, y=0.012)
    fig.savefig(args.out, dpi=165, facecolor=BG)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
