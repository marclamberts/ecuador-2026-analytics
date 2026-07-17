"""
A conceptual schematic contrasting what the current simulator does (a
single fixed-dynamics pass) against the fictitious-play extension
described as future work in the Game-Theoretic Limitations discussion
(paper Section 11): iterating policy against a simulated opponent
response instead of assuming the opponent stands still.

This is a diagram, not a data chart -- it takes no arguments and reads
no event data -- so it lives standalone rather than inside run_analysis.py.

Usage: python3 game_theory_schematic.py [--out PATH]
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionStyle, FancyArrowPatch, FancyBboxPatch

from run_analysis import BG, C_CORAL, C_NAVY, INK, MUTED, add_source_line


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Game-theoretic extension concept schematic.")
    p.add_argument("--out", type=pathlib.Path, default=here / "game_theory_schematic.png")
    return p.parse_args()


def box(ax, xy, w, h, text, sub=None, ec=INK, text_color=INK, sub_color=MUTED,
        fontsize=12.5, sub_fontsize=9.3, lw=2.0, alpha=1.0):
    x, y = xy
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=0.12",
                            facecolor=BG, edgecolor=ec, linewidth=lw, zorder=3, alpha=alpha)
    ax.add_patch(patch)
    if sub:
        ax.text(x, y + h * 0.18, text, ha="center", va="center", color=text_color, fontsize=fontsize,
                 fontweight="bold", zorder=4, alpha=alpha, linespacing=1.3)
        ax.text(x, y - h * 0.26, sub, ha="center", va="center", color=sub_color, fontsize=sub_fontsize,
                 zorder=4, alpha=alpha, linespacing=1.3)
    else:
        ax.text(x, y, text, ha="center", va="center", color=text_color, fontsize=fontsize,
                 fontweight="bold", zorder=4, alpha=alpha, linespacing=1.3)
    return patch


def curved_arrow(ax, start, end, color, lw, style="-", rad=0.28, shrink=140):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17, color=color, linewidth=lw,
                         linestyle=style, zorder=2, shrinkA=shrink, shrinkB=shrink,
                         connectionstyle=ConnectionStyle("Arc3", rad=rad))
    ax.add_patch(a)


def main() -> None:
    args = parse_args()

    fig = plt.figure(figsize=(12.6, 11.6), facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 11.8)
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(6.3, 11.45, "One Fixed-Dynamics Pass vs. a Two-Sided Loop", ha="center", va="center",
            color=INK, fontsize=18, fontweight="bold")
    ax.text(6.3, 10.98,
            "What this project computes (solid) against the fictitious-play extension it doesn't yet attempt (dashed).",
            ha="center", va="center", color=MUTED, fontsize=10.4)

    cx, cy, r = 6.3, 6.1, 3.55
    top = (cx, cy + r)
    right = (cx + r * 0.97, cy + r * 0.18)
    bottom = (cx, cy - r)
    left = (cx - r * 0.97, cy + r * 0.18)

    box(ax, top, 4.4, 1.35, "TEAM'S POLICY π", sub="e.g. \"+15% more direct\"", ec=C_NAVY, fontsize=13.5)
    box(ax, right, 4.6, 1.45, "SIMULATE OUTCOME", sub="against fixed dynamics\nP(s′ | s, a)", ec=C_NAVY,
        fontsize=13)
    box(ax, bottom, 4.6, 1.45, "OPPONENT ADAPTS", sub="best-response dynamics\nP′(s′ | s, a)",
        ec=MUTED, text_color=MUTED, sub_color=MUTED, fontsize=13, alpha=0.85)
    box(ax, left, 4.6, 1.45, "UPDATED DYNAMICS", sub="fed back into the\nnext simulated round",
        ec=MUTED, text_color=MUTED, sub_color=MUTED, fontsize=13, alpha=0.85)

    # This-paper step: solid navy arrow, top -> right
    curved_arrow(ax, top, right, C_NAVY, 2.4, rad=-0.32)
    ax.text(cx + r * 0.66, cy + r * 0.62, "what this paper\ncomputes", ha="center", va="center",
            color=C_NAVY, fontsize=9.8, style="italic", linespacing=1.4, fontweight="bold")

    # Future-work steps: dashed muted arrows, right -> bottom -> left -> top
    curved_arrow(ax, right, bottom, MUTED, 1.8, style=(0, (5, 3)), rad=-0.32, shrink=95)
    curved_arrow(ax, bottom, left, MUTED, 1.8, style=(0, (5, 3)), rad=-0.32, shrink=95)
    curved_arrow(ax, left, top, MUTED, 1.8, style=(0, (5, 3)), rad=-0.32, shrink=95)
    ax.text(cx, bottom[1] - 1.45 / 2 - 0.5, "future work: fictitious play -- iterate until neither side wants to adjust",
            ha="center", va="center", color=MUTED, fontsize=10, style="italic")

    add_source_line(fig, y=0.012)
    fig.savefig(args.out, dpi=165, facecolor=BG)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
