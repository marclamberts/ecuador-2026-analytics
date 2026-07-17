"""
A conceptual schematic of the three-level Bayesian hierarchy (README.md
Section 5 / paper Section 5): league -> team -> lineup shrinkage, shown as
a cascade of pooled estimates plus "blend bars" illustrating how a
sparsely-observed lineup gets pulled hard toward its team's tendency while
a well-observed lineup is dominated by its own data.

This is a diagram, not a data chart -- it takes no arguments and reads no
event data -- so it lives standalone rather than inside run_analysis.py.

Usage: python3 hierarchy_schematic.py [--out PATH]
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from run_analysis import BG, C_CORAL, C_NAVY, INK, MUTED, add_source_line


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Bayesian hierarchy concept schematic.")
    p.add_argument("--out", type=pathlib.Path, default=here / "hierarchy_schematic.png")
    return p.parse_args()


def box(ax, xy, w, h, text, sub=None, ec=INK, text_color=INK, sub_color=MUTED,
        fontsize=13, sub_fontsize=10, lw=2.0, boxstyle="round,pad=0.32,rounding_size=0.12"):
    x, y = xy
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=boxstyle,
                            facecolor=BG, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(patch)
    if sub:
        ax.text(x, y + h * 0.16, text, ha="center", va="center", color=text_color,
                 fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(x, y - h * 0.28, sub, ha="center", va="center", color=sub_color,
                 fontsize=sub_fontsize, zorder=4)
    else:
        ax.text(x, y, text, ha="center", va="center", color=text_color,
                 fontsize=fontsize, fontweight="bold", zorder=4)
    return patch


def arrow(ax, start, end, color=MUTED, lw=2.0, shrink=6):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                         color=color, linewidth=lw, zorder=2, shrinkA=shrink, shrinkB=shrink)
    ax.add_patch(a)


def blend_bar(ax, cx, cy, w, h, own_frac, label):
    """Horizontal stacked bar: coral = pulled toward parent's prior, navy = own data."""
    left = cx - w / 2
    pulled_w = w * (1 - own_frac)
    own_w = w * own_frac
    ax.add_patch(Rectangle((left, cy - h / 2), pulled_w, h, facecolor=C_CORAL, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((left + pulled_w, cy - h / 2), own_w, h, facecolor=C_NAVY, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((left, cy - h / 2), w, h, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4))
    ax.text(cx, cy - h * 1.35, label, ha="center", va="center", color=MUTED, fontsize=9.3)


def main() -> None:
    args = parse_args()

    fig = plt.figure(figsize=(15.4, 9.4), facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 9.6)
    ax.set_facecolor(BG)
    ax.axis("off")

    # -- Title ----------------------------------------------------------------
    ax.text(7.7, 9.2, "Pooling Strength Across the Season", ha="center", va="center",
            color=INK, fontsize=19, fontweight="bold")
    ax.text(7.7, 8.72,
            "A three-level Bayesian hierarchy: a sparse lineup is shrunk hard toward its team's tendency;\n"
            "a settled lineup with hundreds of possessions is dominated by its own data.",
            ha="center", va="center", color=MUTED, fontsize=10.6, linespacing=1.6)

    # -- League level -----------------------------------------------------------
    league_xy = (7.7, 7.35)
    box(ax, league_xy, 5.4, 1.2, "LEAGUE", sub="base rate across all 16 teams  ·  κ = 2.0",
        ec=INK, fontsize=14, sub_fontsize=10.3, lw=2.0)

    # -- Team level ---------------------------------------------------------------
    team_xy = (7.7, 5.55)
    box(ax, team_xy, 4.6, 1.2, "TEAM", sub="pooled across a club's lineups  ·  κ = 8.0",
        ec=C_NAVY, fontsize=14, sub_fontsize=10.3, lw=2.2)
    arrow(ax, (league_xy[0], league_xy[1] - 0.62), (team_xy[0], team_xy[1] + 0.62), color=MUTED, lw=2.0)
    ax.text(league_xy[0] + 3.1, (league_xy[1] + team_xy[1]) / 2, "shrinkage pull\ntoward parent",
            ha="center", va="center", color=MUTED, fontsize=9, style="italic", linespacing=1.4)

    # -- Lineup level: three example lineups with blend bars -----------------------
    lineup_y = 3.55
    lineup_w, lineup_h = 4.0, 1.35
    lineup_xs = [2.6, 7.7, 12.8]
    lineups = [
        ("Fringe substitution", "11 minutes, 1 match", 0.12),
        ("Rotation option", "~40 possessions", 0.50),
        ("First-choice XI", "240+ possessions", 0.88),
    ]

    for x, (name, vol, own_frac) in zip(lineup_xs, lineups):
        box(ax, (x, lineup_y), lineup_w, lineup_h, name, sub=vol, ec=C_CORAL if own_frac < 0.4 else
            (INK if own_frac < 0.7 else C_NAVY), fontsize=12.3, sub_fontsize=9.6, lw=1.9)
        arrow(ax, (team_xy[0] + (x - team_xy[0]) * 0.12, team_xy[1] - 0.62),
              (x, lineup_y + lineup_h / 2 + 0.02), color=MUTED, lw=1.5)
        blend_bar(ax, x, lineup_y - lineup_h / 2 - 0.55, lineup_w, 0.4, own_frac,
                  f"{own_frac:.0%} own data  ·  {1 - own_frac:.0%} pulled from team")

    # -- Legend ------------------------------------------------------------------
    leg_y = 1.15
    ax.add_patch(Rectangle((3.9, leg_y - 0.14), 0.34, 0.28, facecolor=C_CORAL, edgecolor="none"))
    ax.text(4.4, leg_y, "pulled toward the parent level's prior", ha="left", va="center",
            color=INK, fontsize=10)
    ax.add_patch(Rectangle((9.3, leg_y - 0.14), 0.34, 0.28, facecolor=C_NAVY, edgecolor="none"))
    ax.text(9.8, leg_y, "the lineup's own observed data", ha="left", va="center",
            color=INK, fontsize=10)

    ax.text(7.7, 0.55,
            "Counts are also weighted by recency (≈2% decay per week), so current form outweighs early-season play "
            "without discarding it.",
            ha="center", va="center", color=MUTED, fontsize=9.3, style="italic")

    add_source_line(fig, y=0.012)
    fig.savefig(args.out, dpi=165, facecolor=BG)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
