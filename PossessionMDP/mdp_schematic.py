"""
A conceptual schematic of the possession-clock MDP for the "Chapter 1"
explainer (README.md Section 4 / paper Section 4): state -> action ->
transition -> next state or terminal outcome, with the possession clock
shown as the variable that makes the transition nonstationary.

This is a diagram, not a data chart -- it takes no arguments and reads
no event data -- so it lives standalone rather than inside run_analysis.py.

Usage: python3 mdp_schematic.py [--out PATH]
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionStyle, FancyArrowPatch, FancyBboxPatch

from run_analysis import BG, C_CORAL, C_NAVY, INK, MUTED, add_source_line

C_GOAL = "#f2c14e"
C_TERMINAL = "#6b7684"


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="MDP concept schematic.")
    p.add_argument("--out", type=pathlib.Path, default=here / "mdp_schematic.png")
    return p.parse_args()


def box(ax, xy, w, h, text, sub=None, ec=INK, text_color=INK, sub_color=MUTED,
        fontsize=12, sub_fontsize=9.5, lw=1.8, boxstyle="round,pad=0.32,rounding_size=0.12"):
    x, y = xy
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=boxstyle,
                            facecolor=BG, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(patch)
    if sub:
        ax.text(x, y + h * 0.16, text, ha="center", va="center", color=text_color,
                 fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(x, y - h * 0.26, sub, ha="center", va="center", color=sub_color,
                 fontsize=sub_fontsize, zorder=4)
    else:
        ax.text(x, y, text, ha="center", va="center", color=text_color,
                 fontsize=fontsize, fontweight="bold", zorder=4, linespacing=1.4)
    return patch


def arrow(ax, start, end, color=MUTED, lw=1.8, style="-", connectionstyle=None, shrink=6):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                         color=color, linewidth=lw, linestyle=style, zorder=2,
                         shrinkA=shrink, shrinkB=shrink,
                         connectionstyle=connectionstyle)
    ax.add_patch(a)


def main() -> None:
    args = parse_args()

    fig = plt.figure(figsize=(16.6, 9.0), facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0, 9.3)
    ax.set_facecolor(BG)
    ax.axis("off")

    # -- Title --------------------------------------------------------------
    ax.text(8.3, 8.9, "The Possession as a Markov Decision Process",
            ha="center", va="center", color=INK, fontsize=19, fontweight="bold")
    ax.text(8.3, 8.42,
            "Every possession is a walk through this diagram: a state, an attempted action, a transition\n"
            "whose odds depend on the possession clock, and either another state or an absorbing outcome.",
            ha="center", va="center", color=MUTED, fontsize=10.6, linespacing=1.6)

    # -- State box ------------------------------------------------------------
    state_xy = (2.6, 6.3)
    box(ax, state_xy, 3.5, 1.55, "STATE  s = (z, c)",
        sub="pitch zone z  ×  possession clock c", ec=C_NAVY, fontsize=13.5, sub_fontsize=10, lw=2.1)

    # -- Action pills ---------------------------------------------------------
    actions = ["Advance", "Sideways", "Back", "Cross", "Shot"]
    action_y = 6.3
    pill_w, pill_h, gap = 1.5, 0.68, 0.24
    total_w = len(actions) * pill_w + (len(actions) - 1) * gap
    action_x0 = 11.6
    xs = [action_x0 - total_w / 2 + pill_w / 2 + i * (pill_w + gap) for i in range(len(actions))]

    ax.text(action_x0, action_y + 1.05, "ACTION  a", ha="center", va="center",
            color=INK, fontsize=13.5, fontweight="bold")
    ax.text(action_x0, action_y + 0.72, "the attempted decision, independent of outcome",
            ha="center", va="center", color=MUTED, fontsize=9.8)
    for x, label in zip(xs, actions):
        box(ax, (x, action_y), pill_w, pill_h, label, ec=C_CORAL, fontsize=10.2, lw=1.5,
            boxstyle="round,pad=0.26,rounding_size=0.32")

    arrow(ax, (state_xy[0] + 1.75, state_xy[1]), (action_x0 - total_w / 2 - 0.35, action_y), color=MUTED, lw=1.9)

    # -- Transition kernel ------------------------------------------------------
    trans_xy = (8.0, 4.0)
    box(ax, trans_xy, 5.4, 1.5, "TRANSITION  P(s′ | s, a)",
        sub="depends on the clock bucket c  →  nonstationary", ec=INK, sub_color=C_CORAL,
        fontsize=13.5, sub_fontsize=10.2, lw=1.9)

    for x in xs:
        arrow(ax, (x, action_y - pill_h / 2 - 0.02), (trans_xy[0] + (x - action_x0) * 0.42, trans_xy[1] + 0.78),
              color=MUTED, lw=1.2)

    # -- Continue branch: loop back to state -------------------------------------
    cont_xy = (2.6, 1.55)
    box(ax, cont_xy, 3.5, 1.4, "NEXT STATE  s′ = (z′, c+1)",
        sub="possession continues", ec=C_NAVY, fontsize=12.5, sub_fontsize=9.8, lw=2.0)
    arrow(ax, (trans_xy[0] - 2.75, trans_xy[1] - 0.6), (cont_xy[0] + 1.9, cont_xy[1] + 0.42), color=C_NAVY, lw=1.9)
    arrow(ax, (cont_xy[0] - 1.5, cont_xy[1] + 0.55), (state_xy[0] - 1.5, state_xy[1] - 0.65),
          color=C_NAVY, lw=2.1, style=(0, (5, 3)),
          connectionstyle=ConnectionStyle("Arc3", rad=-0.55), shrink=2)
    ax.text(0.55, 3.9, "loops until the possession ends", ha="center", va="center",
            color=MUTED, fontsize=9.3, style="italic", rotation=90)

    # -- Terminal outcomes ---------------------------------------------------------
    term_y = 1.55
    term_w, term_gap = 2.05, 0.45
    term_specs = [("GOAL", "r(s′) = 1", C_GOAL), ("SHOT\n(no goal)", "r(s′) = 0", C_TERMINAL),
                  ("TURNOVER", "r(s′) = 0", C_TERMINAL)]
    term_x0 = 12.85
    term_total = 3 * term_w + 2 * term_gap
    term_xs = [term_x0 - term_total / 2 + term_w / 2 + i * (term_w + term_gap) for i in range(3)]
    for (label, r, color), x in zip(term_specs, term_xs):
        box(ax, (x, term_y), term_w, 1.4, label, sub=r, ec=color, sub_color=color,
            fontsize=11.5, sub_fontsize=9.8, lw=2.0, boxstyle="round,pad=0.3,rounding_size=0.1")
        arrow(ax, (trans_xy[0] + 2.5, trans_xy[1] - 0.55), (x, term_y + 0.72), color=color, lw=1.5)
    ax.text(term_x0, 3.35, "absorbing terminal states", ha="center", va="center",
            color=MUTED, fontsize=9.3, style="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=BG, edgecolor="none"))

    add_source_line(fig, y=0.012)
    fig.savefig(args.out, dpi=165, facecolor=BG)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
