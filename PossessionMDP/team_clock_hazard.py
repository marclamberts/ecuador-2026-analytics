"""
Turnover risk by possession-clock bucket, one line per Liga Pro team —
extends league_visuals.py's league-wide possession_clock_hazard chart
(README.md Section 3's nonstationarity claim) into a team comparison: which
teams get more fragile the longer they hold the ball without progress, and
which stay steady?

Usage: python3 team_clock_hazard.py [--out PATH]
"""
from __future__ import annotations

import argparse
import pathlib
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.ticker import FuncFormatter

import mdp_model as mm
from hierarchy import aggregate_counts
from league_visuals import CLOCK_LABELS
from run_analysis import BG, INK, MUTED, add_source_line

MUTED_2 = "#4a5568"


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Per-team possession-clock turnover hazard.")
    p.add_argument("--data-dir", type=pathlib.Path, default=here.parent / "Event")
    p.add_argument("--out", type=pathlib.Path, default=here / "team_clock_hazard.png")
    p.add_argument("--min-episodes", type=int, default=200)
    return p.parse_args()


def team_turnover_by_clock(transition_counts, cid: str) -> np.ndarray:
    totals = np.zeros((mm.N_CLOCK, 2))  # [other, turnover]
    for (c, lineup), by_s in transition_counts.items():
        if c != cid:
            continue
        for s, by_a in by_s.items():
            clk = s % mm.N_CLOCK
            for a, ctr in by_a.items():
                for outcome, w in ctr.items():
                    totals[clk, 1 if outcome == mm.TURNOVER else 0] += w
    row_sums = totals.sum(axis=1)
    return np.divide(totals[:, 1], row_sums, out=np.full(mm.N_CLOCK, np.nan), where=row_sums > 0)


def main() -> None:
    args = parse_args()
    episodes, id_to_name = mm.load_all_episodes(args.data_dir)
    transition_counts, _ = aggregate_counts(episodes)

    episode_counts = Counter(ep.team for ep in episodes)
    curves = {}
    for cid, name in id_to_name.items():
        if episode_counts.get(cid, 0) < args.min_episodes:
            continue
        curves[name] = team_turnover_by_clock(transition_counts, cid)

    league_mean = np.nanmean(np.vstack(list(curves.values())), axis=0)

    # Highlighting all 16 end-labels crowds the right margin with crossing
    # leader lines; instead spotlight the most and least turnover-resilient
    # teams late in the possession and let the rest sit as quiet context.
    # A dedicated cool/warm palette for just these 6 (rather than reusing
    # the 16-team palette, whose neighboring slots can look nearly
    # identical once the other 10 lines are no longer there to separate
    # them) keeps "steady" and "fragile" visually distinct at a glance.
    n_highlight = 3
    STEADY_COLORS = ["#4da3e8", "#5ec8d8", "#57c785"]
    FRAGILE_COLORS = ["#ff9179", "#e0555f", "#f2c14e"]
    by_final = sorted(curves, key=lambda n: curves[n][-1])
    steadiest = by_final[:n_highlight]
    most_fragile = by_final[-n_highlight:]
    color_map = dict(zip(steadiest, STEADY_COLORS)) | dict(zip(most_fragile, FRAGILE_COLORS))
    highlighted = set(steadiest + most_fragile)

    fig, ax = plt.subplots(figsize=(11.5, 8.5), facecolor=BG)
    ax.set_facecolor(BG)
    x = np.arange(mm.N_CLOCK)

    for name, curve in curves.items():
        if name in highlighted:
            continue
        ax.plot(x, curve * 100, color=MUTED_2, linewidth=1.1, alpha=0.55, zorder=2)

    ax.plot(x, league_mean * 100, color=INK, linewidth=2.4, linestyle=(0, (5, 3)),
            zorder=3, label="League average")

    texts = []
    for name in highlighted:
        curve = curves[name]
        color = color_map[name]
        ax.plot(x, curve * 100, color=color, linewidth=2.2, alpha=0.95, zorder=4)
        ax.scatter(x[-1], curve[-1] * 100, color=color, s=32, zorder=5)
        texts.append(ax.text(x[-1] + 0.07, curve[-1] * 100, name, color=color,
                              fontsize=9.8, va="center", zorder=6, fontweight="bold"))

    adjust_text(texts, ax=ax, only_move={"text": "y", "static": "y", "explode": "y", "pull": "y"},
                force_text=(0.4, 1.6), expand=(1.2, 2.4), max_move=(5, 120), iter_lim=400,
                arrowprops=dict(arrowstyle="-", color=MUTED_2, lw=0.6, alpha=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(CLOCK_LABELS, color=INK, fontsize=10.5)
    ax.set_xlim(-0.15, x[-1] + 0.42)
    ax.set_xlabel("Possession clock (seconds since gaining the ball)", color=MUTED)
    ax.set_ylabel("Turnover risk on the next action", color=MUTED)
    ax.tick_params(colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", color="#1c2534", linewidth=0.8, zorder=0)

    ax.legend(frameon=False, loc="upper center", labelcolor=INK, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("Liga Pro: turnover risk by possession clock", color=INK, fontsize=16, y=0.995)
    fig.text(0.5, 0.955, f"The {n_highlight} steadiest and {n_highlight} most fragile teams late in a "
                         f"possession, highlighted against all {len(curves)} teams",
              color=MUTED, fontsize=10.5, ha="center")
    fig.text(0.5, 0.028, "Most teams share the league-wide shape — a spike in the first 5 seconds after "
                        "winning the ball, then a flatter plateau — but starting level and slope both vary.",
              color=MUTED, fontsize=9.2, ha="center")

    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    add_source_line(fig, y=0.006)
    fig.savefig(args.out, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
