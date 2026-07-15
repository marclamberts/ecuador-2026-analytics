"""
Season-long form guide for every Liga Pro team: match-by-match shot-ending
rate per possession (the fraction of a team's possession episodes, per
README.md's MDP formulation, that terminate in a shot attempt), with a
5-match rolling average line over faint per-match dots.

Small multiples (one panel per team, shared y-axis) rather than 16
overlaid lines, since a single-axes spaghetti plot of 16 raw series is
unreadable regardless of color choice. Panels are ordered by each team's
rolling-average level at the end of the sample, so the grid itself reads
top-left-to-bottom-right as a ranking.

Usage: python3 team_form_trend.py [--out PATH] [--window N]
"""
from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import mdp_model as mm
from palette import team_color_map
from run_analysis import BG, INK, MUTED, add_source_line

MUTED_2 = "#4a5568"
DOT_ALPHA = 0.35


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Match-by-match team form guide (rolling shot rate).")
    p.add_argument("--data-dir", type=pathlib.Path, default=here.parent / "Event")
    p.add_argument("--out", type=pathlib.Path, default=here / "team_form_trend.png")
    p.add_argument("--window", type=int, default=5, help="Rolling-average window, in matches.")
    p.add_argument("--min-matches", type=int, default=8)
    return p.parse_args()


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    out = np.empty(len(values))
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = np.mean(values[lo:i + 1])
    return out


def main() -> None:
    args = parse_args()
    files = mm.find_matches(args.data_dir)
    id_to_name = {cid: name for name, cid in mm.build_team_to_cid(files).items()}
    print(f"Loading {len(files)} matches from {args.data_dir} ...")

    by_team = defaultdict(list)  # cid -> list of (date, shot_rate)
    for path in files:
        match = mm.load_match(path)
        date = mm.match_date(path)
        episodes = mm.extract_episodes(match)
        terminals_by_cid = defaultdict(list)
        for ep in episodes:
            terminals_by_cid[ep.team].append(ep.terminal)
        for cid, terminals in terminals_by_cid.items():
            n = len(terminals)
            if n == 0:
                continue
            shots = sum(1 for t in terminals if t in (mm.GOAL, mm.NOGOAL_SHOT))
            by_team[cid].append((date, shots / n))

    series = {}
    for cid, points in by_team.items():
        if len(points) < args.min_matches:
            continue
        points.sort(key=lambda p: p[0])
        values = [v for _, v in points]
        series[cid] = rolling_mean(values, args.window)

    order = sorted(series, key=lambda cid: series[cid][-1], reverse=True)
    names = [id_to_name.get(cid, cid) for cid in order]
    color_map = team_color_map(names)

    n = len(order)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.5 * nrows), facecolor=BG, sharey=True)
    axes = np.array(axes).reshape(-1)

    y_max = max(s.max() for s in series.values()) * 1.15

    for i, cid in enumerate(order):
        ax = axes[i]
        ax.set_facecolor(BG)
        name = id_to_name.get(cid, cid)
        color = color_map[name]
        vals = series[cid]
        x = np.arange(1, len(vals) + 1)

        raw_vals = [v for _, v in sorted(by_team[cid], key=lambda p: p[0])]
        ax.scatter(x, raw_vals, s=10, color=color, alpha=DOT_ALPHA, zorder=2, linewidth=0)
        ax.plot(x, vals, color=color, linewidth=2.2, zorder=3)
        ax.fill_between(x, vals, 0, color=color, alpha=0.08, zorder=1)

        ax.set_ylim(0, y_max)
        ax.set_xlim(1, max(x.max(), args.min_matches))
        ax.tick_params(colors=MUTED_2, labelsize=7.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#232c3a")
        ax.set_title(f"{i+1}. {name}", color=INK, fontsize=10, loc="left", pad=4)
        ax.text(0.98, 0.92, f"{vals[-1]*100:.0f}%", transform=ax.transAxes, ha="right", va="top",
                 color=color, fontsize=10, fontweight="bold")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Liga Pro: team form guide", color=INK, fontsize=17, y=0.998)
    fig.text(0.5, 0.965,
              f"{args.window}-match rolling average of shot-ending rate per possession "
              "(share of possessions reaching a shot attempt) — faint dots are single-match values",
              color=MUTED, fontsize=10.5, ha="center")
    fig.text(0.01, 0.5, "Shots per possession", color=MUTED, fontsize=10, rotation=90, va="center")

    fig.tight_layout(rect=(0.015, 0, 1, 0.94))
    add_source_line(fig)
    fig.savefig(args.out, dpi=155, facecolor=BG)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
