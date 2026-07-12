"""
Attack vs Defense rating quadrant: each team positioned by attacking
strength (xG for per game, percentile-ranked 0-100 across the league) and
defensive strength (xG against per game, inverted and percentile-ranked,
so higher = harder to score against). Uses the real shot-level xg column
in Danger/*_danger_models.csv rather than raw goals, since we have it.

Usage: python3 attack_defense_quadrant.py [out.png]
"""
import glob
import os
import re
import sys
import collections

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text

import pi_ratings_lib as pil

DANGER_DIR = "/Users/marclamberts/Event data/Ecuador 2026/Danger"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"


def add_logo(fig, width=0.13, margin=0.014):
    import matplotlib.image as mpimg
    try:
        img = mpimg.imread(LOGO_PATH)
    except FileNotFoundError:
        return
    fig_w, fig_h = fig.get_size_inches()
    img_h, img_w = img.shape[0], img.shape[1]
    width_in = width * fig_w
    height_in = width_in * (img_h / img_w)
    height = height_in / fig_h
    left = 1 - margin - width
    bottom = 1 - margin - height
    logo_ax = fig.add_axes([left, bottom, width, height], zorder=10)
    logo_ax.patch.set_alpha(0)
    logo_ax.set_xlim(0, img_w)
    logo_ax.set_ylim(img_h, 0)
    logo_ax.imshow(img)
    logo_ax.axis("off")


def percentile_rank(values):
    """0-100 percentile rank for each value in a dict {key: value}."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(items)
    ranks = {}
    for i, (k, v) in enumerate(items):
        ranks[k] = i / (n - 1) * 100 if n > 1 else 50.0
    return ranks


def collect_xg(team_to_cid):
    cid_to_team = {v: k for k, v in team_to_cid.items()}
    xg_for = collections.defaultdict(float)
    xg_against = collections.defaultdict(float)
    games = collections.defaultdict(int)

    csvs = sorted(glob.glob(f"{DANGER_DIR}/*_danger_models.csv"))
    for path in csvs:
        df = pd.read_csv(path)
        cids_in_match = df["contestant_id"].unique()
        teams_in_match = [cid_to_team[c] for c in cids_in_match if c in cid_to_team]
        if len(teams_in_match) != 2:
            continue
        t0, t1 = teams_in_match
        xg0 = df.loc[df["contestant_id"] == team_to_cid[t0], "xg"].sum()
        xg1 = df.loc[df["contestant_id"] == team_to_cid[t1], "xg"].sum()
        xg_for[t0] += xg0
        xg_against[t0] += xg1
        xg_for[t1] += xg1
        xg_against[t1] += xg0
        games[t0] += 1
        games[t1] += 1

    return xg_for, xg_against, games


def make_plot(d, out_path):
    team_to_cid, points = d["team_to_cid"], d["points"]
    xg_for, xg_against, games = collect_xg(team_to_cid)

    xg_for_pg = {t: xg_for[t] / games[t] for t in games if games[t] > 0}
    xg_against_pg = {t: xg_against[t] / games[t] for t in games if games[t] > 0}

    attack_rank = percentile_rank(xg_for_pg)
    # invert defense: teams with LOW xG against get a HIGH defense rating
    defense_rank = percentile_rank({t: -v for t, v in xg_against_pg.items()})

    teams = list(games.keys())
    teams.sort(key=lambda t: -points.get(t, 0))

    fig, ax = plt.subplots(figsize=(15.5, 10.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.axhline(50, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.45, zorder=1)
    ax.axvline(50, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.45, zorder=1)

    pts_vals = [points.get(t, 0) for t in teams]
    cmap = LinearSegmentedColormap.from_list("pts", ["#e05c5c", "#ffc247", "#4ade80"])
    pmin, pmax = min(pts_vals), max(pts_vals)
    sizes = [220 + (points.get(t, 0) - pmin) / (pmax - pmin) * 700 if pmax > pmin else 400 for t in teams]

    xs = [attack_rank[t] for t in teams]
    ys = [defense_rank[t] for t in teams]
    sc = ax.scatter(xs, ys, s=sizes, c=pts_vals, cmap=cmap, edgecolors="white",
                    linewidths=1.3, alpha=0.9, zorder=3)

    texts = []
    for i, t in enumerate(teams):
        rank = i + 1
        label = f"#{rank} {pil.clean_name(t)}"
        texts.append(ax.text(xs[i], ys[i] + 2.2, label, fontsize=10, color=TEXT_MAIN,
                             fontweight="bold" if rank <= 3 else "normal", ha="center", zorder=4))
    adjust_text(texts, ax=ax, only_move={"text": "y"}, expand=(1.2, 1.6))

    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.text(2, 100, "Defensively solid", fontsize=10, color="#5b9bd5", style="italic", va="top")
    ax.text(98, 100, "Elite", fontsize=10, color="#4ade80", style="italic", va="top", ha="right")
    ax.text(2, 2, "Struggling", fontsize=10, color="#e05c5c", style="italic")
    ax.text(98, 2, "Attack-heavy", fontsize=10, color="#ffc247", style="italic", ha="right")

    ax.set_xlabel("ATTACK RATING  (xG for per game, percentile)", fontsize=12, color=TEXT_MAIN,
                 fontweight="bold", labelpad=12)
    ax.set_ylabel("DEFENSE RATING  (xG against per game, inverted percentile)", fontsize=12,
                 color=TEXT_MAIN, fontweight="bold", labelpad=12)
    ax.tick_params(colors=TEXT_SUB, labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.5)

    cbar = fig.colorbar(sc, ax=ax, pad=0.012, fraction=0.035)
    cbar.set_label("Points", color=TEXT_MAIN, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB)

    fig.text(0.05, 0.965, "Ecuador 2026  ·  Attack vs Defense Rating — Team Quadrants",
             fontsize=20, fontweight="bold", color="white")
    fig.text(0.05, 0.93, "Each team positioned by attacking and defensive strength (real shot-level xG)  ·  "
             "Dot size = points total", fontsize=11.5, color=TEXT_SUB)
    fig.text(0.05, 0.908, "Top-right = elite (score & prevent)  ·  Bottom-right = attack-heavy  ·  "
             "Top-left = defensive  ·  Bottom-left = struggling", fontsize=9, color="#6b7684")
    fig.text(0.05, 0.012, "Data via Opta | Ecuador 2026 event data · xg = mean of 5 calibrated shot "
             "models (danger_score methodology)", fontsize=8, color="#6b7684")
    fig.text(0.98, 0.012, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.06, right=0.94, top=0.86, bottom=0.09)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/attack_defense_quadrant.png"
    d = pil.load_all()
    make_plot(d, out)
