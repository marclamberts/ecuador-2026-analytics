"""
Zone Control: 18-zone grid (6 columns x 3 rows) showing each team's own
touch/action distribution across the pitch as a % share, all 16 teams
plotted as small multiples in a single PNG. Every event with a location
(pass, take-on, shot, tackle, interception, clearance, aerial, etc.)
counts as a touch, in that team's own attacking direction (x=100 = their
attacking end, as normalized by Opta).

Usage: python3 zone_control.py [out.png]
"""
import glob
import json
import math
import re
import sys
import collections

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch

import pi_ratings_lib as pil

DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
BG = "#0d1117"
PANEL_BG = "#11161f"
PITCH_LINE = "#3a4658"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"

N_COLS, N_ROWS = 6, 3
CMAP = LinearSegmentedColormap.from_list("zone_control", ["#131a24", "#1f4e79", "#2f8fd1", "#ffc247"])

# administrative/metadata event types (cards, subs, delays, deleted events,
# formation changes, etc.) -- verified these carry placeholder x=0,y=0
# rather than a real location, so they'd otherwise fake a hot zone at the
# pitch corner for every single team
NON_TOUCH_TYPES = {17, 18, 19, 27, 28, 30, 32, 34, 37, 40, 43, 58, 65, 70, 71, 79, 84}


def add_logo(fig, width=0.09, margin=0.012):
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


def collect_touches(files, team_to_cid):
    cid_to_team = {v: k for k, v in team_to_cid.items()}
    touches = collections.defaultdict(lambda: [[], []])
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data.get("event", []):
            if e.get("typeId") in NON_TOUCH_TYPES:
                continue
            cid = e.get("contestantId")
            t = cid_to_team.get(cid)
            if t is None:
                continue
            x, y = e.get("x"), e.get("y")
            if x is None or y is None:
                continue
            if x == 0 and y == 0:
                continue
            touches[t][0].append(x)
            touches[t][1].append(y)
    return touches


def make_plot(d, out_path):
    files, team_to_cid, points = d["files"], d["team_to_cid"], d["points"]
    touches = collect_touches(files, team_to_cid)
    teams = [t for t in team_to_cid if touches.get(t)]
    teams.sort(key=lambda t: -points.get(t, 0))

    pitch = Pitch(pitch_type="opta", pitch_color=PANEL_BG, line_color=PITCH_LINE, linewidth=1.1)

    pct_grids = {}
    for t in teams:
        xs, ys = touches[t]
        stat = pitch.bin_statistic(xs, ys, statistic="count", bins=(N_COLS, N_ROWS))
        total = stat["statistic"].sum()
        stat["statistic"] = stat["statistic"] / total * 100
        pct_grids[t] = stat

    all_vals = np.concatenate([g["statistic"].ravel() for g in pct_grids.values()])
    vmax = math.ceil(all_vals.max() / 2) * 2

    n = len(teams)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.35 * nrows))
    fig.patch.set_facecolor(BG)
    axes = axes.ravel()

    for i, t in enumerate(teams):
        ax = axes[i]
        ax.set_facecolor(BG)
        pitch.draw(ax=ax)
        stat = pct_grids[t]
        pitch.heatmap(stat, ax=ax, cmap=CMAP, vmin=0, vmax=vmax, edgecolors=BG, linewidth=1.4, zorder=1)
        for row in range(N_ROWS):
            for col in range(N_COLS):
                val = stat["statistic"][row, col]
                cx, cy = stat["cx"][row, col], stat["cy"][row, col]
                txt_color = "#0d1117" if val >= vmax * 0.55 else TEXT_MAIN
                ax.text(cx, cy, f"{val:.0f}%", ha="center", va="center", fontsize=9.5,
                        color=txt_color, fontweight="bold", zorder=3)
        rank = i + 1
        ax.set_title(f"#{rank}  {pil.clean_name(t)}", fontsize=11.5, color=TEXT_MAIN,
                    fontweight="bold", pad=6)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.35, 0.028, 0.3, 0.014])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("% of team's own touches in zone", color=TEXT_SUB, fontsize=9.5)
    cbar.ax.xaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB, labelsize=8.5)
    cbar.outline.set_visible(False)

    fig.text(0.045, 0.978, "Ecuador 2026  ·  Zone Control — 18-Zone Grid (6×3)",
             fontsize=21, fontweight="bold", color="white")
    fig.text(0.045, 0.958, "Each team's own touches, all competitive actions, own attacking direction "
             "(→ attacking right)  ·  All 16 teams", fontsize=11, color=TEXT_SUB)
    fig.text(0.045, 0.006, "Data via Opta | Ecuador 2026 event data · % of each team's own season touches "
             "falling in each of 18 pitch zones", fontsize=8, color="#6b7684")
    fig.text(0.975, 0.006, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.06, hspace=0.38, wspace=0.12)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/zone_control.png"
    d = pil.load_all()
    make_plot(d, out)
