"""
Set Piece Routines -- Corner Delivery Zones: for every corner a team
takes, where do they aim it? Deliveries are mirrored onto a single side
(near-post low, far-post high) so left- and right-side corners combine
into one "near / central / far post" x "six-yard / edge-of-box" 6-zone
breakdown, plus a short/recycled corner share. All 16 teams, one PNG.

Usage: python3 corner_routines.py [out.png]
"""
import sys
import json
import math
import collections

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import VerticalPitch

import pi_ratings_lib as pil

LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
BG = "#0d1117"
PANEL_BG = "#11161f"
PITCH_LINE = "#3a4658"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"

CMAP = LinearSegmentedColormap.from_list("corners", ["#131a24", "#1f4e79", "#2f8fd1", "#ffc247"])

BOX_FRONT_X = 83.0
SIX_YARD_FRONT_X = 94.0
NEAR_CUT, FAR_CUT = 41.0, 59.0
BOX_Y_LO, BOX_Y_HI = 21.0, 79.0

ZONES = [
    ("near", "edge", NEAR_CUT, BOX_Y_LO, BOX_FRONT_X, SIX_YARD_FRONT_X - BOX_FRONT_X),
    ("central", "edge", FAR_CUT - NEAR_CUT, NEAR_CUT, BOX_FRONT_X, SIX_YARD_FRONT_X - BOX_FRONT_X),
    ("far", "edge", BOX_Y_HI - FAR_CUT, FAR_CUT, BOX_FRONT_X, SIX_YARD_FRONT_X - BOX_FRONT_X),
    ("near", "six", NEAR_CUT, BOX_Y_LO, SIX_YARD_FRONT_X, 100 - SIX_YARD_FRONT_X),
    ("central", "six", FAR_CUT - NEAR_CUT, NEAR_CUT, SIX_YARD_FRONT_X, 100 - SIX_YARD_FRONT_X),
    ("far", "six", BOX_Y_HI - FAR_CUT, FAR_CUT, SIX_YARD_FRONT_X, 100 - SIX_YARD_FRONT_X),
]


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


def pass_end_xy(e):
    qmap = {q["qualifierId"]: q.get("value") for q in e.get("qualifier", [])}
    ex, ey = qmap.get(140), qmap.get(141)
    return (float(ex) if ex is not None else e["x"]), (float(ey) if ey is not None else e["y"])


def zone_key(end_x, end_y, start_y):
    my = 100 - end_y if start_y >= 50 else end_y
    if end_x < BOX_FRONT_X:
        return "short"
    col = "near" if my < NEAR_CUT else ("far" if my > FAR_CUT else "central")
    row = "six" if end_x >= SIX_YARD_FRONT_X else "edge"
    return (col, row)


def collect_corners(files, team_to_cid):
    cid_to_team = {v: k for k, v in team_to_cid.items()}
    counts = collections.defaultdict(lambda: collections.Counter())

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data.get("event", []):
            if e.get("typeId") != 1:
                continue
            qids = {q["qualifierId"] for q in e.get("qualifier", [])}
            if 6 not in qids:
                continue
            t = cid_to_team.get(e.get("contestantId"))
            if t is None:
                continue
            ex, ey = pass_end_xy(e)
            key = zone_key(ex, ey, e["y"])
            counts[t][key] += 1

    return counts


def make_plot(d, out_path):
    files, team_to_cid, points = d["files"], d["team_to_cid"], d["points"]
    counts = collect_corners(files, team_to_cid)
    teams = [t for t in team_to_cid if sum(counts.get(t, {}).values()) >= 10]
    teams.sort(key=lambda t: -points.get(t, 0))

    pct = {}
    short_pct = {}
    for t in teams:
        c = counts[t]
        total = sum(c.values())
        pct[t] = {(col, row): c.get((col, row), 0) / total * 100 for col, row, *_ in ZONES}
        short_pct[t] = c.get("short", 0) / total * 100

    all_vals = [v for t in teams for v in pct[t].values()]
    vmax = math.ceil(max(all_vals) / 2) * 2

    pitch = VerticalPitch(pitch_type="opta", pitch_color=PANEL_BG, line_color=PITCH_LINE, linewidth=1.1,
                          half=True)

    n = len(teams)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.9 * nrows))
    fig.patch.set_facecolor(BG)
    axes = axes.ravel()

    for i, t in enumerate(teams):
        ax = axes[i]
        ax.set_facecolor(BG)
        pitch.draw(ax=ax)
        # VerticalPitch transposes the axes: native x = pitch width (y),
        # native y = pitch length (x) -- raw ax.add_patch needs that swap
        for col, row, w, y0, x0, width_x in ZONES:
            val = pct[t][(col, row)]
            color = CMAP(min(1.0, val / vmax))
            ax.add_patch(Rectangle((y0, x0), w, width_x, facecolor=color, edgecolor=BG,
                                   linewidth=1.2, zorder=2))
            txt_color = "#0d1117" if val >= vmax * 0.55 else TEXT_MAIN
            ax.text(y0 + w / 2, x0 + width_x / 2, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9.5, color=txt_color, fontweight="bold", zorder=3)
        ax.set_xlim(8, 92)
        ax.set_ylim(68, 101)
        rank = i + 1
        ax.set_title(f"#{rank}  {pil.clean_name(t)}", fontsize=11, color=TEXT_MAIN,
                    fontweight="bold", pad=16)
        ax.text(0.03, 0.97, f"Short/recycled: {short_pct[t]:.0f}%", fontsize=8, color=TEXT_SUB,
                va="top", transform=ax.transAxes)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.35, 0.025, 0.3, 0.013])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("% of team's own corners delivered to zone", color=TEXT_SUB, fontsize=9.5)
    cbar.ax.xaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB, labelsize=8.5)
    cbar.outline.set_visible(False)

    fig.text(0.045, 0.978, "Ecuador 2026  ·  Set Piece Routines — Corner Delivery Zones",
             fontsize=21, fontweight="bold", color="white")
    fig.text(0.045, 0.960, "Near/far post mirrored onto one side  ·  Minimum 10 corners  ·  All "
             "qualifying teams", fontsize=11, color=TEXT_SUB)
    fig.text(0.045, 0.005, "Data via Opta | Ecuador 2026 event data · Zones: near/central/far post x "
             "six-yard box/edge-of-box, by pass end-location · \"Short\" = delivery not reaching the box",
             fontsize=8, color="#6b7684")
    fig.text(0.975, 0.005, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.06, hspace=0.45, wspace=0.12)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/corner_routines.png"
    d = pil.load_all()
    make_plot(d, out)
