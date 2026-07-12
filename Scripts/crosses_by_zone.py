"""
Team map of open-play crosses (Opta qualifier 2), split by the lane the
cross was delivered from. The pitch width is divided into five lanes
(each 20% of the 0-100 y scale): wide channels (y<=20 or y>=80), half-
spaces (20<y<40 or 60<y<80), and the central lane (40-60). Excludes
corners/free kicks/throw-ins.

Usage: python3 crosses_by_zone.py "Independiente del Valle" <zone> [out.png]
  <zone> in: wide, halfspace
"""
import glob
import json
import re
import sys
import collections

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mplsoccer import VerticalPitch

DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
BG = "#0d1117"
LINE_COLOR = "#c7ccd4"
PITCH_LINE = "#2c3a4d"
ZONE_SHADE = "#1e3a5f"

SET_PIECE_QIDS = {5, 6, 107}
CROSS_QID = 2

# five-lane pitch division, each lane 20% of the width
LANE = 100 / 5
ZONES = {
    "wide": {
        "title": "Crosses From Wide Areas",
        "bands": [(0, LANE), (100 - LANE, 100)],
    },
    "halfspace": {
        "title": "Crosses From the Half-Spaces",
        "bands": [(LANE, 2 * LANE), (100 - 2 * LANE, 100 - LANE)],
    },
}

PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name):
    return PREFIX_RE.sub("", name)


def add_logo(fig, width=0.175, margin=0.018):
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


def build_team_map(files):
    team_cid_sets = collections.defaultdict(list)
    for fn in files:
        m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", fn.split("/")[-1])
        if not m:
            continue
        home, away = m.group(1), m.group(2)
        with open(fn) as f:
            data = json.load(f)
        cids = set(e["contestantId"] for e in data["event"] if "contestantId" in e)
        team_cid_sets[home].append(cids)
        team_cid_sets[away].append(cids)
    team_to_cid = {}
    for team, sets in team_cid_sets.items():
        inter = set.intersection(*sets)
        if len(inter) == 1:
            team_to_cid[team] = next(iter(inter))
    return team_to_cid


def pass_end_xy(e, qmap):
    return float(qmap.get(140, e["x"])), float(qmap.get(141, e["y"]))


def in_bands(y, bands):
    return any(lo <= y <= hi for lo, hi in bands)


def collect(files, cid, bands):
    events = []
    player_counts = collections.Counter()
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e["typeId"] != 1 or e.get("contestantId") != cid:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            qids = set(qmap.keys())
            if CROSS_QID not in qids or SET_PIECE_QIDS & qids:
                continue
            x0, y0 = float(e["x"]), float(e["y"])
            if not in_bands(y0, bands):
                continue
            x1, y1 = pass_end_xy(e, qmap)
            success = e["outcome"] == 1
            events.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "success": success})
            if success:
                player_counts[e.get("playerName", "?")] += 1
    return events, player_counts


def make_plot(team_name, zone, events, player_counts, out_path):
    info = ZONES[zone]
    n_total = len(events)
    n_success = sum(1 for e in events if e["success"])
    pct = n_success / n_total * 100 if n_total else 0

    fig = plt.figure(figsize=(11, 10.6))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.955, clean_name(team_name), fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.918, f"{info['title']}  ·  Ecuador 2026  ·  Season  ·  excludes set pieces",
             fontsize=12, ha="center", color="#9aa4b2")

    pitch_ax = fig.add_axes([0.04, 0.10, 0.92, 0.72])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                           linewidth=1.1, half=True)
    pitch.draw(ax=pitch_ax)

    # shade the origin lane(s) -- polygon verts are (x, y), y = pitch width,
    # so the width-axis coordinate is the SECOND element of each vertex pair.
    for lo, hi in info["bands"]:
        pitch.polygon([[[0, lo], [0, hi], [100, hi], [100, lo]]],
                      ax=pitch_ax, facecolor=ZONE_SHADE, edgecolor="none", alpha=0.45, zorder=0.5)

    for e in sorted(events, key=lambda e: e["success"]):
        color = C_AMBER if e["success"] else LINE_COLOR
        alpha = 0.85 if e["success"] else 0.3
        pitch.lines(e["x0"], e["y0"], e["x1"], e["y1"], ax=pitch_ax, color=color,
                    lw=1.4, alpha=alpha, zorder=2, comet=False)
        pitch.scatter(e["x0"], e["y0"], ax=pitch_ax, s=16, color=color,
                      alpha=alpha, zorder=3, linewidths=0)
        pitch.scatter(e["x1"], e["y1"], ax=pitch_ax, s=30, facecolor="none",
                      edgecolors=color, linewidths=1.2, alpha=alpha, zorder=3)

    legend_elems = [
        Line2D([0], [0], color=C_AMBER, linewidth=2, label="Completed"),
        Line2D([0], [0], color=LINE_COLOR, linewidth=2, alpha=0.5, label="Incomplete"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.885),
               ncol=2, frameon=False, fontsize=10, labelcolor="#c7ccd4")

    caption = f"{n_total} crosses · {pct:.0f}% completed"
    fig.text(0.5, 0.045, caption, fontsize=12, ha="center", color="#c7ccd4")

    if player_counts:
        top = player_counts.most_common(5)
        top_str = "  ·  ".join(f"{name} ({n})" for name, n in top)
        fig.text(0.5, 0.022, f"Top completers: {top_str}", fontsize=9, ha="center", color="#6b7684")

    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.006, "Data via Opta | Ecuador 2026 event data · qualifier 2 (cross), "
             "excludes corners/free kicks/throw-ins", fontsize=8, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"zone={zone} n={n_total} success={n_success} pct={pct:.1f}%")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    zone = sys.argv[2] if len(sys.argv) > 2 else "wide"
    out = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/crosses_{zone}_{team_name.replace(' ', '_')}.png"

    if zone not in ZONES:
        print(f"Unknown zone '{zone}'. Choose from: {list(ZONES)}")
        sys.exit(1)

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    events, player_counts = collect(files, cid, ZONES[zone]["bands"])
    make_plot(match, zone, events, player_counts, out)
