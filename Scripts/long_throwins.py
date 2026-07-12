"""
Team map of long throw-ins (Opta qualifier 107, real distance >= 25m) and
where they land, with a KDE density contour showing the landing zone.
Throw-ins can be launched from either half, so this uses the full pitch.

Usage: python3 long_throwins.py "Independiente del Valle" [out.png]
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

THROW_IN_QID = 107
MIN_DIST_M = 25
DEF_THIRD_END = 100 / 3
FINAL_THIRD_START = 200 / 3

ZONE_COLORS = {
    "Defensive Third": C_NAVY,
    "Middle Third": C_INDIGO,
    "Final Third": C_AMBER,
}


def landing_zone(x1):
    if x1 < DEF_THIRD_END:
        return "Defensive Third"
    if x1 > FINAL_THIRD_START:
        return "Final Third"
    return "Middle Third"

PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name):
    return PREFIX_RE.sub("", name)


def add_logo(fig, width=0.15, margin=0.016):
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


def dist_m(x0, y0, x1, y1):
    dx = (x1 - x0) / 100 * 105.0
    dy = (y1 - y0) / 100 * 68.0
    return (dx ** 2 + dy ** 2) ** 0.5


def pass_end_xy(e, qmap):
    return float(qmap.get(140, e["x"])), float(qmap.get(141, e["y"]))


def collect(files, cid):
    events = []
    player_counts = collections.Counter()
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e["typeId"] != 1 or e.get("contestantId") != cid:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            if THROW_IN_QID not in qmap:
                continue
            x0, y0 = float(e["x"]), float(e["y"])
            x1, y1 = pass_end_xy(e, qmap)
            if dist_m(x0, y0, x1, y1) < MIN_DIST_M:
                continue
            success = e["outcome"] == 1
            events.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "success": success,
                           "zone": landing_zone(x1)})
            if success:
                player_counts[e.get("playerName", "?")] += 1
    return events, player_counts


def make_plot(team_name, events, player_counts, out_path):
    n_total = len(events)
    n_success = sum(1 for e in events if e["success"])
    pct = n_success / n_total * 100 if n_total else 0

    fig = plt.figure(figsize=(11, 13.5))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.965, clean_name(team_name), fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.935, f"Long Throw-Ins & Landing Zone  ·  Ecuador 2026  ·  Season  ·  min. {MIN_DIST_M}m",
             fontsize=12, ha="center", color="#9aa4b2")

    pitch_ax = fig.add_axes([0.04, 0.08, 0.92, 0.77])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                           linewidth=1.1, half=False)
    pitch.draw(ax=pitch_ax)
    for cut in (DEF_THIRD_END, FINAL_THIRD_START):
        pitch.lines(cut, 0, cut, 100, ax=pitch_ax, color="#3a4658", lw=1.0,
                    linestyle=(0, (5, 4)), alpha=0.7, zorder=1, comet=False)

    for e in sorted(events, key=lambda e: e["success"]):
        color = ZONE_COLORS[e["zone"]]
        alpha = 0.85 if e["success"] else 0.3
        marker = "o" if e["success"] else "x"
        pitch.lines(e["x0"], e["y0"], e["x1"], e["y1"], ax=pitch_ax, color=color,
                    lw=1.5, alpha=alpha, zorder=2, comet=True)
        if marker == "o":
            pitch.scatter(e["x1"], e["y1"], ax=pitch_ax, s=30, color=color,
                          edgecolors="white", linewidths=0.8, alpha=alpha, zorder=3)
        else:
            pitch.scatter(e["x1"], e["y1"], ax=pitch_ax, s=30, color=color,
                          marker="x", linewidths=1.6, alpha=alpha, zorder=3)

    legend_elems = [
        Line2D([0], [0], color=C_NAVY, linewidth=2.5, label="Lands in defensive third"),
        Line2D([0], [0], color=C_INDIGO, linewidth=2.5, label="Lands in middle third"),
        Line2D([0], [0], color=C_AMBER, linewidth=2.5, label="Lands in final third"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9aa4b2",
               markeredgecolor="white", markersize=8, label="Retained"),
        Line2D([0], [0], marker="x", color="#9aa4b2", markersize=8,
               markeredgewidth=1.6, linewidth=0, label="Lost"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.895),
               ncol=3, frameon=False, fontsize=9.5, labelcolor="#c7ccd4")

    caption = f"{n_total} long throw-ins · {pct:.0f}% retained possession"
    fig.text(0.5, 0.045, caption, fontsize=12, ha="center", color="#c7ccd4")

    if player_counts:
        top = player_counts.most_common(5)
        top_str = "  ·  ".join(f"{name} ({n})" for name, n in top)
        fig.text(0.5, 0.022, f"Top throwers: {top_str}", fontsize=9, ha="center", color="#6b7684")

    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.006, "Data via Opta | Ecuador 2026 event data · qualifier 107 (throw-in), "
             "line colour = third of pitch the throw lands in", fontsize=7.3, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"n={n_total} success={n_success} pct={pct:.1f}%")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/long_throwins_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    events, player_counts = collect(files, cid)
    make_plot(match, events, player_counts, out)
