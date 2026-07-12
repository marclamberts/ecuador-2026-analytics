"""
Team map of defensive actions (tackles, interceptions, clearances, aerial
duels) with a dashed line marking the average height (x position) of
those actions -- the standard "defensive line" indicator: a high average
line means a high press, a low one means a deep block.

Usage: python3 defensive_actions.py "Independiente del Valle" [out.png]
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
PITCH_LINE = "#2c3a4d"

ACTION_TYPES = {7: "Tackle", 8: "Interception", 12: "Clearance", 44: "Aerial Duel"}
ACTION_COLORS = {
    "Tackle": C_NAVY,
    "Interception": C_PINK,
    "Clearance": C_AMBER,
    "Aerial Duel": C_INDIGO,
}

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


def collect(files, cid):
    events = []
    type_counts = collections.Counter()
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e["typeId"] not in ACTION_TYPES or e.get("contestantId") != cid:
                continue
            if e.get("x") is None or e.get("y") is None:
                continue
            action = ACTION_TYPES[e["typeId"]]
            events.append({"x": float(e["x"]), "y": float(e["y"]), "action": action})
            type_counts[action] += 1
    return events, type_counts


def make_plot(team_name, events, type_counts, out_path):
    n_total = len(events)
    avg_x = sum(e["x"] for e in events) / n_total if n_total else 0
    avg_x_m = avg_x / 100 * 105

    fig = plt.figure(figsize=(11, 13.5))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.965, clean_name(team_name), fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.935, "Defensive Actions & Average Defensive Line  ·  Ecuador 2026  ·  Season",
             fontsize=12, ha="center", color="#9aa4b2")

    pitch_ax = fig.add_axes([0.13, 0.08, 0.83, 0.79])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                           linewidth=1.1, half=False)
    pitch.draw(ax=pitch_ax)

    for e in events:
        color = ACTION_COLORS[e["action"]]
        pitch.scatter(e["x"], e["y"], ax=pitch_ax, s=26, color=color,
                      alpha=0.55, linewidths=0, zorder=2)

    pitch.lines(avg_x, 0, avg_x, 100, ax=pitch_ax, color="white", lw=2.2,
               linestyle=(0, (6, 4)), alpha=0.95, zorder=3, comet=False)
    pitch.annotate(f"AVG. LINE: {avg_x:.0f}", xy=(avg_x, 103), ax=pitch_ax, ha="right",
                   va="center", fontsize=10.5, fontweight="bold", color="white",
                   zorder=4, clip_on=False)

    legend_elems = [Line2D([0], [0], marker="o", color="none", markerfacecolor=ACTION_COLORS[a],
                            markersize=9, label=f"{a} ({type_counts.get(a, 0)})")
                     for a in ["Tackle", "Interception", "Clearance", "Aerial Duel"]]
    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.885),
               ncol=4, frameon=False, fontsize=9, labelcolor="#c7ccd4")

    caption = f"{n_total} defensive actions this season  ·  avg. defensive line {avg_x:.0f} (~{avg_x_m:.0f}m from own goal)"
    fig.text(0.5, 0.045, caption, fontsize=12, ha="center", color="#c7ccd4")

    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.006, "Data via Opta | Ecuador 2026 event data · defensive actions = tackles, "
             "interceptions, clearances, aerial duels", fontsize=7.5, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"n={n_total} avg_x={avg_x:.2f} breakdown={dict(type_counts)}")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/defensive_actions_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    events, type_counts = collect(files, cid)
    make_plot(match, events, type_counts, out)
