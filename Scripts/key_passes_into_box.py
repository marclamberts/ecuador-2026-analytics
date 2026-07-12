"""
Team map of key passes (Opta's own keyPass/assist tags) whose end point
lands inside the penalty box. Assists are additionally linked back to
their shot's xg (via qualifier 55 on the shot event) to show xA, same
approach as key_pass_map.py.

Usage: python3 key_passes_into_box.py "Independiente del Valle" [out.png]
"""
import csv as csvmod
import glob
import json
import os
import re
import sys
import collections

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mplsoccer import VerticalPitch

DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"
DANGER_DIR = "/Users/marclamberts/Event data/Ecuador 2026/Danger"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
C_GREEN = "#4ade80"
BG = "#0d1117"
PITCH_LINE = "#2c3a4d"

BOX_X = 83.0
BOX_Y_LO, BOX_Y_HI = 21.1, 78.9
EXCLUDE_QIDS = {5, 6}  # free kick taken, corner taken

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


def in_box(x, y):
    return x >= BOX_X and BOX_Y_LO <= y <= BOX_Y_HI


def collect(files, cid):
    events = []
    player_counts = collections.Counter()
    total_xa = 0.0

    for fn in files:
        fn_base = fn.split("/")[-1]
        csv_path = os.path.join(DANGER_DIR, fn_base[:-5] + "_danger_models.csv")
        shot_xg = {}
        if os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                for row in csvmod.DictReader(f):
                    shot_xg[int(row["event_id"])] = float(row["xg"])

        with open(fn) as f:
            data = json.load(f)
        evs = data["event"]

        assist_pass_xa = {}
        for e in evs:
            if e["typeId"] not in (13, 14, 15, 16):
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            related = qmap.get(55)
            if related is None:
                continue
            try:
                related_local_id = int(related)
            except ValueError:
                continue
            xg = shot_xg.get(e["id"])
            if xg is not None:
                assist_pass_xa[related_local_id] = xg

        for e in evs:
            if e["typeId"] != 1 or e.get("contestantId") != cid:
                continue
            if not (e.get("keyPass") == 1 or e.get("assist") == 1):
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            if EXCLUDE_QIDS & set(qmap.keys()):
                continue
            x0, y0 = float(e["x"]), float(e["y"])
            x1 = float(qmap.get(140, x0))
            y1 = float(qmap.get(141, y0))
            if not in_box(x1, y1):
                continue
            is_assist = e.get("assist") == 1
            xa = assist_pass_xa.get(e["eventId"], 0.04 if is_assist else 0.0)
            events.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                           "is_assist": is_assist, "xa": xa})
            player_counts[e.get("playerName", "?")] += 1
            total_xa += xa

    return events, player_counts, total_xa


def make_plot(team_name, events, player_counts, total_xa, out_path):
    n_total = len(events)
    n_assist = sum(1 for e in events if e["is_assist"])

    fig = plt.figure(figsize=(11, 10.6))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.955, clean_name(team_name), fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.918, "Key Passes Into the Box  ·  Open Play Only  ·  Ecuador 2026  ·  Season",
             fontsize=12, ha="center", color="#9aa4b2")

    pitch_ax = fig.add_axes([0.04, 0.10, 0.92, 0.72])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                           linewidth=1.1, half=True)
    pitch.draw(ax=pitch_ax)

    for e in sorted(events, key=lambda e: e["is_assist"]):
        color = C_GREEN if e["is_assist"] else C_AMBER
        lw = 2.2 if e["is_assist"] else 1.3
        alpha = 0.9 if e["is_assist"] else 0.55
        pitch.lines(e["x0"], e["y0"], e["x1"], e["y1"], ax=pitch_ax, color=color,
                    lw=lw, alpha=alpha, zorder=2, comet=True)
        pitch.scatter(e["x1"], e["y1"], ax=pitch_ax, s=40 if e["is_assist"] else 22,
                      color=color, edgecolors="white", linewidths=0.8, alpha=alpha, zorder=3)

    legend_elems = [
        Line2D([0], [0], color=C_GREEN, linewidth=2.5, label=f"Assist ({n_assist})"),
        Line2D([0], [0], color=C_AMBER, linewidth=2.5, label=f"Key pass, no assist ({n_total - n_assist})"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.885),
               ncol=2, frameon=False, fontsize=10, labelcolor="#c7ccd4")

    caption = f"{n_total} key passes into the box this season  ·  {n_assist} assists  ·  {total_xa:.1f} total xA"
    fig.text(0.5, 0.045, caption, fontsize=11.5, ha="center", color="#c7ccd4")

    if player_counts:
        top = player_counts.most_common(5)
        top_str = "  ·  ".join(f"{name} ({n})" for name, n in top)
        fig.text(0.5, 0.022, f"Top providers: {top_str}", fontsize=9, ha="center", color="#6b7684")

    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.006, "Data via Opta | Ecuador 2026 event data · excludes corners/free kicks · "
             "keyPass/assist tags, xA = shot xg "
             "linked via qualifier 55", fontsize=7.3, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"n={n_total} assists={n_assist} total_xa={total_xa:.2f}")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/key_passes_into_box_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    events, player_counts, total_xa = collect(files, cid)
    make_plot(match, events, player_counts, total_xa, out)
