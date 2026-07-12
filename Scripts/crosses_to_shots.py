"""
Team map of completed open-play crosses whose very next event is a shot
by the same team -- the cross-to-shot chain. Mirrors the project's assist
definition (pass whose immediate next action is a goal): here a cross
"leads to a shot" if the next event in the match is a shot by this team.

Usage: python3 crosses_to_shots.py "Independiente del Valle" [out.png]
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
C_GREEN = "#4ade80"
BG = "#0d1117"
PITCH_LINE = "#2c3a4d"

SET_PIECE_QIDS = {5, 6, 107}
CROSS_QID = 2
SHOT_TYPES = {13, 14, 15, 16}
SHOT_NAMES = {13: "Missed", 14: "Hit Post", 15: "Saved", 16: "Goal"}

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


def minute_value(e):
    return float(e.get("timeMin") or 0) + float(e.get("timeSec") or 0) / 60.0


def pass_end_xy(e, qmap):
    return float(qmap.get(140, e["x"])), float(qmap.get(141, e["y"]))


def collect(files, cid):
    events = []
    player_counts = collections.Counter()
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        evs = [e for e in data["event"] if e.get("periodId") in (1, 2)]
        evs.sort(key=lambda e: (e["periodId"], minute_value(e), e.get("eventId", 0)))

        for i, e in enumerate(evs):
            if e["typeId"] != 1 or e.get("contestantId") != cid or e["outcome"] != 1:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            qids = set(qmap.keys())
            if CROSS_QID not in qids or (SET_PIECE_QIDS & qids):
                continue
            if i + 1 >= len(evs):
                continue
            nxt = evs[i + 1]
            if nxt["typeId"] not in SHOT_TYPES or nxt.get("contestantId") != cid:
                continue
            x0, y0 = float(e["x"]), float(e["y"])
            x1, y1 = pass_end_xy(e, qmap)
            events.append({
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "is_goal": nxt["typeId"] == 16, "shot_type": nxt["typeId"],
                "crosser": e.get("playerName", "?"), "shooter": nxt.get("playerName", "?"),
            })
            player_counts[e.get("playerName", "?")] += 1
    return events, player_counts


def make_plot(team_name, events, player_counts, n_crosses_total, out_path):
    n_total = len(events)
    n_goals = sum(1 for e in events if e["is_goal"])
    pct = n_total / n_crosses_total * 100 if n_crosses_total else 0

    fig = plt.figure(figsize=(11, 10.6))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.955, clean_name(team_name), fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.918, "Crosses Completed to Shots  ·  Ecuador 2026  ·  Season",
             fontsize=12, ha="center", color="#9aa4b2")

    pitch_ax = fig.add_axes([0.04, 0.10, 0.92, 0.72])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                           linewidth=1.1, half=True)
    pitch.draw(ax=pitch_ax)

    for e in sorted(events, key=lambda e: e["is_goal"]):
        color = C_GREEN if e["is_goal"] else C_AMBER
        lw = 2.4 if e["is_goal"] else 1.6
        pitch.lines(e["x0"], e["y0"], e["x1"], e["y1"], ax=pitch_ax, color=color,
                    lw=lw, alpha=0.85, zorder=2, comet=True)
        pitch.scatter(e["x1"], e["y1"], ax=pitch_ax, s=46 if e["is_goal"] else 30,
                      color=color, edgecolors="white", linewidths=1.0, alpha=0.95, zorder=3)

    legend_elems = [
        Line2D([0], [0], color=C_GREEN, linewidth=2.5, label=f"Led to goal ({n_goals})"),
        Line2D([0], [0], color=C_AMBER, linewidth=2.5, label=f"Led to shot, no goal ({n_total - n_goals})"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.885),
               ncol=2, frameon=False, fontsize=10, labelcolor="#c7ccd4")

    caption = f"{n_total} of {n_crosses_total} completed crosses led directly to a shot ({pct:.0f}%)  ·  {n_goals} goals"
    fig.text(0.5, 0.045, caption, fontsize=11.5, ha="center", color="#c7ccd4")

    if player_counts:
        top = player_counts.most_common(5)
        top_str = "  ·  ".join(f"{name} ({n})" for name, n in top)
        fig.text(0.5, 0.022, f"Top providers: {top_str}", fontsize=9, ha="center", color="#6b7684")

    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.006, "Data via Opta | Ecuador 2026 event data · qualifier 2 (cross), excludes set "
             "pieces · shot must be the immediate next event", fontsize=7.3, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"n_to_shot={n_total} n_goals={n_goals} pct_of_completed={pct:.1f}%")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/crosses_to_shots_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]

    n_crosses_total = 0
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e["typeId"] != 1 or e.get("contestantId") != cid or e["outcome"] != 1:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            qids = set(qmap.keys())
            if CROSS_QID in qids and not (SET_PIECE_QIDS & qids):
                n_crosses_total += 1

    events, player_counts = collect(files, cid)
    make_plot(match, events, player_counts, n_crosses_total, out)
