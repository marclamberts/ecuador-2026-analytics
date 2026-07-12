"""
xDanger shot map builder for the Ecuador 2026 Opta F24-style event dataset.
Sizes/colors shots by danger_score (mean of xG, PSxG, situation danger,
P(on target) and xGOT) from Danger/*_danger_models.csv, labelled "xDanger".

Usage: python3 shotmap_xdanger.py "E. Mero"
"""
import csv
import glob
import json
import os
import re
import sys
import collections

import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.lines import Line2D
from mplsoccer import VerticalPitch

LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"


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


DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"
DANGER_DIR = os.path.join(DATA_DIR, "Danger")

SHOT_TYPES = {13, 14, 15, 16}

# ---------------------------------------------------------------- team map --

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
    return {v: k for k, v in team_to_cid.items()}


# --------------------------------------------------------- danger csv load --

def load_danger_csv(fn_base):
    csv_path = os.path.join(DANGER_DIR, fn_base[:-5] + "_danger_models.csv")
    rows = {}
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["event_id"])] = row
    return rows


def body_part(qids):
    if 15 in qids:
        return "Head"
    if 72 in qids:
        return "Left Foot"
    if 20 in qids:
        return "Right Foot"
    return "Other"


def outcome_bucket(row):
    if row["is_goal"] == "1":
        return "Goal"
    if row["is_post"] == "1":
        return "Post"
    if row["is_outfield_block"] == "1":
        return "Blocked"
    if row["is_on_target"] == "1":
        return "On Target"
    return "Off Target"


# ------------------------------------------------------------- extraction --

def collect_shots(player_name):
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    cid_to_team = build_team_map(files)
    shots = []
    matches_played = set()
    for fn in files:
        fn_base = fn.split("/")[-1]
        danger_rows = load_danger_csv(fn_base)
        with open(fn) as f:
            data = json.load(f)
        player_in_match = False
        for e in data["event"]:
            if e.get("playerName") != player_name:
                continue
            player_in_match = True
            if e["typeId"] not in SHOT_TYPES:
                continue
            row = danger_rows.get(e["id"])
            if row is None:
                continue
            if row["is_penalty"] == "1":
                continue
            qids = set(q["qualifierId"] for q in e["qualifier"])
            shots.append({
                "x": e["x"], "y": e["y"],
                "body_part": body_part(qids),
                "outcome": outcome_bucket(row),
                "xg": float(row["xg"]),
                "psxg": float(row["psxg"]),
                "xdanger": float(row["danger_score"]),
                "match": fn_base,
                "team": cid_to_team.get(e["contestantId"], "Unknown"),
                "min": e.get("timeMin"),
            })
        if player_in_match:
            matches_played.add(fn_base)
    return shots, matches_played


# ------------------------------------------------------------------ plot --

COL_GOAL = "#ffc247"
COL_ON = "#f06fa3"
COL_OFF = "#2f8fd1"
COL_BLOCKED = "#7b7fd6"
BG = "#0d1117"


def shot_distance_m(x_opta, y_opta):
    dx = (100 - x_opta) / 100 * 105.0
    dy = (y_opta - 50) / 100 * 68.0
    return (dx ** 2 + dy ** 2) ** 0.5


def make_plot(player_name, shots, matches_played, out_path):
    n_shots = len(shots)
    n_goals = sum(1 for s in shots if s["outcome"] == "Goal")
    n_on = sum(1 for s in shots if s["outcome"] in ("Goal", "On Target"))
    n_blocked = sum(1 for s in shots if s["outcome"] == "Blocked")
    n_off = sum(1 for s in shots if s["outcome"] in ("Off Target", "Post"))
    total_xg = sum(s["xg"] for s in shots)
    total_xdanger = sum(s["xdanger"] for s in shots)
    xdanger_per_shot = total_xdanger / n_shots if n_shots else 0
    team = shots[0]["team"] if shots else "Unknown"

    by_bp = collections.defaultdict(lambda: [0, 0.0])
    for s in shots:
        by_bp[s["body_part"]][0] += 1
        by_bp[s["body_part"]][1] += s["xdanger"]

    n_close = sum(1 for s in shots if shot_distance_m(s["x"], s["y"]) < 12)
    n_far = sum(1 for s in shots if shot_distance_m(s["x"], s["y"]) >= 20)

    # -------- layout constants (figure-fraction coords, fig is 15x10in) --------
    PITCH_LEFT, PITCH_WIDTH = 0.02, 0.62
    PITCH_TOP = 0.850
    PITCH_ASPECT = 1.298  # width_in / height_in for a whitespace-free half pitch
    FIG_W, FIG_H = 15, 11
    pitch_width_in = PITCH_WIDTH * FIG_W
    pitch_height_in = pitch_width_in / PITCH_ASPECT
    PITCH_HEIGHT = pitch_height_in / FIG_H
    PITCH_BOTTOM = PITCH_TOP - PITCH_HEIGHT

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG)

    pitch_ax = fig.add_axes([PITCH_LEFT, PITCH_BOTTOM, PITCH_WIDTH, PITCH_HEIGHT])
    pitch = VerticalPitch(pitch_type="opta", half=True, pitch_color=BG,
                           line_color="#3a4a5c", linewidth=1.3, pad_bottom=2)
    pitch.draw(ax=pitch_ax)

    def size_for(danger):
        return 90 + danger * 2600

    order = {"Off Target": 0, "Blocked": 1, "Post": 1.5, "On Target": 2, "Goal": 3}
    for s in sorted(shots, key=lambda s: order.get(s["outcome"], 0)):
        color = {"Goal": COL_GOAL, "On Target": COL_ON, "Post": COL_ON,
                 "Off Target": COL_OFF, "Blocked": COL_BLOCKED}[s["outcome"]]
        pitch.scatter(s["x"], s["y"], ax=pitch_ax, s=size_for(s["xdanger"]),
                      color=color, edgecolors="white", linewidths=1.1,
                      marker="o", zorder=order.get(s["outcome"], 0) + 2,
                      alpha=0.92)

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_GOAL, markeredgecolor="white", markersize=13, label="Goal"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_ON, markeredgecolor="white", markersize=13, label="On target"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_OFF, markeredgecolor="white", markersize=13, label="Off target"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_BLOCKED, markeredgecolor="white", markersize=13, label="Blocked"),
    ]
    fig.legend(handles=legend_elems, loc="center",
               bbox_to_anchor=(PITCH_LEFT + PITCH_WIDTH / 2, PITCH_TOP + 0.028),
               bbox_transform=fig.transFigure, ncol=4, frameon=False, fontsize=9.5, labelcolor="white")

    fig.text(0.02, 0.965, player_name, fontsize=30, fontweight="bold", family="sans-serif", color="white")
    fig.text(0.02, 0.932, "Shot Map (xDanger) · excl. penalties", fontsize=13, color="#c7ccd4")
    fig.text(0.02, 0.907, f"{team} · Ecuador 2026 · {len(matches_played)} matches played",
             fontsize=11, color="#9aa4b2")

    fig_w, fig_h = fig.get_size_inches()
    aspect = fig_w / fig_h  # compensates so Ellipse renders as a visual circle

    STAT_LABEL_Y, STAT_VAL_Y = PITCH_BOTTOM - 0.035, PITCH_BOTTOM - 0.095

    def stat_bubble(x, label, value, color):
        fig.text(x, STAT_LABEL_Y, label, fontsize=11, ha="center", color="#d7dbe0")
        r = 0.036
        ell = matplotlib.patches.Ellipse((x, STAT_VAL_Y), width=2 * r, height=2 * r * aspect,
                                          transform=fig.transFigure,
                                          facecolor=color, edgecolor="none", zorder=5)
        fig.add_artist(ell)
        fig.text(x, STAT_VAL_Y, str(value), fontsize=16, fontweight="bold", ha="center",
                 va="center", color="white" if color != "#4a5568" else "#f0f2f5", zorder=6)

    bubble_xs = [PITCH_LEFT + PITCH_WIDTH * f for f in (0.08, 0.24, 0.40)]
    stat_bubble(bubble_xs[0], "Shots", n_shots, "#4a5568")
    stat_bubble(bubble_xs[1], "On Target", n_on, COL_ON)
    stat_bubble(bubble_xs[2], "Goals", n_goals, COL_GOAL)
    plain_xs = [PITCH_LEFT + PITCH_WIDTH * f for f in (0.60, 0.77, 0.94)]
    for x, label, val in zip(plain_xs, ("xDanger", "xDanger/Shot", "xG"),
                              (f"{total_xdanger:.2f}", f"{xdanger_per_shot:.2f}", f"{total_xg:.2f}")):
        fig.text(x, STAT_LABEL_Y, label, fontsize=11, ha="center", color="#d7dbe0")
        fig.text(x, STAT_VAL_Y, val, fontsize=18, fontweight="bold", ha="center", va="center", color="white")

    fig.text(PITCH_LEFT, 0.02,
             "xDanger = danger_score (mean of xG, PSxG, situation danger, P(on target), xGOT)"
             " from Danger/*_danger_models.csv.",
             fontsize=8.5, color="#7b8794")

    # ---------------- right stat panel ----------------
    rx0, rx1 = 0.70, 0.97
    RTOP, RBOTTOM = PITCH_TOP + 0.028, STAT_VAL_Y - 0.045
    section_h = (RTOP - RBOTTOM) / 4

    def section_header(y, title):
        fig.text(rx0, y, title, fontsize=11.5, fontweight="bold", color="#f0f2f5")
        fig.add_artist(plt.Line2D([rx0, rx1], [y - 0.010, y - 0.010],
                                   transform=fig.transFigure, color="#2c3540", linewidth=1))

    # 1) shots by body part
    y0 = RTOP
    section_header(y0, "SHOTS (xDANGER) BY BODY PART")
    bp_items = [
        ("LEFT FOOT", *tuple(by_bp["Left Foot"])),
        ("RIGHT FOOT", *tuple(by_bp["Right Foot"])),
        ("HEADED", *tuple(by_bp["Head"])),
        ("OTHER", *tuple(by_bp["Other"])),
    ]
    xs2 = [rx0, rx0 + (rx1 - rx0) / 2]
    for i, (label, count, danger) in enumerate(bp_items):
        col = xs2[i % 2]
        row = i // 2
        yy = y0 - 0.045 - row * 0.062
        fig.text(col, yy, label, fontsize=9.5, color="#8b93a1")
        fig.text(col, yy - 0.026, f"{count}  ({danger:.2f})", fontsize=15, fontweight="bold", color="#ffffff")

    # 2) shot accuracy
    y0 -= section_h
    section_header(y0, "SHOT ACCURACY")
    acc_items = [
        ("ON TARGET", n_on, n_on / n_shots * 100 if n_shots else 0),
        ("BLOCKED", n_blocked, n_blocked / n_shots * 100 if n_shots else 0),
        ("OFF TARGET", n_off, n_off / n_shots * 100 if n_shots else 0),
    ]
    xs3 = [rx0, rx0 + (rx1 - rx0) / 2]
    for i, (label, count, pct) in enumerate(acc_items):
        col = xs3[i % 2]
        row = i // 2
        yy = y0 - 0.045 - row * 0.062
        fig.text(col, yy, label, fontsize=9.5, color="#8b93a1")
        fig.text(col, yy - 0.026, f"{count} ({pct:.1f}%)", fontsize=14, fontweight="bold", color="#ffffff")

    # 3) shot distance
    y0 -= section_h
    section_header(y0, "SHOT DISTANCE")
    dist_items = [("SHOTS: <12M", n_close, n_close / n_shots * 100 if n_shots else 0),
                  ("SHOTS: 20M+", n_far, n_far / n_shots * 100 if n_shots else 0)]
    for i, (label, count, pct) in enumerate(dist_items):
        col = xs3[i % 2]
        yy = y0 - 0.045
        fig.text(col, yy, label, fontsize=9.5, color="#8b93a1")
        fig.text(col, yy - 0.026, f"{count} ({pct:.1f}%)", fontsize=14, fontweight="bold", color="#ffffff")

    # 4) outcome breakdown
    y0 -= section_h
    section_header(y0, "SHOT OUTCOME BREAKDOWN")
    outc = collections.Counter(s["outcome"] for s in shots)
    yy = y0 - 0.05
    row_h = min(0.036, max(0.024, (y0 - RBOTTOM) / 5.5))
    for label in ["Goal", "On Target", "Blocked", "Off Target", "Post"]:
        if outc.get(label, 0) == 0:
            continue
        fig.text(rx0, yy, label, fontsize=10.5, color="#c7ccd4")
        fig.text(rx1, yy, str(outc[label]), fontsize=10.5, fontweight="bold", color="#ffffff", ha="right")
        yy -= row_h

    fig.text(rx1, 0.02, "Source: Ecuador 2026 event data", fontsize=8.5, color="#7b8794", ha="right")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG, bbox_inches=None)
    print("Saved:", out_path)
    print(f"Shots={n_shots} Goals={n_goals} OnTarget={n_on} Blocked={n_blocked} OffTarget={n_off} "
          f"xDanger={total_xdanger:.2f} xDanger/shot={xdanger_per_shot:.3f} xG={total_xg:.2f}")


if __name__ == "__main__":
    player = sys.argv[1] if len(sys.argv) > 1 else "E. Mero"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/shotmap_xdanger_{player.replace(' ', '_')}.png"
    shots, matches = collect_shots(player)
    if not shots:
        print(f"No shots found for player '{player}'")
        sys.exit(1)
    make_plot(player, shots, matches, out)
