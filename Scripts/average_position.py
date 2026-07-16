"""
Average Position chart for a single player: where on the pitch they perform
defensive actions, passes, receptions, and shots, on average.

Four mini pitches (2x2), each showing every event of that type as a faint
dot plus a highlighted marker + dashed crosshair at the mean (x, y) location.

Usage: python3 average_position.py "E. Mero"
"""
import glob
import json
import re
import sys
import collections

from mplsoccer import VerticalPitch

LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"


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


import matplotlib.pyplot as plt  # noqa: E402  (after backend-agnostic helpers above)

DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
BG = "#0d1117"
PITCH_LINE = "#2c3a4d"

DEF_ACTION_TYPES = {7, 8, 12, 44}  # Tackle, Interception, Clearance, Aerial Duel
SHOT_TYPES = {13, 14, 15, 16}
PENALTY_QID = 9


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


# ------------------------------------------------------------- extraction --

def collect(player_name):
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    cid_to_team = build_team_map(files)
    defensive, passing, receiving, shooting = [], [], [], []
    team = "Unknown"

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        events = data["event"]
        for i, e in enumerate(events):
            name = e.get("playerName")
            tid = e["typeId"]
            if e.get("x") is None or e.get("y") is None:
                continue

            if name == player_name:
                team = cid_to_team.get(e.get("contestantId"), team)

                if tid in DEF_ACTION_TYPES:
                    defensive.append({"x": e["x"], "y": e["y"]})

                elif tid == 1:
                    passing.append({"x": e["x"], "y": e["y"]})

                elif tid in SHOT_TYPES:
                    qids = set(q["qualifierId"] for q in e["qualifier"])
                    if PENALTY_QID in qids:
                        continue
                    shooting.append({"x": e["x"], "y": e["y"]})

            # reception heuristic: successful pass by a teammate immediately
            # followed by this player's next action for the same team
            if tid == 1 and e.get("outcome") == 1 and name != player_name:
                if i + 1 < len(events):
                    nxt = events[i + 1]
                    if (nxt.get("playerName") == player_name
                            and nxt.get("contestantId") == e.get("contestantId")):
                        qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
                        ex = float(qmap.get(140, e["x"]))
                        ey = float(qmap.get(141, e["y"]))
                        receiving.append({"x": ex, "y": ey})

    return defensive, passing, receiving, shooting, team


# ------------------------------------------------------------------ plot --

PANELS = [
    ("DEFENSIVE ACTIONS", C_PINK, "AVG DEFENSIVE LINE"),
    ("PASSING", C_NAVY, "AVG PASS ORIGIN"),
    ("RECEIVING", C_PURPLE, "AVG RECEIVE POINT"),
    ("SHOOTING", C_AMBER, "AVG SHOT LOCATION"),
]


def make_plot(player_name, defensive, passing, receiving, shooting, team, out_path):
    datasets = [defensive, passing, receiving, shooting]

    fig = plt.figure(figsize=(13.5, 15))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.975, player_name, fontsize=28, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.950, "Average Position by Action Type", fontsize=13, fontweight="bold",
              ha="center", color="#d7dbe0")
    fig.text(0.5, 0.928, f"{team} · Ecuador 2026 · All Competitions", fontsize=11, ha="center", color="#9aa4b2")

    col_x = [0.045, 0.535]
    row_y = [0.500, 0.045]
    width, height = 0.435, 0.395

    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                           linewidth=1.0, half=False)

    for idx, ((title, color, line_label), events) in enumerate(zip(PANELS, datasets)):
        r, c = divmod(idx, 2)
        ax = fig.add_axes([col_x[c], row_y[r], width, height])
        pitch.draw(ax=ax)
        ax.set_title(title, fontsize=12.5, fontweight="bold", color="#f0f2f5", pad=10)

        n = len(events)
        if n:
            xs = [e["x"] for e in events]
            ys = [e["y"] for e in events]
            avg_x = sum(xs) / n
            avg_y = sum(ys) / n

            pitch.scatter(xs, ys, ax=ax, s=22, color=color, alpha=0.35, linewidths=0, zorder=2)
            pitch.lines(avg_x, 0, avg_x, 100, ax=ax, color="white", lw=1.4,
                        linestyle=(0, (5, 4)), alpha=0.7, zorder=3, comet=False)
            pitch.lines(0, avg_y, 100, avg_y, ax=ax, color="white", lw=1.4,
                        linestyle=(0, (5, 4)), alpha=0.7, zorder=3, comet=False)
            pitch.scatter(avg_x, avg_y, ax=ax, s=340, marker="*", color=color,
                          edgecolors="white", linewidths=1.3, zorder=5)

            fig.text(col_x[c] + width / 2, row_y[r] - 0.014,
                      f"{line_label}: x={avg_x:.0f}, y={avg_y:.0f}  ·  n={n}",
                      fontsize=9.5, ha="center", color="#c7ccd4")
        else:
            ax.text(50, 50, "No data", fontsize=11, color="#6b7684", ha="center", va="center")
            fig.text(col_x[c] + width / 2, row_y[r] - 0.014, f"{line_label}: n=0",
                      fontsize=9.5, ha="center", color="#6b7684")

    fig.text(0.02, 0.012, "Data via Opta | Ecuador 2026 event data", fontsize=8.5, color="#7b8794")
    fig.text(0.985, 0.012,
             "Defensive actions = tackles, interceptions, clearances, aerial duels. "
             "Receptions are a heuristic (next same-team touch after a completed pass). "
             "Shots exclude penalties.",
             fontsize=7.3, color="#6b7684", ha="right")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"def={len(defensive)} pass={len(passing)} recv={len(receiving)} shots={len(shooting)}")


if __name__ == "__main__":
    player = sys.argv[1] if len(sys.argv) > 1 else "E. Mero"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/average_position_{player.replace(' ', '_')}.png"
    defensive, passing, receiving, shooting, team = collect(player)
    if not any([defensive, passing, receiving, shooting]):
        print(f"No data found for player '{player}'")
        sys.exit(1)
    make_plot(player, defensive, passing, receiving, shooting, team, out)
