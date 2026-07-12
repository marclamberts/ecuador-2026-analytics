"""
Set-piece second balls: after this team's own corners, free kicks, and
long throw-ins delivered into the final third, who wins the loose ball
that follows the first contact?

Method: for each qualifying delivery, look at the next events (within a
~12s window) and find contested actions (aerial duel, tackle,
interception, clearance, ball recovery). The SECOND such contested
action is treated as "the second ball" -- the first is usually just the
target player's initial header/contact, the second is who actually comes
away with the loose ball. Plotted wherever that second contest happened,
which is very often well outside the box since a strong clearance can
travel 40+ metres before the second ball is won.

Usage: python3 set_piece_second_balls.py "Independiente del Valle" [out.png]
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

CORNER_QID = 6
FREEKICK_QID = 5
THROWIN_QID = 107
CONTESTED_TYPES = {44, 7, 8, 12, 49}  # Aerial, Tackle, Interception, Clearance, BallRecovery
FINAL_THIRD_START = 200 / 3
LONG_THROW_MIN_M = 25
WINDOW_EVENTS = 5
WINDOW_MIN = 12 / 60
MAX_CLEAN_PASSES_BETWEEN = 1

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


def dist_m(x0, y0, x1, y1):
    dx = (x1 - x0) / 100 * 105.0
    dy = (y1 - y0) / 100 * 68.0
    return (dx ** 2 + dy ** 2) ** 0.5


def collect(files, cid):
    events = []
    player_counts = collections.Counter()
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        evs = [e for e in data["event"] if e.get("periodId") in (1, 2)]
        evs.sort(key=lambda e: (e["periodId"], minute_value(e), e.get("eventId", 0)))

        for i, e in enumerate(evs):
            if e["typeId"] != 1 or e.get("contestantId") != cid:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            qids = set(qmap.keys())
            x0, y0 = float(e["x"]), float(e["y"])
            x1 = float(qmap.get(140, x0))
            y1 = float(qmap.get(141, y0))
            if x1 < FINAL_THIRD_START:
                continue
            is_corner = CORNER_QID in qids
            is_fk = FREEKICK_QID in qids
            is_long_throw = THROWIN_QID in qids and dist_m(x0, y0, x1, y1) >= LONG_THROW_MIN_M
            if not (is_corner or is_fk or is_long_throw):
                continue

            t0 = minute_value(e)
            window = [ev for ev in evs[i + 1:i + 1 + WINDOW_EVENTS]
                      if minute_value(ev) - t0 <= WINDOW_MIN]
            first_idx = next((j for j, ev in enumerate(window) if ev["typeId"] in CONTESTED_TYPES), None)
            if first_idx is None:
                continue
            # second contested action, allowing at most one clean pass in
            # between (e.g. a knockdown into a header) -- if the ball
            # strings together more than that, possession has settled and
            # any later contest is a separate, unrelated phase of play
            sb, clean_passes = None, 0
            for ev in window[first_idx + 1:]:
                if ev["typeId"] in CONTESTED_TYPES:
                    sb = ev
                    break
                if ev["typeId"] == 1 and ev["outcome"] == 1:
                    clean_passes += 1
                    if clean_passes > MAX_CLEAN_PASSES_BETWEEN:
                        break
            if sb is None or sb.get("x") is None or sb.get("y") is None:
                continue
            won = sb.get("contestantId") == cid
            events.append({"x": float(sb["x"]), "y": float(sb["y"]), "won": won})
            if won:
                player_counts[sb.get("playerName", "?")] += 1
    return events, player_counts


def make_plot(team_name, events, player_counts, out_path):
    n_total = len(events)
    n_won = sum(1 for e in events if e["won"])
    pct = n_won / n_total * 100 if n_total else 0

    fig = plt.figure(figsize=(11, 13.5))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.965, clean_name(team_name), fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.935, "Set-Piece Second Balls  ·  Own Corners, Free Kicks & Long Throws  ·  Ecuador 2026",
             fontsize=11.5, ha="center", color="#9aa4b2")
    fig.text(0.5, 0.912, "Second contest after the delivery -- often well outside the box once "
             "the first header is cleared", fontsize=8.8, ha="center", color="#6b7684")

    pitch_ax = fig.add_axes([0.04, 0.08, 0.92, 0.79])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                           linewidth=1.1, half=False)
    pitch.draw(ax=pitch_ax)

    for e in sorted(events, key=lambda e: e["won"]):
        color = C_AMBER if e["won"] else LINE_COLOR
        alpha = 0.85 if e["won"] else 0.4
        pitch.scatter(e["x"], e["y"], ax=pitch_ax, s=90, color=color,
                      edgecolors=BG, linewidths=1.2, alpha=alpha, zorder=3)

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_AMBER, markersize=10,
               label="Won by this team"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LINE_COLOR, markersize=10,
               alpha=0.6, label="Won by opposition"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.885),
               ncol=2, frameon=False, fontsize=10, labelcolor="#c7ccd4")

    caption = f"{n_total} identifiable second-ball contests · won {n_won} ({pct:.0f}%)"
    fig.text(0.5, 0.045, caption, fontsize=12, ha="center", color="#c7ccd4")

    if player_counts:
        top = player_counts.most_common(5)
        top_str = "  ·  ".join(f"{name} ({n})" for name, n in top)
        fig.text(0.5, 0.022, f"Top second-ball winners: {top_str}", fontsize=9, ha="center", color="#6b7684")

    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.006, "Data via Opta | Ecuador 2026 event data · deliveries landing in the final "
             "third only, second contested action within ~12s of the delivery, max 1 clean pass between",
             fontsize=7.2, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"n={n_total} won={n_won} pct={pct:.1f}%")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/set_piece_second_balls_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    events, player_counts = collect(files, cid)
    make_plot(match, events, player_counts, out)
