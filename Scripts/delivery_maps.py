"""
Origin & Delivery maps for the Ecuador 2026 dataset: progressive passes,
assists, second assists, through balls, crosses, dribbles, and carries.
All exclude set pieces.

Usage: python3 delivery_maps.py "E. Mero" <type> [out.png]
  <type> in: progressive, assist, second_assist, through_ball, crosses,
             dribbles, carries
"""
import glob
import json
import re
import sys
import collections

import matplotlib.pyplot as plt
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

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
BG = "#0d1117"
LINE_COLOR = "#c7ccd4"
PITCH_LINE = "#2c3a4d"

# Exclude actual set pieces (corner/free kick/throw-in) from "open play" maps.
# Through ball / cross detection uses the real Opta pass qualifiers (verified
# against this feed: qualifier 4 = through ball, 252 uses league-wide;
# qualifier 2 = cross, ~1500 uses league-wide). Progressive pass has no Opta
# qualifier -- it's an industry-standard derived stat, kept geometric.
SET_PIECE_QIDS = {5, 6, 107}
CROSS_QID = 2
THROUGH_BALL_QID = 4

TITLES = {
    "progressive": "Progressive Passes",
    "assist": "Assists",
    "second_assist": "Second Assists",
    "through_ball": "Through Balls",
    "crosses": "Crosses",
    "dribbles": "Dribbles",
    "carries": "Carries",
}

# A carry is a TakeOn (typeId 3) through to that same player's next touch:
# start point is the take-on location, end point is wherever the player's
# very next event happened.


def minute_value(e):
    return float(e.get("timeMin") or 0) + float(e.get("timeSec") or 0) / 60.0


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


def is_progressive(x0, y0, x1, y1):
    dx = x1 - x0
    return dx >= 10 or (x0 < 50 and x1 >= 66.7) or (x0 < 66.7 and x1 >= 83)


def pass_end_xy(e, qmap):
    return float(qmap.get(140, e["x"])), float(qmap.get(141, e["y"]))


def collect_carries(files, player_name, cid_to_team):
    events = []
    team = "Unknown"
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        evs = [e for e in data["event"] if e.get("periodId") in (1, 2) and e.get("playerId")]
        evs.sort(key=lambda e: (e.get("periodId", 0), minute_value(e), e.get("eventId", 0)))

        by_player = collections.defaultdict(list)
        for idx, e in enumerate(evs):
            by_player[e["playerId"]].append(idx)

        for i, e in enumerate(evs):
            if e["typeId"] != 3 or e.get("playerName") != player_name:
                continue
            x0, y0 = e.get("x"), e.get("y")
            if x0 is None or y0 is None:
                continue
            x0, y0 = float(x0), float(y0)

            idx_list = by_player[e["playerId"]]
            pos = idx_list.index(i)
            if pos + 1 >= len(idx_list):
                continue
            nxt = evs[idx_list[pos + 1]]
            if nxt.get("periodId") != e.get("periodId"):
                continue
            x1, y1 = nxt.get("x"), nxt.get("y")
            if x1 is None or y1 is None:
                continue
            x1, y1 = float(x1), float(y1)

            team = cid_to_team.get(e.get("contestantId"), team)
            events.append({
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "success": is_progressive(x0, y0, x1, y1),
            })
    return events, team


def collect(player_name, map_type):
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    cid_to_team = build_team_map(files)
    events = []
    team = "Unknown"

    if map_type == "carries":
        return collect_carries(files, player_name, cid_to_team)

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        evs = data["event"]

        if map_type == "dribbles":
            for e in evs:
                if e.get("playerName") != player_name or e["typeId"] != 3:
                    continue
                team = cid_to_team.get(e["contestantId"], team)
                events.append({
                    "x0": e["x"], "y0": e["y"], "x1": e["x"], "y1": e["y"],
                    "success": e["outcome"] == 1, "point": True,
                })
            continue

        if map_type in ("assist", "second_assist"):
            # assist = a pass whose next action (same team) is a goal.
            # second assist = a pass whose next action is a pass that is
            # itself an assist (i.e. two passes before the goal).
            for i, e in enumerate(evs):
                if e["typeId"] != 1 or e.get("playerName") != player_name or e["outcome"] != 1:
                    continue
                if e.get("contestantId") is None:
                    continue
                qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
                if SET_PIECE_QIDS & set(qmap.keys()):
                    continue

                if map_type == "assist":
                    if i + 1 >= len(evs):
                        continue
                    nxt = evs[i + 1]
                    is_hit = nxt["typeId"] == 16 and nxt.get("contestantId") == e["contestantId"]
                else:
                    if i + 2 >= len(evs):
                        continue
                    nxt1, nxt2 = evs[i + 1], evs[i + 2]
                    is_hit = (nxt1["typeId"] == 1 and nxt1.get("contestantId") == e["contestantId"]
                              and nxt2["typeId"] == 16 and nxt2.get("contestantId") == e["contestantId"])
                if not is_hit:
                    continue

                team = cid_to_team.get(e["contestantId"], team)
                ex, ey = pass_end_xy(e, qmap)
                events.append({"x0": e["x"], "y0": e["y"], "x1": ex, "y1": ey, "success": True})
            continue

        for e in evs:
            if e.get("playerName") != player_name or e["typeId"] != 1:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            qids = set(qmap.keys())
            if SET_PIECE_QIDS & qids:
                continue
            team = cid_to_team.get(e["contestantId"], team)
            ex, ey = pass_end_xy(e, qmap)
            success = e["outcome"] == 1

            if map_type == "progressive":
                if not (success and is_progressive(e["x"], e["y"], ex, ey)):
                    continue
            elif map_type == "through_ball":
                if THROUGH_BALL_QID not in qids:
                    continue
            elif map_type == "crosses":
                if CROSS_QID not in qids:
                    continue

            events.append({"x0": e["x"], "y0": e["y"], "x1": ex, "y1": ey, "success": success})

    return events, team


def make_plot(player_name, events, team, map_type, out_path):
    title = TITLES[map_type]
    n_total = len(events)
    n_success = sum(1 for e in events if e["success"])
    pct = n_success / n_total * 100 if n_total else 0

    fig = plt.figure(figsize=(11, 13.5))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.965, player_name, fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.935, f"{title}  ·  {team}  ·  Ecuador 2026  ·  Origin & Delivery",
             fontsize=12, ha="center", color="#9aa4b2")

    pitch_ax = fig.add_axes([0.04, 0.08, 0.92, 0.82])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                           linewidth=1.1, half=False)
    pitch.draw(ax=pitch_ax)

    for e in events:
        color = C_AMBER if e["success"] else LINE_COLOR
        alpha = 0.9 if e["success"] else 0.35
        if e.get("point"):
            if e["success"]:
                pitch.scatter(e["x0"], e["y0"], ax=pitch_ax, s=70, color=color,
                              marker="o", linewidths=0, alpha=alpha, zorder=3)
            else:
                pitch.scatter(e["x0"], e["y0"], ax=pitch_ax, s=70, color=color,
                              marker="x", linewidths=1.8, alpha=alpha, zorder=3)
            continue
        pitch.lines(e["x0"], e["y0"], e["x1"], e["y1"], ax=pitch_ax, color=color,
                    lw=1.4, alpha=alpha, zorder=2, comet=False)
        pitch.scatter(e["x0"], e["y0"], ax=pitch_ax, s=16, color=color,
                      alpha=alpha, zorder=3, linewidths=0)
        pitch.scatter(e["x1"], e["y1"], ax=pitch_ax, s=28, facecolor="none",
                      edgecolors=color, linewidths=1.3, alpha=alpha, zorder=3)

    from matplotlib.lines import Line2D

    if n_total == 0:
        fig.text(0.5, 0.5, "No events recorded this season", ha="center", va="center",
                 fontsize=14, color="#6b7684", style="italic")
        caption = "0 events"
    elif map_type in ("assist", "second_assist"):
        caption = f"{n_total} {title.lower()} this season"
    elif map_type == "carries":
        caption = f"{n_total} carries · {n_success} progressive"
        legend_elems = [
            Line2D([0], [0], color=C_AMBER, linewidth=2, label="Progressive"),
            Line2D([0], [0], color=LINE_COLOR, linewidth=2, alpha=0.5, label="Other"),
        ]
        fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.885),
                   ncol=2, frameon=False, fontsize=10, labelcolor="#c7ccd4")
    elif map_type == "dribbles":
        caption = f"{n_total} {title.lower()} · {pct:.0f}% completed"
        legend_elems = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=C_AMBER, markersize=9, label="Completed"),
            Line2D([0], [0], marker="x", color=LINE_COLOR, markersize=9, markeredgewidth=1.8, alpha=0.6, label="Incomplete"),
        ]
        fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.885),
                   ncol=2, frameon=False, fontsize=10, labelcolor="#c7ccd4")
    else:
        caption = f"{n_total} {title.lower()} · {pct:.0f}% completed"
        legend_elems = [
            Line2D([0], [0], color=C_AMBER, linewidth=2, label="Completed"),
            Line2D([0], [0], color=LINE_COLOR, linewidth=2, alpha=0.5, label="Incomplete"),
        ]
        fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.885),
                   ncol=2, frameon=False, fontsize=10, labelcolor="#c7ccd4")

    fig.text(0.5, 0.045, caption, fontsize=12, ha="center", color="#c7ccd4")
    fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.012, "Data via Opta | Ecuador 2026 event data", fontsize=8.5, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"{title}: n={n_total} success={n_success} pct={pct:.1f}%")


if __name__ == "__main__":
    player = sys.argv[1] if len(sys.argv) > 1 else "E. Mero"
    map_type = sys.argv[2] if len(sys.argv) > 2 else "progressive"
    out = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/{map_type}_map_{player.replace(' ', '_')}.png"
    if map_type not in TITLES:
        print(f"Unknown map type '{map_type}'. Choose from: {list(TITLES)}")
        sys.exit(1)
    events, team = collect(player, map_type)
    make_plot(player, events, team, map_type, out)
