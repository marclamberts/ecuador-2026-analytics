"""
Team map of line-breaking passes: open-play passes that start in front of
and end beyond the opposition's average defensive-line height for that
match. The opposition's defensive line is computed per match from their
own TeamSetup position codes (players tagged DEF, code 2), using the
average x of their touches during the match, then mirrored (100 - x) into
the attacking team's own coordinate frame -- Opta normalises every event
so the team on the ball always attacks toward x=100, so an opponent's own
defensive x (low, close to their own goal) becomes high when viewed from
the attacking team's side of the pitch.

Usage: python3 linebreaking_passes.py "Independiente del Valle" [out.png]
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
MIN_DIST_M = 30  # minimum real pass distance, to exclude trivial line-nudging passes
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


def dist_m(x0, y0, x1, y1):
    dx = (x1 - x0) / 100 * 105.0
    dy = (y1 - y0) / 100 * 68.0
    return (dx ** 2 + dy ** 2) ** 0.5


def opponent_defensive_line(data, opp_cid):
    """Average x (opponent's own attacking frame) of their DEF-coded
    players' touches during the match, or None if undetermined."""
    def_players = set()
    for e in data["event"]:
        if e["typeId"] != 34 or e.get("contestantId") != opp_cid:
            continue
        qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
        ids = (qmap.get(30) or "").split(", ")
        codes = (qmap.get(44) or "").split(", ")
        for pid, code in zip(ids, codes):
            if code == "2":
                def_players.add(pid)
    if not def_players:
        return None
    xs = [e["x"] for e in data["event"]
          if e.get("playerId") in def_players and e.get("periodId") in (1, 2) and e.get("x") is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def collect(files, cid):
    events = []
    player_counts = collections.Counter()
    line_heights = []

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        cids = set(e["contestantId"] for e in data["event"] if "contestantId" in e)
        if cid not in cids or len(cids) != 2:
            continue
        opp_cid = next(iter(cids - {cid}))

        opp_avg_own_frame = opponent_defensive_line(data, opp_cid)
        if opp_avg_own_frame is None:
            continue
        line_x = 100 - opp_avg_own_frame
        line_heights.append(line_x)

        for e in data["event"]:
            if e["typeId"] != 1 or e.get("contestantId") != cid:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            if SET_PIECE_QIDS & set(qmap.keys()):
                continue
            x0, y0 = float(e["x"]), float(e["y"])
            if x0 >= line_x:
                continue
            x1, y1 = pass_end_xy(e, qmap)
            if x1 < line_x or dist_m(x0, y0, x1, y1) < MIN_DIST_M:
                continue
            success = e["outcome"] == 1
            events.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                           "success": success, "line_x": line_x})
            if success:
                player_counts[e.get("playerName", "?")] += 1

    avg_line = sum(line_heights) / len(line_heights) if line_heights else None
    return events, player_counts, avg_line


def make_plot(team_name, events, player_counts, avg_line, out_path):
    n_total = len(events)
    n_success = sum(1 for e in events if e["success"])
    pct = n_success / n_total * 100 if n_total else 0

    fig = plt.figure(figsize=(11, 13.5))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.965, clean_name(team_name), fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.935, f"Line-Breaking Passes  ·  vs. Opposition Average Defensive Line  ·  Ecuador 2026  ·  "
             f"min. {MIN_DIST_M}m",
             fontsize=12, ha="center", color="#9aa4b2")

    pitch_ax = fig.add_axes([0.04, 0.08, 0.92, 0.82])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                           linewidth=1.1, half=False)
    pitch.draw(ax=pitch_ax)

    if avg_line is not None:
        # pitch.polygon verts are (x, y) with x = pitch length, so the
        # length-axis coordinate must be the FIRST element of each vertex pair.
        pitch.polygon([[[avg_line, 0], [avg_line, 100], [100, 100], [100, 0]]],
                      ax=pitch_ax, facecolor=ZONE_SHADE, edgecolor="none", alpha=0.4, zorder=0.5)
        pitch.lines(avg_line, 0, avg_line, 100, ax=pitch_ax, color=C_PINK, lw=1.6,
                    linestyle=(0, (5, 4)), alpha=0.9, zorder=1, comet=False)
        pitch.annotate(f"AVG. OPPOSITION DEFENSIVE LINE ({avg_line:.0f})", xy=(avg_line, 3),
                       ax=pitch_ax, ha="left", va="center", fontsize=8, fontweight="bold",
                       color=C_PINK, zorder=1, xytext=(avg_line + 1.2, 3), rotation=90)

    # only draw completed line-breaks: an incomplete attempt didn't actually
    # beat the line, it's counted in the caption stats but not plotted
    for e in events:
        if not e["success"]:
            continue
        pitch.lines(e["x0"], e["y0"], e["x1"], e["y1"], ax=pitch_ax, color=C_AMBER,
                    lw=1.3, alpha=0.75, zorder=2, comet=False)
        pitch.scatter(e["x0"], e["y0"], ax=pitch_ax, s=14, color=C_AMBER,
                      alpha=0.85, zorder=3, linewidths=0)
        pitch.scatter(e["x1"], e["y1"], ax=pitch_ax, s=26, facecolor="none",
                      edgecolors=C_AMBER, linewidths=1.1, alpha=0.85, zorder=3)

    legend_elems = [
        Line2D([0], [0], color=C_AMBER, linewidth=2, label="Completed line-break"),
        Line2D([0], [0], color=C_PINK, linewidth=1.6, linestyle=(0, (5, 4)), label="Avg. opp. line"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.885),
               ncol=2, frameon=False, fontsize=9.5, labelcolor="#c7ccd4")

    caption = f"{n_total} line-breaking attempts · {n_success} completed ({pct:.0f}%) · shown: completed only"
    fig.text(0.5, 0.045, caption, fontsize=12, ha="center", color="#c7ccd4")

    if player_counts:
        top = player_counts.most_common(5)
        top_str = "  ·  ".join(f"{name} ({n})" for name, n in top)
        fig.text(0.5, 0.022, f"Top completers: {top_str}", fontsize=9, ha="center", color="#6b7684")

    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.006, "Data via Opta | Ecuador 2026 event data · line computed per match from "
             "opposition DEF-coded players' avg. touch position, excludes corners/free kicks/throw-ins",
             fontsize=7.3, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"n={n_total} success={n_success} pct={pct:.1f}% avg_line={avg_line}")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/linebreaking_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    events, player_counts, avg_line = collect(files, cid)
    make_plot(match, events, player_counts, avg_line, out)
