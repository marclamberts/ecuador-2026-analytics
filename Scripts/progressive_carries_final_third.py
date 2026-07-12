"""
Team map of progressive carries that bring the ball into the final
third. A carry is a TakeOn (typeId 3) through to that same player's next
touch. Progressive uses the standard geometric definition (dx>=10, or
crossing into the middle/final third from further back), and here is
additionally required to land in the final third (x>=66.7).

Usage: python3 progressive_carries_final_third.py "Independiente del Valle" [out.png]
"""
import glob
import json
import re
import sys
import collections

import matplotlib.pyplot as plt
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
ZONE_SHADE = "#1e3a5f"

FINAL_THIRD_START = 200 / 3

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


def is_progressive(x0, y0, x1, y1):
    dx = x1 - x0
    return dx >= 10 or (x0 < 50 and x1 >= 66.7) or (x0 < 66.7 and x1 >= 83)


def collect(files, cid):
    """A carry = a TakeOn (typeId 3) through to that same player's next
    touch. Kept if progressive and the end point lands in the final third."""
    events = []
    player_counts = collections.Counter()
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        evs = [e for e in data["event"] if e.get("periodId") in (1, 2) and e.get("playerId")]
        evs.sort(key=lambda e: (e.get("periodId", 0), minute_value(e), e.get("eventId", 0)))

        by_player = collections.defaultdict(list)
        for idx, e in enumerate(evs):
            by_player[e["playerId"]].append(idx)

        for i, e in enumerate(evs):
            if e["typeId"] != 3 or e.get("contestantId") != cid:
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

            if x1 >= FINAL_THIRD_START and is_progressive(x0, y0, x1, y1):
                events.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
                player_counts[e.get("playerName", "?")] += 1
    return events, player_counts


def make_plot(team_name, events, player_counts, out_path):
    n_total = len(events)

    fig = plt.figure(figsize=(11, 13.5))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.965, clean_name(team_name), fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.935, "Progressive Carries Into the Final Third  ·  Ecuador 2026  ·  Season",
             fontsize=12, ha="center", color="#9aa4b2")

    pitch_ax = fig.add_axes([0.04, 0.08, 0.92, 0.82])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                           linewidth=1.1, half=False)
    pitch.draw(ax=pitch_ax)

    pitch.polygon([[[FINAL_THIRD_START, 0], [FINAL_THIRD_START, 100], [100, 100], [100, 0]]],
                  ax=pitch_ax, facecolor=ZONE_SHADE, edgecolor="none", alpha=0.4, zorder=0.5)
    pitch.lines(FINAL_THIRD_START, 0, FINAL_THIRD_START, 100, ax=pitch_ax, color="#3a4658",
               lw=1.2, linestyle=(0, (5, 4)), alpha=0.8, zorder=1, comet=False)

    for e in events:
        pitch.lines(e["x0"], e["y0"], e["x1"], e["y1"], ax=pitch_ax, color=C_AMBER,
                    lw=2.0, alpha=0.8, zorder=2, comet=True)
        pitch.scatter(e["x0"], e["y0"], ax=pitch_ax, s=20, color=C_AMBER,
                      alpha=0.9, zorder=3, linewidths=0)
        pitch.scatter(e["x1"], e["y1"], ax=pitch_ax, s=34, facecolor="none",
                      edgecolors=C_AMBER, linewidths=1.3, alpha=0.9, zorder=3)

    caption = f"{n_total} progressive carries into the final third this season"
    fig.text(0.5, 0.045, caption, fontsize=12, ha="center", color="#c7ccd4")

    if player_counts:
        top = player_counts.most_common(5)
        top_str = "  ·  ".join(f"{name} ({n})" for name, n in top)
        fig.text(0.5, 0.022, f"Most carries: {top_str}", fontsize=9, ha="center", color="#6b7684")

    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.006, "Data via Opta | Ecuador 2026 event data · carry = take-on through to that "
             "player's next touch · progressive = dx>=10 or crosses into a new attacking third",
             fontsize=7.3, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"n={n_total}")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/progressive_carries_final_third_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    events, player_counts = collect(files, cid)
    make_plot(match, events, player_counts, out)
