"""
Box entries via pass vs via carry: two panels comparing how the team gets
the ball into the opponent's penalty area. A pass entry is a completed
open-play pass starting outside the box and ending inside it. A carry
entry is a TakeOn (typeId 3) through to that player's next touch, starting
outside the box and ending inside it.

Usage: python3 box_entries_pass_vs_carry.py "Independiente del Valle" [out.png]
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
BOX_SHADE = "#1e3a5f"

SET_PIECE_QIDS = {5, 6, 107}
BOX_X = 83.0
BOX_Y_LO, BOX_Y_HI = 21.1, 78.9

PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name):
    return PREFIX_RE.sub("", name)


def add_logo(fig, width=0.15, margin=0.014):
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


def in_box(x, y):
    return x >= BOX_X and BOX_Y_LO <= y <= BOX_Y_HI


def pass_end_xy(e, qmap):
    return float(qmap.get(140, e["x"])), float(qmap.get(141, e["y"]))


def collect_pass_entries(files, cid):
    events = []
    player_counts = collections.Counter()
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e["typeId"] != 1 or e.get("contestantId") != cid or e["outcome"] != 1:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            if SET_PIECE_QIDS & set(qmap.keys()):
                continue
            x0, y0 = float(e["x"]), float(e["y"])
            if in_box(x0, y0):
                continue
            x1, y1 = pass_end_xy(e, qmap)
            if in_box(x1, y1):
                events.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
                player_counts[e.get("playerName", "?")] += 1
    return events, player_counts


def collect_carry_entries(files, cid):
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
            if in_box(x0, y0):
                continue

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

            if in_box(x1, y1):
                events.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
                player_counts[e.get("playerName", "?")] += 1
    return events, player_counts


def draw_panel(ax, events, color):
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                           linewidth=1.1, half=False)
    pitch.draw(ax=ax)
    pitch.polygon([[[BOX_X, BOX_Y_LO], [BOX_X, BOX_Y_HI], [100, BOX_Y_HI], [100, BOX_Y_LO]]],
                  ax=ax, facecolor=BOX_SHADE, edgecolor=C_CORAL, linewidth=1.0,
                  alpha=0.4, zorder=0.6)
    for e in events:
        pitch.lines(e["x0"], e["y0"], e["x1"], e["y1"], ax=ax, color=color,
                    lw=1.4, alpha=0.75, zorder=2, comet=True)
        pitch.scatter(e["x1"], e["y1"], ax=ax, s=26, color=color,
                      edgecolors="white", linewidths=0.6, alpha=0.9, zorder=3)


def make_plot(team_name, pass_events, pass_counts, carry_events, carry_counts, out_path):
    n_pass, n_carry = len(pass_events), len(carry_events)
    total = n_pass + n_carry
    pct_pass = n_pass / total * 100 if total else 0
    pct_carry = n_carry / total * 100 if total else 0

    fig = plt.figure(figsize=(17, 12.2))
    fig.patch.set_facecolor(BG)

    fig.text(0.03, 0.955, clean_name(team_name), fontsize=27, fontweight="bold", color="white")
    fig.text(0.03, 0.918, "Box Entries: Pass vs Carry  ·  Ecuador 2026  ·  Season",
             fontsize=13.5, fontweight="bold", color="#c7ccd4")
    fig.text(0.03, 0.892, f"{total} total box entries this season  ·  excludes corners/free kicks/throw-ins",
             fontsize=10, color="#9aa4b2")

    axes = [fig.add_axes([0.02 + i * 0.49, 0.05, 0.46, 0.80]) for i in range(2)]
    draw_panel(axes[0], pass_events, C_AMBER)
    draw_panel(axes[1], carry_events, C_PINK)

    for ax, label, n, pct in zip(axes, ["PASS", "CARRY"], [n_pass, n_carry], [pct_pass, pct_carry]):
        x_center = ax.get_position().x0 + ax.get_position().width / 2
        fig.text(x_center, 0.855, f"{label}  ({n} · {pct:.0f}%)", fontsize=15, fontweight="bold",
                 ha="center", color="#e6e9ee", transform=fig.transFigure)

    if pass_counts:
        top = pass_counts.most_common(3)
        top_str = "  ·  ".join(f"{name} ({n})" for name, n in top)
        fig.text(axes[0].get_position().x0 + axes[0].get_position().width / 2, 0.028,
                 f"Top: {top_str}", fontsize=9, ha="center", color="#6b7684", transform=fig.transFigure)
    if carry_counts:
        top = carry_counts.most_common(3)
        top_str = "  ·  ".join(f"{name} ({n})" for name, n in top)
        fig.text(axes[1].get_position().x0 + axes[1].get_position().width / 2, 0.028,
                 f"Top: {top_str}", fontsize=9, ha="center", color="#6b7684", transform=fig.transFigure)

    fig.text(0.98, 0.008, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.008, "Data via Opta | Ecuador 2026 event data · carry = take-on through to that "
             "player's next touch", fontsize=8, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"pass={n_pass} carry={n_carry} pct_pass={pct_pass:.1f}% pct_carry={pct_carry:.1f}%")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/box_entries_pass_vs_carry_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    pass_events, pass_counts = collect_pass_entries(files, cid)
    carry_events, carry_counts = collect_carry_entries(files, cid)
    make_plot(match, pass_events, pass_counts, carry_events, carry_counts, out)
