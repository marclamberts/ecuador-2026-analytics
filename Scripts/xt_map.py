"""
Team xT (Expected Threat) map: builds a 12x8 xT value grid from the whole
Ecuador 2026 league (all 136 matches, all teams) using the standard Karun
Singh possession-value model, then plots the team's highest xT-adding
completed passes this season on top of the grid as a heatmap.

The xT grid is a Markov-chain value model: from each pitch cell, on the
next action a player either (a) shoots, valued by that cell's shot
probability times goal-conversion rate, or (b) makes a successful move
(pass), valued by the xT of the destination cell, or (c) loses the ball
(contributes 0). Solved by value iteration. This is computed from this
league's own data, not an imported/published grid.

Usage: python3 xt_map.py "Independiente del Valle" [out.png]
"""
import glob
import json
import re
import sys
import collections

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import VerticalPitch

DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
XT_CACHE = "/Users/marclamberts/Event data/Ecuador 2026/.xt_grid_cache.json"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
BG = "#0d1117"
PITCH_LINE = "#3a4658"

SET_PIECE_QIDS = {5, 6, 107}
SHOT_TYPES = {13, 14, 15, 16}
GOAL_TYPE = 16
N_COLS, N_ROWS = 12, 8  # length x width grid cells
N_ITER = 12

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


def cell_of(x, y):
    col = min(int(x / 100 * N_COLS), N_COLS - 1)
    row = min(int(y / 100 * N_ROWS), N_ROWS - 1)
    return col, row


def pass_end_xy(e, qmap):
    return float(qmap.get(140, e["x"])), float(qmap.get(141, e["y"]))


def build_xt_grid(files):
    """League-wide value-iteration xT grid. Cached to disk since it scans
    every match in the league (not just one team)."""
    try:
        with open(XT_CACHE) as f:
            cached = json.load(f)
        if cached.get("n_files") == len(files):
            return np.array(cached["grid"])
    except FileNotFoundError:
        pass

    total_actions = np.zeros((N_COLS, N_ROWS))
    shots = np.zeros((N_COLS, N_ROWS))
    goals = np.zeros((N_COLS, N_ROWS))
    moves = np.zeros((N_COLS, N_ROWS))
    transitions = np.zeros((N_COLS, N_ROWS, N_COLS, N_ROWS))

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e["typeId"] == 1 and e.get("x") is not None:
                qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
                c0 = cell_of(float(e["x"]), float(e["y"]))
                total_actions[c0] += 1
                if e["outcome"] == 1:
                    ex, ey = pass_end_xy(e, qmap)
                    c1 = cell_of(ex, ey)
                    moves[c0] += 1
                    transitions[c0[0], c0[1], c1[0], c1[1]] += 1
            elif e["typeId"] in SHOT_TYPES and e.get("x") is not None:
                c0 = cell_of(float(e["x"]), float(e["y"]))
                total_actions[c0] += 1
                shots[c0] += 1
                if e["typeId"] == GOAL_TYPE:
                    goals[c0] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        shot_prob = np.where(total_actions > 0, shots / total_actions, 0)
        move_prob = np.where(total_actions > 0, moves / total_actions, 0)
        goal_prob = np.where(shots > 0, goals / shots, 0)
        transition_prob = np.where(
            moves[:, :, None, None] > 0, transitions / moves[:, :, None, None], 0
        )

    xt = np.zeros((N_COLS, N_ROWS))
    for _ in range(N_ITER):
        move_value = np.tensordot(transition_prob, xt, axes=([2, 3], [0, 1]))
        xt = shot_prob * goal_prob + move_prob * move_value

    with open(XT_CACHE, "w") as f:
        json.dump({"n_files": len(files), "grid": xt.tolist()}, f)
    return xt


def collect_team_actions(files, cid, xt):
    events = []
    player_xt = collections.Counter()
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
            x1, y1 = pass_end_xy(e, qmap)
            c0, c1 = cell_of(x0, y0), cell_of(x1, y1)
            added = float(xt[c1] - xt[c0])
            events.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "added": added,
                           "player": e.get("playerName", "?")})
            player_xt[e.get("playerName", "?")] += added
    return events, player_xt


def make_plot(team_name, xt, events, player_xt, out_path):
    fig = plt.figure(figsize=(11, 13.7))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.965, clean_name(team_name), fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.934, "Expected Threat (xT)  ·  Highest-Value Passes  ·  Ecuador 2026  ·  Season",
             fontsize=12, ha="center", color="#9aa4b2")
    fig.text(0.5, 0.912, "Background = league-wide xT value grid (136 matches, value-iteration model)",
             fontsize=8.8, ha="center", color="#6b7684")

    pitch_ax = fig.add_axes([0.04, 0.08, 0.92, 0.80])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                           linewidth=1.0, half=False)
    pitch.draw(ax=pitch_ax)

    cmap = LinearSegmentedColormap.from_list("xt", [BG, "#1e3a5f", "#7b3fa0", "#c1447e", "#ff8a3d", C_AMBER])
    bins = pitch.bin_statistic(np.array([50.0]), np.array([50.0]), statistic="count",
                               bins=(N_COLS, N_ROWS))
    bins["statistic"] = xt.T
    pitch.heatmap(bins, ax=pitch_ax, cmap=cmap, alpha=0.85, zorder=0.5, edgecolors=BG, linewidth=0.5)

    top_events = sorted(events, key=lambda e: -e["added"])[:60]
    max_added = max((e["added"] for e in top_events), default=1) or 1
    for e in sorted(top_events, key=lambda e: e["added"]):
        w = e["added"] / max_added
        lw = 1.0 + w * 3.2
        alpha = 0.45 + w * 0.5
        pitch.lines(e["x0"], e["y0"], e["x1"], e["y1"], ax=pitch_ax, color="white",
                    lw=lw, alpha=alpha, zorder=2, comet=True)
        pitch.scatter(e["x1"], e["y1"], ax=pitch_ax, s=18 + w * 40, color="white",
                      alpha=alpha, zorder=3, linewidths=0)

    caption = f"Top {len(top_events)} highest-xT completed passes shown (of {len(events)} total, excl. set pieces)"
    fig.text(0.5, 0.045, caption, fontsize=11, ha="center", color="#c7ccd4")

    if player_xt:
        top = player_xt.most_common(5)
        top_str = "  ·  ".join(f"{name} ({v:.2f})" for name, v in top)
        fig.text(0.5, 0.022, f"Top xT added: {top_str}", fontsize=9, ha="center", color="#6b7684")

    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.006, "Data via Opta | Ecuador 2026 event data · xT grid fit league-wide, "
             "12x8 cells, value iteration (Karun Singh method)", fontsize=7.3, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"total completed passes considered: {len(events)}")
    print("grid min/max/mean:", xt.min(), xt.max(), xt.mean())


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/xt_map_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]

    xt = build_xt_grid(files)
    events, player_xt = collect_team_actions(files, cid, xt)
    make_plot(match, xt, events, player_xt, out)
