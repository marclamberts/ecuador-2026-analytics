"""
Passing, Receiving, and Ball Winning Actions chart for the Ecuador 2026 dataset.

Usage: python3 passing_profile.py "E. Mero"
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

SET_PIECE_QIDS = {5, 6, 107}  # free kick taken, corner taken, throw-in


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


def dist_to_goal_m(x, y):
    dx = (100 - x) / 100 * 105.0
    dy = (y - 50) / 100 * 68.0
    return (dx ** 2 + dy ** 2) ** 0.5


def is_progressive(x0, y0, x1, y1):
    d0, d1 = dist_to_goal_m(x0, y0), dist_to_goal_m(x1, y1)
    if d1 >= d0:
        return False
    reduction = (d0 - d1) / d0
    if x0 < 50 and x1 < 50:
        return reduction >= 0.30
    elif x0 < 50 <= x1:
        return reduction >= 0.15
    return reduction >= 0.10


def collect(player_name):
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    cid_to_team = build_team_map(files)
    passes = []          # this player's attempted passes
    receptions = []       # locations where this player received a completed pass
    defensive = []        # defensive actions by this player
    team = "Unknown"

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        events = data["event"]
        for i, e in enumerate(events):
            name = e.get("playerName")
            tid = e["typeId"]

            if name == player_name and tid == 1:
                qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
                team = cid_to_team.get(e["contestantId"], team)
                is_set_piece = bool(SET_PIECE_QIDS & set(qmap.keys()))
                ex = float(qmap.get(140, e["x"]))
                ey = float(qmap.get(141, e["y"]))
                passes.append({
                    "x0": e["x"], "y0": e["y"], "x1": ex, "y1": ey,
                    "success": e["outcome"] == 1,
                    "set_piece": is_set_piece,
                    "key_pass": e.get("keyPass") == 1,
                    "progressive": is_progressive(e["x"], e["y"], ex, ey),
                })

            if name == player_name and tid in (4, 7, 8, 12, 49):
                defensive.append({"x": e["x"], "y": e["y"], "typeId": tid, "outcome": e["outcome"]})

            # reception heuristic: successful pass by a teammate, followed by
            # this player's next action for the same team
            if tid == 1 and e.get("outcome") == 1 and name != player_name:
                if i + 1 < len(events):
                    nxt = events[i + 1]
                    if (nxt.get("playerName") == player_name
                            and nxt.get("contestantId") == e.get("contestantId")):
                        qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
                        ex = float(qmap.get(140, e["x"]))
                        ey = float(qmap.get(141, e["y"]))
                        is_set_piece = bool(SET_PIECE_QIDS & set(qmap.keys()))
                        receptions.append({
                            "x": ex, "y": ey,
                            "progressive": is_progressive(e["x"], e["y"], ex, ey),
                            "set_piece": is_set_piece,
                        })

    return passes, receptions, defensive, team


def make_plot(player_name, passes, receptions, defensive, team, out_path):
    open_passes = [p for p in passes if not p["set_piece"]]
    completed = [p for p in open_passes if p["success"]]
    n_att = len(open_passes)
    n_comp = len(completed)
    comp_pct = n_comp / n_att * 100 if n_att else 0
    n_prog = sum(1 for p in completed if p["progressive"])
    n_key = sum(1 for p in open_passes if p["key_pass"])

    open_recv = [r for r in receptions if not r["set_piece"]]
    n_recv = len(open_recv)
    n_prog_recv = sum(1 for r in open_recv if r["progressive"])
    n_deep_touch = sum(1 for p in passes if p["x0"] >= 83) + sum(1 for r in open_recv if r["x"] >= 83)

    n_int = sum(1 for d in defensive if d["typeId"] == 8)
    tackles = [d for d in defensive if d["typeId"] == 7]
    n_tackle = len(tackles)
    n_tackle_succ = sum(1 for d in tackles if d["outcome"] == 1)
    n_foul = sum(1 for d in defensive if d["typeId"] == 4 and d["outcome"] == 0)
    n_recov = sum(1 for d in defensive if d["typeId"] == 49)
    n_clear = sum(1 for d in defensive if d["typeId"] == 12)

    fig = plt.figure(figsize=(15, 9.2))
    fig.patch.set_facecolor(BG)

    axs = [fig.add_axes([0.02 + i * 0.327, 0.185, 0.30, 0.66]) for i in range(3)]
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color="#3a4a5c",
                           linewidth=1.1, half=False)
    for ax in axs:
        pitch.draw(ax=ax)

    # ---------------- panel 1: open play passes ----------------
    ax = axs[0]
    ax.set_title("OPEN PLAY PASSES", fontsize=11, fontweight="bold", color="#d7dbe0", pad=10)
    for p in open_passes:
        if p["key_pass"]:
            color, lw, z, a = C_AMBER, 1.8, 5, 0.95
        elif p["success"] and p["progressive"]:
            color, lw, z, a = C_PURPLE, 1.3, 4, 0.85
        else:
            color, lw, z, a = "#3a4452", 0.6, 2, 0.55
        pitch.lines(p["x0"], p["y0"], p["x1"], p["y1"], ax=ax, color=color,
                    lw=lw, zorder=z, alpha=a, comet=False)
        pitch.scatter(p["x0"], p["y0"], ax=ax, s=10, color=color, alpha=min(a, 0.8), zorder=z + 1, linewidths=0)

    # top-3 pass-origin hotspots
    if open_passes:
        xs = np.array([p["x0"] for p in open_passes])
        ys = np.array([p["y0"] for p in open_passes])
        xedges = np.linspace(0, 100, 11)
        yedges = np.linspace(0, 100, 9)
        hist, _, _ = np.histogram2d(xs, ys, bins=[xedges, yedges])
        flat_idx = np.argsort(hist.ravel())[::-1][:3]
        for idx in flat_idx:
            ix, iy = np.unravel_index(idx, hist.shape)
            if hist[ix, iy] == 0:
                continue
            cx = (xedges[ix] + xedges[ix + 1]) / 2
            cy = (yedges[iy] + yedges[iy + 1]) / 2
            pitch.scatter(cx, cy, ax=ax, s=260, facecolor="none",
                          edgecolors="#ffffff", linewidths=1.6, zorder=6)

    fig.text(0.02, 0.145, f"Passes: {n_att} ({comp_pct:.1f}%)", fontsize=10, color="#d7dbe0")
    fig.text(0.02, 0.108, f"Progressive Passes: {n_prog}", fontsize=10, color=C_PURPLE, fontweight="bold")
    fig.text(0.02, 0.071, f"Key Passes: {n_key}", fontsize=10, color=C_AMBER, fontweight="bold")

    # ---------------- panel 2: passes received (KDE) ----------------
    ax = axs[1]
    ax.set_title("PASSES RECEIVED", fontsize=11, fontweight="bold", color="#d7dbe0", pad=10)
    heat_cmap = LinearSegmentedColormap.from_list(
        "heat", ["#ffffff00", C_AMBER, C_CORAL, C_PINK], N=256)
    if len(open_recv) >= 5:
        pitch.kdeplot([r["x"] for r in open_recv], [r["y"] for r in open_recv],
                      ax=ax, cmap=heat_cmap, fill=True, levels=60, thresh=0.02, zorder=1)
    for r in open_recv:
        pitch.scatter(r["x"], r["y"], ax=ax, s=6, color=C_INDIGO, alpha=0.35, zorder=2, linewidths=0)

    fig.text(0.35, 0.145, f"Passes Received: {n_recv}", fontsize=10, color="#d7dbe0")
    fig.text(0.35, 0.108, f"Progressive Passes Received: {n_prog_recv}", fontsize=10, color="#d7dbe0")
    fig.text(0.35, 0.071, f"Deep Touches: {n_deep_touch}", fontsize=10, color="#d7dbe0")

    # ---------------- panel 3: defensive actions ----------------
    ax = axs[2]
    ax.set_title("DEFENSIVE ACTIONS", fontsize=11, fontweight="bold", color="#d7dbe0", pad=10)
    marker_spec = {
        8: (C_INDIGO, "D", "Interceptions"),
        7: (C_NAVY, "s", "Tackles"),
        4: (C_PINK, "x", "Fouls"),
        49: (C_PURPLE, "o", "Ball Recoveries"),
        12: (C_AMBER, "*", "Clearances"),
    }
    for d in defensive:
        if d["typeId"] == 4 and d["outcome"] != 0:
            continue
        color, marker, _ = marker_spec[d["typeId"]]
        size = 110 if marker == "*" else 55
        pitch.scatter(d["x"], d["y"], ax=ax, s=size, color=color, marker=marker,
                      linewidths=1.1 if marker in ("x", "D", "s") else 0,
                      edgecolors=color if marker not in ("x",) else None,
                      alpha=0.85, zorder=3)

    legend_y = [0.145, 0.108, 0.071]
    fig.text(0.685, legend_y[0], f"Interceptions: {n_int}   Tackles: {n_tackle} ({n_tackle_succ} successful)",
             fontsize=9.5, color="#d7dbe0")
    fig.text(0.685, legend_y[1], f"Fouls: {n_foul}   Ball Recoveries: {n_recov}",
             fontsize=9.5, color="#d7dbe0")
    fig.text(0.685, legend_y[2], f"Clearances: {n_clear}", fontsize=9.5, color="#d7dbe0")

    fig.text(0.02, 0.965, player_name, fontsize=28, fontweight="bold", family="sans-serif", color="#ffffff")
    fig.text(0.02, 0.925, "Passing, Receiving, and Ball Winning Actions", fontsize=14, fontweight="bold", color="#d7dbe0")
    fig.text(0.02, 0.895, f"{team} · Ecuador 2026 · All Competitions", fontsize=11, color="#9aa4b2")

    total_def = n_int + n_tackle + n_foul + n_recov + n_clear
    fig.text(0.985, 0.965, f"2025-2026 · {n_att} passes · {n_recv} received · {total_def} def. actions",
             fontsize=9, color="#7b8794", ha="right")
    fig.text(0.985, 0.020, "Data via Opta | Ecuador 2026 event data", fontsize=8.5, color="#7b8794", ha="right")
    fig.text(0.02, 0.020,
             "Progressive pass = reduces distance-to-goal by ≥30% (own half), "
             "≥15% (crossing halfway), ≥10% (final third). "
             "Passes received / deep touches are heuristics (next same-team event).",
             fontsize=7.3, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"Passes={n_att} comp={comp_pct:.1f}% prog={n_prog} key={n_key} "
          f"recv={n_recv} prog_recv={n_prog_recv} deep_touch={n_deep_touch} "
          f"int={n_int} tackles={n_tackle}/{n_tackle_succ} fouls={n_foul} recov={n_recov} clear={n_clear}")


if __name__ == "__main__":
    player = sys.argv[1] if len(sys.argv) > 1 else "E. Mero"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/passing_profile_{player.replace(' ', '_')}.png"
    passes, receptions, defensive, team = collect(player)
    if not passes and not defensive:
        print(f"No data found for player '{player}'")
        sys.exit(1)
    make_plot(player, passes, receptions, defensive, team, out)
