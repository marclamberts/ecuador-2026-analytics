"""
Key Pass Map for the Ecuador 2026 dataset.

Usage: python3 key_pass_map.py "E. Mero"
"""
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

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
BG = "#0d1117"

SET_PIECE_QIDS = {5, 6, 107}
CROSS_QID, THROUGH_BALL_QID, CUTBACK_QID = 2, 4, 195


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


# ---------------------------------------------------------- minutes played --

def clock_to_elapsed(period_id, time_min, p1_len, p2_len):
    if period_id == 1:
        return min(time_min, p1_len)
    if period_id == 2:
        return p1_len + max(0, time_min - 45)
    return p1_len + p2_len


def player_minutes(player_name, files):
    total = 0.0
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        events = data["event"]
        periods = data["matchDetails"]["period"]
        p1_len = next((p["lengthMin"] for p in periods if p["id"] == 1), 45)
        p2_len = next((p["lengthMin"] for p in periods if p["id"] == 2), 45)
        match_total = p1_len + p2_len

        pid = None
        for e in events:
            if e.get("playerName") == player_name:
                pid = e.get("playerId")
                break
        if pid is None:
            continue

        on_time, off_time = None, None
        for e in events:
            if e["typeId"] == 34:
                qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
                ids = (qmap.get(30) or "").split(", ")
                codes = (qmap.get(44) or "").split(", ")
                for i2, c2 in zip(ids, codes):
                    if i2 == pid and c2 in ("1", "2", "3", "4"):
                        on_time = 0.0
            if e["typeId"] == 19 and e.get("playerId") == pid:
                on_time = clock_to_elapsed(e["periodId"], e["timeMin"], p1_len, p2_len)
            if e["typeId"] == 18 and e.get("playerId") == pid:
                off_time = clock_to_elapsed(e["periodId"], e["timeMin"], p1_len, p2_len)
        if on_time is None:
            continue
        total += max(0.0, (off_time if off_time is not None else match_total) - on_time)
    return total


# ------------------------------------------------------------- extraction --

def collect_key_passes(player_name):
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    cid_to_team = build_team_map(files)
    key_passes = []
    team = "Unknown"

    for fn in files:
        fn_base = fn.split("/")[-1]
        csv_path = os.path.join(DANGER_DIR, fn_base[:-5] + "_danger_models.csv")
        shot_xg = {}
        if os.path.exists(csv_path):
            import csv as csvmod
            with csv_path and open(csv_path, newline="") as f:
                for row in csvmod.DictReader(f):
                    shot_xg[int(row["event_id"])] = float(row["xg"])

        with open(fn) as f:
            data = json.load(f)
        events = data["event"]
        by_local_id = {e["eventId"]: e for e in events}

        assist_pass_xa = {}
        for e in events:
            if e["typeId"] not in (13, 14, 15, 16):
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            related = qmap.get(55)
            if related is None:
                continue
            try:
                related_local_id = int(related)
            except ValueError:
                continue
            xg = shot_xg.get(e["id"])
            if xg is not None:
                assist_pass_xa[related_local_id] = xg

        for e in events:
            if e.get("playerName") != player_name or e["typeId"] != 1:
                continue
            if not (e.get("keyPass") == 1 or e.get("assist") == 1):
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            qids = set(qmap.keys())
            team = cid_to_team.get(e["contestantId"], team)
            ex = float(qmap.get(140, e["x"]))
            ey = float(qmap.get(141, e["y"]))
            key_passes.append({
                "x0": e["x"], "y0": e["y"], "x1": ex, "y1": ey,
                "is_assist": e.get("assist") == 1,
                "set_piece": bool(SET_PIECE_QIDS & qids),
                "corner": 6 in qids, "free_kick": 5 in qids,
                "cross": CROSS_QID in qids, "through_ball": THROUGH_BALL_QID in qids,
                "cutback": CUTBACK_QID in qids,
                "xa": assist_pass_xa.get(e["eventId"], 0.04),
            })

    return key_passes, team, files


# ------------------------------------------------------------------ plot --

def make_plot(player_name, key_passes, team, nineties, out_path):
    n_total = len(key_passes)
    n_assists = sum(1 for k in key_passes if k["is_assist"])
    n_open = sum(1 for k in key_passes if not k["set_piece"])
    n_set = sum(1 for k in key_passes if k["set_piece"])
    n_through = sum(1 for k in key_passes if k["through_ball"])
    n_cross = sum(1 for k in key_passes if k["cross"])
    n_cutback = sum(1 for k in key_passes if k["cutback"])
    n_corner = sum(1 for k in key_passes if k["corner"])
    n_fk = sum(1 for k in key_passes if k["free_kick"])
    total_xa = sum(k["xa"] for k in key_passes)
    p90 = lambda c: c / nineties if nineties else 0

    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor(BG)

    pitch_ax = fig.add_axes([0.02, 0.20, 0.60, 0.66])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color="#3a4a5c",
                           linewidth=1.2, half=False)
    pitch.draw(ax=pitch_ax)

    for k in key_passes:
        if k["is_assist"]:
            color = C_AMBER
        elif k["set_piece"]:
            color = C_NAVY
        else:
            color = C_PINK
        size = 60 + k["xa"] * 900
        pitch.arrows(k["x0"], k["y0"], k["x1"], k["y1"], ax=pitch_ax, color=color,
                    width=1.3, headwidth=4, headlength=4, alpha=0.75, zorder=2)
        pitch.scatter(k["x0"], k["y0"], ax=pitch_ax, s=size, facecolor="none",
                      edgecolors=color, linewidths=1.8, zorder=3, alpha=0.9)

    legend_elems = [
        Line2D([0], [0], marker="o", color=C_AMBER, markerfacecolor="none", markersize=10,
               markeredgewidth=1.8, label="Assist", linewidth=1.5),
        Line2D([0], [0], marker="o", color=C_PINK, markerfacecolor="none", markersize=10,
               markeredgewidth=1.8, label="Open play KP", linewidth=1.5),
        Line2D([0], [0], marker="o", color=C_NAVY, markerfacecolor="none", markersize=10,
               markeredgewidth=1.8, label="Set play KP", linewidth=1.5),
    ]
    pitch_ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                     ncol=3, frameon=False, fontsize=10)
    fig.text(0.32, 0.175, "Arrow: pass start → pass end · circle size = xA", fontsize=9,
             color="#8b93a1", ha="center")

    fig.text(0.02, 0.965, player_name, fontsize=30, fontweight="bold", family="sans-serif", color="#ffffff")
    fig.text(0.02, 0.930, f"{team} · Ecuador 2026", fontsize=13, color="#9aa4b2")
    fig.text(0.02, 0.905, f"{nineties:.1f} 90s played · All Competitions", fontsize=11, color="#7b8794")
    fig.text(0.02, 0.87, "KEY PASS MAP", fontsize=13, fontweight="bold", color=C_PINK)

    # -------- right stat panel --------
    rx0, rx1 = 0.66, 0.97

    def stat_box(y, title, items):
        fig.text(rx0, y, title, fontsize=11.5, fontweight="bold", color="#f0f2f5")
        fig.add_artist(plt.Line2D([rx0, rx1], [y - 0.012, y - 0.012], transform=fig.transFigure,
                                   color="#2c3540", linewidth=1))
        xs = [rx0, rx0 + (rx1 - rx0) / 2]
        for i, (label, count, per90) in enumerate(items):
            col = xs[i % 2]
            row = i // 2
            yy = y - 0.045 - row * 0.075
            fig.text(col, yy, label, fontsize=9.5, color="#8b93a1")
            fig.text(col, yy - 0.030, f"{count} ({per90:.2f})", fontsize=17, fontweight="bold", color="#ffffff")

    stat_box(0.965, "PHASE OF PLAY", [
        ("OPEN PLAY", n_open, p90(n_open)), ("SET PLAY", n_set, p90(n_set)),
    ])
    stat_box(0.825, "KEY PASSES BY TYPE", [
        ("THROUGH BALL", n_through, p90(n_through)), ("CROSS", n_cross, p90(n_cross)),
        ("CUTBACK", n_cutback, p90(n_cutback)),
    ])
    stat_box(0.63, "SET-PIECE BREAKDOWN", [
        ("CORNER", n_corner, p90(n_corner)), ("FREE KICK", n_fk, p90(n_fk)),
    ])
    stat_box(0.49, "RESULT", [
        ("ASSISTS", n_assists, p90(n_assists)),
    ])

    # -------- bottom stat row --------
    fig_w, fig_h = fig.get_size_inches()
    aspect = fig_w / fig_h

    def bubble(x, label, value, color):
        fig.text(x, 0.135, label, fontsize=11, ha="center", color="#d7dbe0", fontweight="bold")
        r = 0.034
        ell = matplotlib.patches.Ellipse((x, 0.075), width=2 * r, height=2 * r * aspect,
                                          transform=fig.transFigure, facecolor=color,
                                          edgecolor="none", zorder=5)
        fig.add_artist(ell)
        fig.text(x, 0.075, str(value), fontsize=16, fontweight="bold", ha="center", va="center",
                 color="white", zorder=6)

    bubble(0.075, "Total KP", n_total, "#4a5568")
    bubble(0.19, "Open Play", n_open, C_PINK)
    bubble(0.31, "Assists", n_assists, C_AMBER)

    open_pct = n_open / n_total * 100 if n_total else 0
    assist_rate = n_assists / n_total * 100 if n_total else 0
    for x, label, val in [(0.435, "xA", f"{total_xa:.2f}"), (0.55, "KP / 90", f"{p90(n_total):.2f}"),
                           (0.65, "Set Play", f"{n_set}")]:
        fig.text(x, 0.135, label, fontsize=11, ha="center", color="#d7dbe0", fontweight="bold")
        fig.text(x, 0.075, val, fontsize=17, fontweight="bold", ha="center", va="center", color="#ffffff")
    for x, label, val in [(0.77, "Open Play %", f"{open_pct:.1f}%"), (0.885, "Assist Rate", f"{assist_rate:.1f}%")]:
        fig.text(x, 0.135, label, fontsize=10.5, ha="center", color="#d7dbe0", fontweight="bold")
        fig.text(x, 0.075, val, fontsize=15, fontweight="bold", ha="center", va="center", color="#ffffff")

    fig.text(0.02, 0.015, "Data via Opta | Ecuador 2026 event data", fontsize=8.5, color="#7b8794")
    fig.text(0.985, 0.015, "xA = xG of the resulting shot (proxy)", fontsize=8, color="#6b7684", ha="right")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"KP={n_total} assists={n_assists} open={n_open} set={n_set} "
          f"through={n_through} cross={n_cross} cutback={n_cutback} xA={total_xa:.2f}")


if __name__ == "__main__":
    player = sys.argv[1] if len(sys.argv) > 1 else "E. Mero"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/key_pass_map_{player.replace(' ', '_')}.png"
    key_passes, team, files = collect_key_passes(player)
    nineties = player_minutes(player, files) / 90
    if not key_passes:
        print(f"No key passes found for player '{player}'")
        sys.exit(1)
    make_plot(player, key_passes, team, nineties, out)
