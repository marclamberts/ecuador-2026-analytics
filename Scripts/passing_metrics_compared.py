"""
Passing Metrics compared to positional peers, for the Ecuador 2026 dataset.
Uses the pre-aggregated Aggregated/player_season_metrics.csv table.

Usage: python3 passing_metrics_compared.py "E. Mero"
"""
import glob
import json
import re
import sys
import collections

import numpy as np
import pandas as pd
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


AGG_DIR = "/Users/marclamberts/Event data/Ecuador 2026/Aggregated"
DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
BG = "#0d1117"

MIN_MINUTES = 270
POS_LABEL = {"1": "GK", "2": "DEF", "3": "MID", "4": "FWD"}
SET_PIECE_QIDS = {5, 6, 107}

RAW_SUM_COLS = [
    "minutes", "matches", "passes", "completed_passes", "forward_passes",
    "final3_entries", "progressive_passes", "pass_value_over_expected",
]


# ---------------------------------------------------------------- position --

def infer_positions():
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    positions = collections.defaultdict(collections.Counter)
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e["typeId"] != 34:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            ids = (qmap.get(30) or "").split(", ")
            codes = (qmap.get(44) or "").split(", ")
            for pid, code in zip(ids, codes):
                if code in POS_LABEL:
                    positions[pid][code] += 1
    return {pid: POS_LABEL[c.most_common(1)[0][0]] for pid, c in positions.items()}


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


# ------------------------------------------------------------- aggregation --

def load_player_table():
    df = pd.read_csv(f"{AGG_DIR}/player_season_metrics.csv")
    positions = infer_positions()
    grouped = df.groupby(["player_id", "player"], as_index=False)[RAW_SUM_COLS].sum()
    grouped["position"] = grouped["player_id"].map(positions)
    grouped["team"] = grouped["player_id"].map(
        df.sort_values("minutes", ascending=False).drop_duplicates("player_id").set_index("player_id")["team"]
    )

    n = grouped["minutes"].replace(0, np.nan) / 90
    grouped["nineties"] = n
    grouped["pass_completion_pct"] = grouped["completed_passes"] / grouped["passes"].replace(0, np.nan) * 100
    grouped["pass_efficiency"] = grouped["pass_value_over_expected"] / grouped["passes"].replace(0, np.nan) * 100
    grouped["gpa_passing_p90"] = grouped["pass_value_over_expected"] / n
    grouped["final3_entries_p90"] = grouped["final3_entries"] / n
    grouped["forward_passes_p90"] = grouped["forward_passes"] / n
    grouped["passes_p90"] = grouped["passes"] / n
    grouped["progressive_passes_p90"] = grouped["progressive_passes"] / n
    return grouped.replace([np.inf, -np.inf], np.nan)


METRICS = [
    ("gpa_passing_p90", "Goal Probability\nAdded Passing", "{:.2f}"),
    ("pass_completion_pct", "Pass Completion\nPercentage", "{:.1f}"),
    ("pass_efficiency", "Pass\nEfficiency", "{:.2f}"),
    ("final3_entries_p90", "Final Third\nEntries", "{:.2f}"),
    ("forward_passes_p90", "Forward\nPasses", "{:.1f}"),
    ("passes_p90", "Pass\nVolume", "{:.1f}"),
    ("progressive_passes_p90", "Progressive\nPasses", "{:.2f}"),
]


def percentile_of(value, sample):
    sample = np.asarray(sample, dtype=float)
    sample = sample[~np.isnan(sample)]
    if len(sample) == 0 or np.isnan(value):
        return 50.0
    return float((sample < value).sum() / len(sample) * 100)


# -------------------------------------------------------------- pass network --

def is_progressive(x0, y0, x1, y1):
    def dist_to_goal(x, y):
        dx = (100 - x) / 100 * 105.0
        dy = (y - 50) / 100 * 68.0
        return (dx ** 2 + dy ** 2) ** 0.5
    d0, d1 = dist_to_goal(x0, y0), dist_to_goal(x1, y1)
    if d1 >= d0:
        return False
    reduction = (d0 - d1) / d0
    if x0 < 50 and x1 < 50:
        return reduction >= 0.30
    elif x0 < 50 <= x1:
        return reduction >= 0.15
    return reduction >= 0.10


def collect_passes(player_name):
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    cid_to_team = build_team_map(files)
    passes = []
    team = "Unknown"
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e.get("playerName") != player_name or e["typeId"] != 1:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            if SET_PIECE_QIDS & set(qmap.keys()):
                continue
            team = cid_to_team.get(e["contestantId"], team)
            ex = float(qmap.get(140, e["x"]))
            ey = float(qmap.get(141, e["y"]))
            passes.append({
                "x0": e["x"], "y0": e["y"], "x1": ex, "y1": ey,
                "success": e["outcome"] == 1,
                "assist": e.get("assist") == 1,
                "key_pass": e.get("keyPass") == 1,
                "progressive": e["outcome"] == 1 and is_progressive(e["x"], e["y"], ex, ey),
            })
    return passes, team


# ------------------------------------------------------------------ plot --

def make_plot(player_name, grouped, out_path):
    me_rows = grouped[grouped["player"] == player_name]
    if me_rows.empty:
        raise SystemExit(f"'{player_name}' not found.")
    me = me_rows.iloc[0]
    peers = grouped[(grouped["position"] == me["position"]) & (grouped["minutes"] >= MIN_MINUTES)]
    if me["player_id"] not in peers["player_id"].values:
        peers = pd.concat([peers, me_rows])

    passes, team = collect_passes(player_name)

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor(BG)

    pitch_ax = fig.add_axes([0.02, 0.08, 0.32, 0.78])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color="#3a4a5c",
                           linewidth=1.0, half=False)
    pitch.draw(ax=pitch_ax)

    order = {"incomplete": 0, "completed": 1, "progressive": 2, "key_pass": 3, "assist": 4}

    def category(p):
        if p["assist"]:
            return "assist"
        if p["key_pass"]:
            return "key_pass"
        if not p["success"]:
            return "incomplete"
        if p["progressive"]:
            return "progressive"
        return "completed"

    colors = {"incomplete": "#333d4a", "completed": "#5b6472", "progressive": C_INDIGO,
              "key_pass": C_PINK, "assist": C_AMBER}
    for p in sorted(passes, key=lambda p: order[category(p)]):
        cat = category(p)
        color = colors[cat]
        lw = 1.6 if cat in ("progressive", "key_pass", "assist") else 0.5
        alpha = 0.85 if cat in ("progressive", "key_pass", "assist") else 0.35
        pitch.lines(p["x0"], p["y0"], p["x1"], p["y1"], ax=pitch_ax, color=color,
                    lw=lw, alpha=alpha, zorder=order[cat] + 1, comet=False)
        if cat in ("assist", "key_pass"):
            pitch.scatter(p["x1"], p["y1"], ax=pitch_ax, s=45, color=color, zorder=5,
                          edgecolors="white", linewidths=0.6)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_AMBER, markersize=8, label="Assist"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_PINK, markersize=8, label="Key Pass"),
        Line2D([0], [0], color=C_INDIGO, linewidth=2, label="Progressive"),
        Line2D([0], [0], color="#5b6472", linewidth=2, label="Completed"),
        Line2D([0], [0], color="#333d4a", linewidth=2, label="Incomplete"),
    ]
    pitch_ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.015),
                     ncol=3, frameon=False, fontsize=8.5)

    fig.text(0.02, 0.965, player_name, fontsize=27, fontweight="bold", family="sans-serif", color="#ffffff")
    fig.text(0.02, 0.928, f"Passing Metrics compared to {me['position']}s", fontsize=13.5,
             fontweight="bold", color="#d7dbe0")
    fig.text(0.02, 0.900, f"{team} · Ecuador 2026 · All Competitions", fontsize=10.5, color="#9aa4b2")

    # -------- right: percentile dot plot --------
    rx0, rx1 = 0.42, 0.94
    top, bottom = 0.83, 0.10
    n_m = len(METRICS)
    row_h = (top - bottom) / n_m
    for i, (key, label, fmt) in enumerate(METRICS):
        y = top - i * row_h - row_h * 0.5
        sample = peers[key].dropna().values
        v = me[key]
        if np.isnan(v):
            v = 0.0
        pct = percentile_of(v, sample)

        fig.text(rx0 - 0.02, y, label.replace("\n", " "), fontsize=11, fontweight="bold",
                 color="#f0f2f5", ha="right", va="center")
        track_y = y
        fig.add_artist(plt.Line2D([rx0, rx1], [track_y, track_y], transform=fig.transFigure,
                                   color="#232a35", linewidth=6, solid_capstyle="round", zorder=1))
        dot_x = rx0 + (rx1 - rx0) * (pct / 100)
        color = C_NAVY if pct >= 66 else (C_PURPLE if pct >= 33 else C_PINK)
        circ = plt.Circle((dot_x, track_y), 0.017, transform=fig.transFigure,
                           facecolor=color, edgecolor="white", linewidth=1.2, zorder=3)
        fig.add_artist(circ)
        fig.text(dot_x, track_y, fmt.format(v), fontsize=9.5, fontweight="bold", ha="center",
                 va="center", color="white", zorder=4)

    fig.text((rx0 + rx1) / 2, bottom - 0.045, f"Percentile vs. {me['position']}s "
             f"({len(peers)} players, ≥{MIN_MINUTES} min)", fontsize=9, color="#7b8794", ha="center")
    for x, lab in [(rx0, "0"), ((rx0 + rx1) / 2, "50"), (rx1, "100")]:
        fig.text(x, bottom - 0.02, lab, fontsize=8.5, color="#6b7684", ha="center")

    fig.text(0.02, 0.02, f"{me['nineties']:.1f} 90s played · Ecuador 2026 · {len(passes)} open play passes",
             fontsize=9, color="#7b8794")
    fig.text(0.985, 0.02, "Data via Opta | Ecuador 2026 event data", fontsize=8.5, color="#7b8794", ha="right")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"{player_name}: pos={me['position']} nineties={me['nineties']:.1f} "
          f"pass_pct={me['pass_completion_pct']:.1f} passes_p90={me['passes_p90']:.1f} "
          f"prog_p90={me['progressive_passes_p90']:.2f} peers_n={len(peers)}")


if __name__ == "__main__":
    player = sys.argv[1] if len(sys.argv) > 1 else "E. Mero"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/passing_metrics_compared_{player.replace(' ', '_')}.png"
    grouped = load_player_table()
    make_plot(player, grouped, out)
