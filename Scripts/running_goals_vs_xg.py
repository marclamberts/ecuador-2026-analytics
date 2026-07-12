"""
Running Goals vs xG chart for the Ecuador 2026 Opta F24-style event dataset.
Uses the pre-computed xG values from Danger/*_danger_models.csv.

Usage: python3 running_goals_vs_xg.py "E. Mero"
"""
import csv
import glob
import json
import os
import re
import sys
import collections

import matplotlib.pyplot as plt

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

SHOT_TYPES = {13, 14, 15, 16}

# ---- shared palette ----
C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"

BG = "#0d1117"


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


def load_danger_csv(fn_base):
    csv_path = os.path.join(DANGER_DIR, fn_base[:-5] + "_danger_models.csv")
    rows = {}
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["event_id"])] = row
    return rows


def collect_shots(player_name):
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    cid_to_team = build_team_map(files)
    shots = []
    matches_played = set()
    for fn in files:
        fn_base = fn.split("/")[-1]
        match_date = fn_base[:10]
        danger_rows = load_danger_csv(fn_base)
        with open(fn) as f:
            data = json.load(f)
        player_in_match = False
        for e in data["event"]:
            if e.get("playerName") != player_name:
                continue
            player_in_match = True
            if e["typeId"] not in SHOT_TYPES:
                continue
            row = danger_rows.get(e["id"])
            if row is None or row["is_penalty"] == "1":
                continue
            shots.append({
                "date": match_date,
                "period": int(e.get("periodId") or 0),
                "min": int(e.get("timeMin") or 0),
                "sec": int(e.get("timeSec") or 0),
                "is_goal": row["is_goal"] == "1",
                "xg": float(row["xg"]),
                "team": cid_to_team.get(e["contestantId"], "Unknown"),
            })
        if player_in_match:
            matches_played.add(fn_base)
    shots.sort(key=lambda s: (s["date"], s["period"], s["min"], s["sec"]))
    return shots, matches_played


def make_plot(player_name, shots, matches_played, out_path):
    n = len(shots)
    xs = list(range(0, n + 1))
    cum_goals = [0.0]
    cum_xg = [0.0]
    g, x = 0.0, 0.0
    for s in shots:
        g += 1.0 if s["is_goal"] else 0.0
        x += s["xg"]
        cum_goals.append(g)
        cum_xg.append(x)

    band_hi = [v * 1.25 for v in cum_xg]
    band_lo = [v * 0.75 for v in cum_xg]

    total_goals = cum_goals[-1]
    total_xg = cum_xg[-1]
    diff = total_goals - total_xg
    team = shots[0]["team"] if shots else "Unknown"

    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.fill_between(xs, band_lo, band_hi, color=C_PURPLE, alpha=0.12, linewidth=0, zorder=1)
    ax.plot(xs, band_hi, color=C_PURPLE, alpha=0.35, linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
    ax.plot(xs, band_lo, color=C_PURPLE, alpha=0.35, linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)

    ax.plot(xs, cum_xg, color=C_NAVY, linewidth=2.4, linestyle=(0, (5, 2)), zorder=3)
    ax.plot(xs, cum_goals, color=C_CORAL, linewidth=2.4, linestyle=(0, (5, 2)), zorder=3)

    ax.text(xs[-1] + max(2, n * 0.01), cum_xg[-1], "xG", color=C_NAVY,
            fontsize=13, fontweight="bold", va="center")
    ax.text(xs[-1] + max(2, n * 0.01), cum_goals[-1], "Goals", color=C_CORAL,
            fontsize=13, fontweight="bold", va="center")

    y_top = max(cum_goals[-1], band_hi[-1]) * 1.12
    sign = "+" if diff >= 0 else ""
    ax.text(xs[-1] * 0.98, y_top * 0.96,
             f"{total_goals:.0f} goals from {total_xg:.1f} xG ({sign}{diff:.1f})",
             ha="right", fontsize=12, color="#d7dbe0")

    ax.set_xlim(0, xs[-1] * 1.10 if n else 1)
    ax.set_ylim(0, y_top)
    ax.set_xlabel("Shots", fontsize=12, fontweight="bold", color="#f0f2f5")
    ax.set_ylabel("Total Goals & xG", fontsize=12, fontweight="bold", color="#f0f2f5")

    ax.grid(axis="y", color="#1c2129", linewidth=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#6b7684")
    ax.tick_params(colors="#9aa4b2")

    fig.text(0.045, 0.955, player_name, fontsize=30, fontweight="bold", family="sans-serif", color="#ffffff")
    fig.text(0.045, 0.912, "Running Goals vs xG (excl. penalties)", fontsize=15, fontweight="bold", color="#d7dbe0")
    fig.text(0.045, 0.885, f"{team} · Ecuador 2026 · All Matches",
             fontsize=11.5, color="#9aa4b2")

    fig.text(0.045, 0.03, f"All Matches · {n} non-penalty shots", fontsize=9.5, color="#7b8794")
    fig.text(0.985, 0.03, "Data via Opta | Ecuador 2026 event data", fontsize=9.5, color="#7b8794", ha="right")

    fig.subplots_adjust(left=0.06, right=0.93, top=0.83, bottom=0.10)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"Shots={n} Goals={total_goals:.0f} xG={total_xg:.2f} Diff={diff:+.2f}")


if __name__ == "__main__":
    player = sys.argv[1] if len(sys.argv) > 1 else "E. Mero"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/running_goals_vs_xg_{player.replace(' ', '_')}.png"
    shots, matches = collect_shots(player)
    if not shots:
        print(f"No shots found for player '{player}'")
        sys.exit(1)
    make_plot(player, shots, matches, out)
