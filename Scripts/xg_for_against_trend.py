"""
xG For vs xG Against per match across the season, with rolling averages.
Uses the pre-computed shot-level xg column in Danger/*_danger_models.csv
(mean of 5 calibrated models -- see danger_score methodology).

Usage: python3 xg_for_against_trend.py "Independiente del Valle" [out.png]
"""
import glob
import re
import sys
import collections

import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"
DANGER_DIR = "/Users/marclamberts/Event data/Ecuador 2026/Danger"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"

ROLL_WINDOW = 5
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
        import json
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


def opponent_name(matchup, team_name):
    m = re.match(r"(.+) - (.+)", matchup)
    if not m:
        return "?"
    home, away = m.group(1), m.group(2)
    other = away if team_name.lower() in home.lower() else home
    return clean_name(other)


def rolling_mean(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def collect(team_name, cid):
    files = [f for f in sorted(glob.glob(f"{DANGER_DIR}/*_danger_models.csv"))
             if team_name.lower() in f.lower()]
    rows = []
    for fn in files:
        df = pd.read_csv(fn)
        base = fn.split("/")[-1].replace("_danger_models.csv", "")
        m = re.match(r"(\d{4}-\d{2}-\d{2})_(.+)", base)
        date, matchup = m.group(1), m.group(2)
        cids_in_file = df["contestant_id"].unique()
        opp = next((c for c in cids_in_file if c != cid), None)
        xg_for = df.loc[df["contestant_id"] == cid, "xg"].sum()
        xg_against = df.loc[df["contestant_id"] == opp, "xg"].sum() if opp else 0.0
        rows.append({
            "date": date, "opponent": opponent_name(matchup, team_name),
            "xg_for": xg_for, "xg_against": xg_against,
        })
    rows.sort(key=lambda r: r["date"])
    return rows


def make_plot(team_name, rows, out_path):
    n = len(rows)
    xs = list(range(1, n + 1))
    xgf = [r["xg_for"] for r in rows]
    xga = [r["xg_against"] for r in rows]
    roll_f = rolling_mean(xgf, ROLL_WINDOW)
    roll_a = rolling_mean(xga, ROLL_WINDOW)
    avg_f, avg_a = sum(xgf) / n, sum(xga) / n

    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(xs, xgf, color=C_AMBER, alpha=0.28, linewidth=1.2, marker="o", markersize=4.5, zorder=2)
    ax.plot(xs, xga, color=C_NAVY, alpha=0.28, linewidth=1.2, marker="o", markersize=4.5, zorder=2)
    ax.plot(xs, roll_f, color=C_AMBER, linewidth=3, zorder=3,
            label=f"xG For ({ROLL_WINDOW}-match avg)")
    ax.plot(xs, roll_a, color=C_NAVY, linewidth=3, zorder=3,
            label=f"xG Against ({ROLL_WINDOW}-match avg)")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"MD{i}" for i in xs], fontsize=9)
    ax.tick_params(colors=TEXT_SUB, labelsize=10)
    ax.set_ylabel("xG", fontsize=12, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.set_ylim(bottom=0)

    ax.legend(loc="upper right", frameon=False, fontsize=10.5, labelcolor=TEXT_MAIN)

    fig.text(0.06, 0.965, clean_name(team_name), fontsize=22, fontweight="bold", color="white")
    fig.text(0.06, 0.928, f"xG For vs xG Against by Match  ·  Season avg {avg_f:.2f} for / "
             f"{avg_a:.2f} against  ·  Ecuador 2026",
             fontsize=12.5, color=TEXT_SUB)
    fig.text(0.06, 0.02, "Data via Opta | Ecuador 2026 event data · xg = mean of 5 calibrated shot "
             "models (see danger_score methodology)", fontsize=8, color="#6b7684")

    fig.subplots_adjust(left=0.06, right=0.96, top=0.87, bottom=0.11)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"n_matches={n} avg_xgf={avg_f:.2f} avg_xga={avg_a:.2f}")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/xg_trend_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    rows = collect(match, cid)
    if not rows:
        print("No danger_models.csv files found for this team.")
        sys.exit(1)
    make_plot(match, rows, out)
