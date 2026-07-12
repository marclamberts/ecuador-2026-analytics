"""
Field Tilt per match across the season, with a rolling average. Field
tilt = this team's final-third passes / (this team's + the opposition's
final-third passes) * 100 -- a territorial-dominance metric independent
of possession share.

Usage: python3 field_tilt_trend.py "Independiente del Valle" [out.png]
"""
import glob
import json
import re
import sys
import collections

import matplotlib.pyplot as plt

DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"
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

FINAL_THIRD_START = 200 / 3
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


def collect(files, cid):
    rows = []
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        cids = set(e["contestantId"] for e in data["event"] if "contestantId" in e)
        if cid not in cids or len(cids) != 2:
            continue
        opp = next(iter(cids - {cid}))

        idv_ft, opp_ft = 0, 0
        for e in data["event"]:
            if e["typeId"] != 1 or e.get("x") is None:
                continue
            if float(e["x"]) < FINAL_THIRD_START:
                continue
            if e.get("contestantId") == cid:
                idv_ft += 1
            elif e.get("contestantId") == opp:
                opp_ft += 1

        total = idv_ft + opp_ft
        if total == 0:
            continue
        m = re.match(r"(\d{4}-\d{2}-\d{2})_", fn.split("/")[-1])
        date = m.group(1) if m else "?"
        rows.append({"date": date, "tilt": idv_ft / total * 100,
                     "idv_ft": idv_ft, "opp_ft": opp_ft})
    rows.sort(key=lambda r: r["date"])
    return rows


def rolling_mean(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def make_plot(team_name, rows, out_path):
    n = len(rows)
    xs = list(range(1, n + 1))
    tilts = [r["tilt"] for r in rows]
    roll = rolling_mean(tilts, ROLL_WINDOW)
    season_avg = sum(tilts) / n

    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.axhline(50, color="#465061", linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    ax.text(n + 0.3, 50, "Parity (50%)", fontsize=9, color="#9aa4b2", va="center", ha="left")

    ax.fill_between(xs, 50, roll, where=[r >= 50 for r in roll], color=C_AMBER, alpha=0.12, zorder=1)
    ax.fill_between(xs, 50, roll, where=[r < 50 for r in roll], color=C_NAVY, alpha=0.12, zorder=1)

    ax.plot(xs, tilts, color=C_PINK, alpha=0.3, linewidth=1.3, marker="o",
            markersize=5, zorder=2, label="Per-match field tilt")
    ax.plot(xs, roll, color=C_AMBER, linewidth=3, zorder=3,
            label=f"{ROLL_WINDOW}-match rolling average")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"MD{i}" for i in xs], fontsize=9)
    ax.tick_params(colors=TEXT_SUB, labelsize=10)
    ax.set_ylabel("Field Tilt %  (share of final-third passes)", fontsize=11.5, color=TEXT_MAIN,
                  fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.set_ylim(0, 100)

    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=TEXT_MAIN)

    fig.text(0.06, 0.965, clean_name(team_name), fontsize=22, fontweight="bold", color="white")
    fig.text(0.06, 0.928, f"Field Tilt by Match  ·  Season avg {season_avg:.0f}%  ·  Ecuador 2026  ·  Season",
             fontsize=12.5, color=TEXT_SUB)
    fig.text(0.06, 0.02, "Data via Opta | Ecuador 2026 event data · field tilt = this team's final-third "
             "passes / (this team's + opposition's final-third passes)", fontsize=8, color="#6b7684")

    fig.subplots_adjust(left=0.07, right=0.96, top=0.87, bottom=0.11)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"n_matches={n} season_avg_tilt={season_avg:.1f}")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/field_tilt_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    rows = collect(files, cid)
    make_plot(match, rows, out)
