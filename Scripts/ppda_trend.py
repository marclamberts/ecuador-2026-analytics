"""
PPDA (passes allowed per defensive action) per match across the season,
with a rolling average -- the standard proxy for pressing intensity.
Lower PPDA = more intense press.

PPDA = (opponent passes in their own defensive+middle two-thirds, x<60 in
their own attacking frame) / (this team's tackles+interceptions+fouls in
the opponent's defensive+middle two-thirds, x>=40 in this team's own
attacking frame). Both thresholds describe the same physical zone since
each team's x is already normalised to its own attacking direction.

Usage: python3 ppda_trend.py "Independiente del Valle" [out.png]
"""
import glob
import json
import re
import sys
import collections

import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch  # noqa: F401 (kept for style parity, unused here)

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

DEF_ACTION_TYPES = {7, 8, 4}  # Tackle, Interception, Foul
OPP_ZONE_MAX_X = 60
OWN_ZONE_MIN_X = 40
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


def opponent_name(fn, team_name):
    m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", fn.split("/")[-1])
    if not m:
        return "?"
    home, away = m.group(1), m.group(2)
    other = away if team_name.lower() in home.lower() else home
    return clean_name(other)


def collect(files, cid, team_name):
    rows = []
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        cids = set(e["contestantId"] for e in data["event"] if "contestantId" in e)
        if cid not in cids or len(cids) != 2:
            continue
        opp = next(iter(cids - {cid}))

        opp_passes_zone = 0
        def_actions_zone = 0
        for e in data["event"]:
            x = e.get("x")
            if x is None:
                continue
            x = float(x)
            if e["typeId"] == 1 and e.get("contestantId") == opp and x < OPP_ZONE_MAX_X:
                opp_passes_zone += 1
            elif e["typeId"] in DEF_ACTION_TYPES and e.get("contestantId") == cid and x >= OWN_ZONE_MIN_X:
                def_actions_zone += 1

        if def_actions_zone == 0:
            continue
        m = re.match(r"(\d{4}-\d{2}-\d{2})_", fn.split("/")[-1])
        date = m.group(1) if m else "?"
        rows.append({
            "date": date, "opponent": opponent_name(fn, team_name),
            "ppda": opp_passes_zone / def_actions_zone,
            "opp_passes": opp_passes_zone, "def_actions": def_actions_zone,
        })
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
    ppdas = [r["ppda"] for r in rows]
    roll = rolling_mean(ppdas, ROLL_WINDOW)
    season_avg = sum(ppdas) / n

    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.axhline(season_avg, color="#465061", linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(n + 0.3, season_avg, f"Season avg {season_avg:.1f}", fontsize=9,
            color="#9aa4b2", va="center", ha="left")

    ax.plot(xs, ppdas, color=C_INDIGO, alpha=0.35, linewidth=1.3, marker="o",
            markersize=5, zorder=2, label="Per-match PPDA")
    ax.plot(xs, roll, color=C_AMBER, linewidth=3, zorder=3,
            label=f"{ROLL_WINDOW}-match rolling average")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"MD{i}" for i in xs], fontsize=9)
    ax.tick_params(colors=TEXT_SUB, labelsize=10)
    ax.set_ylabel("PPDA  (lower = more intense press)", fontsize=11.5, color=TEXT_MAIN,
                  fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)

    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=TEXT_MAIN)

    fig.text(0.06, 0.965, clean_name(team_name), fontsize=22, fontweight="bold", color="white")
    fig.text(0.06, 0.928, "PPDA by Match  ·  Pressing Intensity  ·  Ecuador 2026  ·  Season",
             fontsize=12.5, color=TEXT_SUB)
    fig.text(0.06, 0.02, "Data via Opta | Ecuador 2026 event data · PPDA = opposition passes in their own "
             "defensive two-thirds / this team's tackles+interceptions+fouls in that same zone",
             fontsize=8, color="#6b7684")

    fig.subplots_adjust(left=0.07, right=0.96, top=0.87, bottom=0.11)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"n_matches={n} season_avg_ppda={season_avg:.2f}")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/ppda_trend_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    rows = collect(files, cid, team_name)
    make_plot(match, rows, out)
