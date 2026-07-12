"""
Pressures and pressure regains by third of the pitch. A "pressure" is a
tackle, interception, or foul committed (the same action set PPDA uses to
measure pressing intensity elsewhere in this project). A "regain" is a
pressure that directly wins the ball back: a successful tackle or an
interception (interceptions are always a regain by definition; a foul
committed is a pressure attempt but hands the opposition a free kick, so
it never counts as a regain).

Usage: python3 pressures_by_third.py "Independiente del Valle" [out.png]
"""
import glob
import json
import re
import sys
import collections

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
C_GREEN = "#4ade80"
BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"

PRESSURE_TYPES = {7, 8, 4}  # Tackle, Interception, Foul
DEF_THIRD_END = 100 / 3
FINAL_THIRD_START = 200 / 3
THIRDS = ["Defensive Third", "Middle Third", "Final Third"]

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


def third_of(x):
    if x < DEF_THIRD_END:
        return "Defensive Third"
    if x > FINAL_THIRD_START:
        return "Final Third"
    return "Middle Third"


def collect(files, cid):
    pressures = collections.Counter()
    regains = collections.Counter()
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e["typeId"] not in PRESSURE_TYPES or e.get("contestantId") != cid:
                continue
            if e.get("x") is None:
                continue
            t = third_of(float(e["x"]))
            pressures[t] += 1
            if e["typeId"] == 8 or (e["typeId"] == 7 and e.get("outcome") == 1):
                regains[t] += 1
    return pressures, regains


def make_plot(team_name, pressures, regains, out_path):
    n_pressures = sum(pressures.values())
    n_regains = sum(regains.values())
    pct = n_regains / n_pressures * 100 if n_pressures else 0

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x = np.arange(len(THIRDS))
    width = 0.32
    p_vals = [pressures.get(t, 0) for t in THIRDS]
    r_vals = [regains.get(t, 0) for t in THIRDS]

    bars1 = ax.bar(x - width / 2, p_vals, width, color=C_AMBER, label="Pressures", zorder=3)
    bars2 = ax.bar(x + width / 2, r_vals, width, color=C_GREEN, label="Regains", zorder=3)

    for xi, p, r in zip(x, p_vals, r_vals):
        ax.text(xi - width / 2, p + max(p_vals) * 0.02, str(p), ha="center", va="bottom",
                fontsize=12, color=TEXT_MAIN, fontweight="bold")
        rpct = r / p * 100 if p else 0
        ax.text(xi + width / 2, r + max(p_vals) * 0.02, f"{r}\n({rpct:.0f}%)", ha="center", va="bottom",
                fontsize=11, color=C_GREEN, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(" Third", "\nThird") for t in THIRDS], fontsize=12.5,
                       color=TEXT_MAIN, fontweight="bold")
    ax.tick_params(colors=TEXT_SUB, labelsize=10)
    ax.set_ylabel("Count", fontsize=11.5, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_ylim(0, max(p_vals) * 1.25)

    ax.legend(loc="upper right", frameon=False, fontsize=11, labelcolor=TEXT_MAIN)

    fig.text(0.06, 0.955, clean_name(team_name), fontsize=22, fontweight="bold", color="white")
    fig.text(0.06, 0.905, f"Pressures & Regains by Third  ·  {n_pressures} pressures, {n_regains} regains "
             f"({pct:.0f}%)  ·  Ecuador 2026  ·  Season", fontsize=12, color=TEXT_SUB)
    fig.text(0.06, 0.02, "Data via Opta | Ecuador 2026 event data · pressure = tackle, interception, or "
             "foul committed · regain = successful tackle or interception", fontsize=8, color="#6b7684")

    fig.subplots_adjust(left=0.08, right=0.96, top=0.86, bottom=0.15)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print("pressures:", dict(pressures), "regains:", dict(regains))


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/pressures_by_third_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    pressures, regains = collect(files, cid)
    make_plot(match, pressures, regains, out)
