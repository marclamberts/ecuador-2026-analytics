"""
Shot assists (Opta's keyPass/assist tags) broken down by delivery body
part. Note: in this feed, explicit body-part qualifiers on pass events
are only right-footed (20) / left-footed (72) -- headers and other body
parts are not separately tagged on passes (only on shots), so they fall
into "Other/Unspecified" here. This is a real data-coverage limit, not a
detection bug -- verified by checking qualifier frequency on pass events
league-wide before building this.

Usage: python3 shot_assists_by_body_part.py "Independiente del Valle" [out.png]
"""
import csv as csvmod
import glob
import json
import os
import re
import sys
import collections

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

RIGHT_QID = 20
LEFT_QID = 72
CATEGORY_COLORS = {"Right Foot": C_AMBER, "Left Foot": C_INDIGO, "Other/Unspecified": "#5b6472"}
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


def classify(qids):
    if LEFT_QID in qids:
        return "Left Foot"
    if RIGHT_QID in qids:
        return "Right Foot"
    return "Other/Unspecified"


def collect(files, cid):
    counts = collections.Counter()
    xa_by_cat = collections.Counter()
    assist_counts = collections.Counter()

    for fn in files:
        fn_base = fn.split("/")[-1]
        csv_path = os.path.join(DANGER_DIR, fn_base[:-5] + "_danger_models.csv")
        shot_xg = {}
        if os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                for row in csvmod.DictReader(f):
                    shot_xg[int(row["event_id"])] = float(row["xg"])

        with open(fn) as f:
            data = json.load(f)
        evs = data["event"]

        assist_pass_xa = {}
        for e in evs:
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

        for e in evs:
            if e["typeId"] != 1 or e.get("contestantId") != cid:
                continue
            if not (e.get("keyPass") == 1 or e.get("assist") == 1):
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            cat = classify(set(qmap.keys()))
            counts[cat] += 1
            is_assist = e.get("assist") == 1
            if is_assist:
                assist_counts[cat] += 1
            xa_by_cat[cat] += assist_pass_xa.get(e["eventId"], 0.04 if is_assist else 0.0)

    return counts, xa_by_cat, assist_counts


def make_plot(team_name, counts, xa_by_cat, assist_counts, out_path):
    n_total = sum(counts.values())
    cats = sorted(counts, key=lambda c: -counts[c])

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y_pos = list(range(len(cats)))[::-1]
    bar_vals = [counts[c] for c in cats]
    colors = [CATEGORY_COLORS[c] for c in cats]
    ax.barh(y_pos, bar_vals, color=colors, height=0.58, zorder=3)

    for y, cat, val in zip(y_pos, cats, bar_vals):
        pct = val / n_total * 100 if n_total else 0
        ax.text(val + max(bar_vals) * 0.015, y, f"{val}  ({pct:.0f}%)  ·  {assist_counts.get(cat,0)} assists  "
                f"·  {xa_by_cat.get(cat,0):.1f} xA",
                va="center", ha="left", fontsize=11, color=TEXT_MAIN, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cats, fontsize=13, color=TEXT_MAIN, fontweight="bold")
    ax.set_xlabel("Shot assists (key passes + assists)", fontsize=11, color=TEXT_SUB, labelpad=10)
    ax.tick_params(colors=TEXT_SUB, labelsize=10)
    ax.set_xlim(0, max(bar_vals) * 1.55)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7, zorder=0)

    fig.text(0.05, 0.955, clean_name(team_name), fontsize=22, fontweight="bold", color="white")
    fig.text(0.05, 0.905, f"Shot Assists by Body Part  ·  {n_total} total  ·  Ecuador 2026  ·  Season",
             fontsize=12.5, color=TEXT_SUB)
    fig.text(0.05, 0.02, "Data via Opta | Ecuador 2026 event data · shot assist = keyPass or assist tag · "
             "\"Other/Unspecified\" includes headers -- Opta doesn't tag body part on non-foot passes "
             "in this feed", fontsize=7.8, color="#6b7684")

    fig.subplots_adjust(left=0.14, right=0.96, top=0.86, bottom=0.13)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(dict(counts), dict(xa_by_cat), dict(assist_counts))


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/shot_assists_by_body_part_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    counts, xa_by_cat, assist_counts = collect(files, cid)
    make_plot(match, counts, xa_by_cat, assist_counts, out)
