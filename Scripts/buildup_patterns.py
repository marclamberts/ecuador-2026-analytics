"""
The 3 most common build-up patterns: possession sequences that start with
a touch in the team's own defensive third (x<33.3) and end, without the
ball changing possession, in a shot. Sequences are classified by which
channel (left / center / right, using the average y of the pass chain)
they progress through, and the most representative example from each of
the 3 most common channels is drawn.

A "sequence" is a maximal run of consecutive events by the same
contestantId (same possession-chain heuristic as team_directness.py).
Only sequences with >=3 completed passes are considered, to exclude
scrappy loose-ball scrambles.

Usage: python3 buildup_patterns.py "Independiente del Valle" [out.png]
"""
import glob
import json
import re
import sys
import statistics
import collections

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mplsoccer import VerticalPitch

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
LINE_COLOR = "#c7ccd4"
PITCH_LINE = "#2c3a4d"
DEF_THIRD_END = 100 / 3

SHOT_TYPES = {13, 14, 15, 16}
SHOT_NAMES = {13: "Missed", 14: "Hit Post", 15: "Saved", 16: "GOAL"}
MIN_PASSES = 3

PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name):
    return PREFIX_RE.sub("", name)


def add_logo(fig, width=0.15, margin=0.014):
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


def minute_value(e):
    return float(e.get("timeMin") or 0) + float(e.get("timeSec") or 0) / 60.0


def collect_sequences(files, cid):
    qualifying = []
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        evs = [e for e in data["event"] if e.get("periodId") in (1, 2) and e.get("contestantId")]
        evs.sort(key=lambda e: (e["periodId"], minute_value(e), e.get("eventId", 0)))

        seq, cur_cid = [], None
        runs = []
        for e in evs:
            c = e["contestantId"]
            if c != cur_cid:
                if seq and cur_cid == cid:
                    runs.append(seq)
                seq, cur_cid = [e], c
            else:
                seq.append(e)
        if seq and cur_cid == cid:
            runs.append(seq)

        for seq in runs:
            first = seq[0]
            if first.get("x") is None or first["x"] >= DEF_THIRD_END:
                continue
            last_shot = next((e for e in reversed(seq) if e["typeId"] in SHOT_TYPES), None)
            if last_shot is None:
                continue
            n_passes = sum(1 for e in seq if e["typeId"] == 1)
            if n_passes < MIN_PASSES:
                continue
            qualifying.append({"seq": seq, "n_passes": n_passes, "shot": last_shot,
                               "match": fn.split("/")[-1]})
    return qualifying


def sequence_route(seq):
    points, players = [], []
    for e in seq:
        if e["typeId"] == 1:
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            x0, y0 = float(e["x"]), float(e["y"])
            x1 = float(qmap.get(140, x0))
            y1 = float(qmap.get(141, y0))
            if not points:
                points.append((x0, y0))
                players.append(e.get("playerName", "?"))
            points.append((x1, y1))
            players.append(None)
        elif e["typeId"] == 3:
            pt = (float(e["x"]), float(e["y"]))
            if not points:
                points.append(pt)
            points.append(pt)
            players.append(e.get("playerName", "?"))
        elif e["typeId"] in SHOT_TYPES:
            points.append((float(e["x"]), float(e["y"])))
            players.append(e.get("playerName", "?"))

    # collapse a shot point that lands on (or essentially on) the previous
    # touch, so the star marker isn't drawn stacked on a numbered dot
    if len(points) >= 2:
        (x0, y0), (x1, y1) = points[-2], points[-1]
        if ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 < 1.0:
            points.pop(-2)
            players.pop(-2)
    return points, players


def classify(item):
    passes = [e for e in item["seq"] if e["typeId"] == 1]
    ys = [e["y"] for e in passes if e.get("y") is not None]
    if not ys:
        return None
    avg_y = sum(ys) / len(ys)
    if avg_y < DEF_THIRD_END:
        return "Right Channel"
    if avg_y > (100 - DEF_THIRD_END):
        return "Left Channel"
    return "Central"


def pick_representative(items):
    lengths = sorted(i["n_passes"] for i in items)
    median = statistics.median(lengths)
    goals = [i for i in items if i["shot"]["typeId"] == 16]
    pool = goals if goals else items
    return min(pool, key=lambda i: abs(i["n_passes"] - median))


def make_plot(team_name, categories, total_n, out_path):
    fig = plt.figure(figsize=(19, 12.5))
    fig.patch.set_facecolor(BG)

    fig.text(0.03, 0.955, clean_name(team_name), fontsize=27, fontweight="bold", color="white")
    fig.text(0.03, 0.918, "Most Common Build-Up Patterns  ·  Defensive Third → Shot  ·  Ecuador 2026  ·  Season",
             fontsize=13.5, fontweight="bold", color="#c7ccd4")
    fig.text(0.03, 0.892, f"{total_n} qualifying sequences this season (uninterrupted possession, "
             f"≥{MIN_PASSES} passes, starting in the defensive third and ending in a shot)",
             fontsize=10, color="#9aa4b2")

    n_panels = len(categories)
    axes = [fig.add_axes([0.02 + i * (0.96 / n_panels), 0.05, 0.96 / n_panels - 0.02, 0.75])
            for i in range(n_panels)]

    for ax, cat in zip(axes, categories):
        pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                               linewidth=1.1, half=False)
        pitch.draw(ax=ax)

        item = cat["example"]
        points, players = sequence_route(item["seq"])
        n = len(points)
        for i in range(n - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            is_shot_leg = (i == n - 2)
            color = C_GREEN if (is_shot_leg and item["shot"]["typeId"] == 16) else C_AMBER
            pitch.lines(x0, y0, x1, y1, ax=ax, color=color, lw=2.6 if is_shot_leg else 2.0,
                       alpha=0.95, zorder=2, comet=True,
                       transparent=True if not is_shot_leg else False)

        for i, (x, y) in enumerate(points):
            is_last = i == n - 1
            marker_color = C_GREEN if (is_last and item["shot"]["typeId"] == 16) else C_AMBER
            size = 340 if is_last else 200
            marker = "*" if is_last else "o"
            pitch.scatter(x, y, ax=ax, s=size if marker == "o" else size * 2.2, color=marker_color,
                          edgecolors=BG, linewidths=1.6, zorder=4, marker=marker)
            if not is_last:
                pitch.annotate(str(i + 1), xy=(x, y), ax=ax, ha="center", va="center",
                               fontsize=8.5, fontweight="bold", color=BG, zorder=5)

        pitch.polygon([[[0, 0], [0, 100], [DEF_THIRD_END, 100], [DEF_THIRD_END, 0]]],
                      ax=ax, facecolor="#1e3a5f", edgecolor="none", alpha=0.3, zorder=0.5)

        shot_label = SHOT_NAMES[item["shot"]["typeId"]]
        pct = cat["count"] / total_n * 100
        shooter = item["shot"].get("playerName", "?")
        x_center = ax.get_position().x0 + ax.get_position().width / 2
        fig.text(x_center, 0.855, f"{cat['name']}  ({cat['count']} of {total_n} · {pct:.0f}%)",
                 fontsize=15, fontweight="bold", ha="center", color="#e6e9ee",
                 transform=fig.transFigure)
        fig.text(x_center, 0.825, f"{item['n_passes']} passes · shot by {shooter} · {shot_label}",
                 fontsize=10, ha="center", color="#9aa4b2", transform=fig.transFigure)

    fig.text(0.03, 0.014, "Data via Opta | Ecuador 2026 event data · numbers = pass sequence order · "
             "★ = shot location", fontsize=8.5, color="#6b7684")
    fig.text(0.98, 0.014, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/buildup_patterns_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]

    qualifying = collect_sequences(files, cid)
    by_cat = collections.defaultdict(list)
    for item in qualifying:
        cat = classify(item)
        if cat:
            by_cat[cat].append(item)

    ranked = sorted(by_cat.items(), key=lambda kv: -len(kv[1]))[:3]
    categories = [{"name": name, "count": len(items), "example": pick_representative(items)}
                  for name, items in ranked]

    print(f"Total qualifying sequences: {len(qualifying)}")
    for c in categories:
        print(c["name"], c["count"], "example n_passes=", c["example"]["n_passes"],
              "match=", c["example"]["match"])

    make_plot(match, categories, len(qualifying), out)
