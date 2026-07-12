"""
Team directness scatter for the Ecuador 2026 dataset: average possession
sequence speed (m/s) vs average sequence length (passes), one dot per team.

A "sequence" is a maximal run of consecutive events belonging to the same
team (contestantId) in a match. Length = number of pass events in the run.
Speed = net displacement (first event's location to last pass's end
location, in meters) / sequence duration (seconds). Sequences with fewer
than 3 passes, or degenerate duration/speed, are dropped.

Usage: python3 team_directness.py [out.png]
"""
import glob
import json
import re
import sys
import collections

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
from adjustText import adjust_text

DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"
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

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
BG = "#0d1117"
DOT_COLOR = "#5b6472"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"

HIGHLIGHT_TEAM = "Independiente del Valle"

MIN_PASSES = 3
MIN_DURATION_S = 1.0
MAX_DURATION_S = 300.0
MAX_SPEED_MPS = 20.0

PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name):
    return PREFIX_RE.sub("", name)


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
    return {v: clean_name(k) for k, v in team_to_cid.items()}


def minute_value(e):
    return float(e.get("timeMin") or 0) + float(e.get("timeSec") or 0) / 60.0


def dist_m(x0, y0, x1, y1):
    dx = (x1 - x0) / 100 * 105.0
    dy = (y1 - y0) / 100 * 68.0
    return (dx ** 2 + dy ** 2) ** 0.5


def pass_end_xy(e):
    qmap = {q["qualifierId"]: q.get("value") for q in e.get("qualifier", [])}
    ex = qmap.get(140)
    ey = qmap.get(141)
    return (float(ex) if ex is not None else e["x"]), (float(ey) if ey is not None else e["y"])


def collect_sequences(files, cid_to_team):
    rows = []
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        evs = [e for e in data["event"] if e.get("periodId") in (1, 2, 3, 4) and e.get("contestantId")]
        evs.sort(key=lambda e: (e.get("periodId", 0), minute_value(e), e.get("eventId", 0)))

        seq = []
        cur_cid = None
        for e in evs:
            cid = e["contestantId"]
            if cid != cur_cid:
                if seq:
                    rows.append(_summarize(seq, cur_cid, cid_to_team))
                seq = [e]
                cur_cid = cid
            else:
                seq.append(e)
        if seq:
            rows.append(_summarize(seq, cur_cid, cid_to_team))

    return [r for r in rows if r is not None]


def _summarize(seq, cid, cid_to_team):
    team = cid_to_team.get(cid)
    if team is None:
        return None
    n_passes = sum(1 for e in seq if e["typeId"] == 1)
    if n_passes < MIN_PASSES:
        return None

    first, last = seq[0], seq[-1]
    x0, y0 = first["x"], first["y"]
    x1, y1 = pass_end_xy(last) if last["typeId"] == 1 else (last["x"], last["y"])

    duration_s = (minute_value(last) - minute_value(first)) * 60
    if not (MIN_DURATION_S <= duration_s <= MAX_DURATION_S):
        return None

    distance = dist_m(x0, y0, x1, y1)
    speed = distance / duration_s
    if not (0 < speed <= MAX_SPEED_MPS):
        return None

    return {"team": team, "length": n_passes, "speed": speed}


def aggregate(rows):
    by_team = collections.defaultdict(list)
    for r in rows:
        by_team[r["team"]].append(r)
    out = []
    for team, items in by_team.items():
        n = len(items)
        avg_len = sum(i["length"] for i in items) / n
        avg_speed = sum(i["speed"] for i in items) / n
        out.append({"team": team, "avg_length": avg_len, "avg_speed": avg_speed, "n": n})
    return out


def make_plot(team_stats, out_path):
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    xs = [t["avg_length"] for t in team_stats]
    ys = [t["avg_speed"] for t in team_stats]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    ax.axvline(mean_x, color="#465061", linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.axhline(mean_y, color="#465061", linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)

    texts = []
    for t in team_stats:
        is_hl = t["team"] == HIGHLIGHT_TEAM
        color = C_AMBER if is_hl else DOT_COLOR
        size = 320 if is_hl else 170
        edge = "white" if is_hl else "#8b93a1"
        ax.scatter(t["avg_length"], t["avg_speed"], s=size, color=color,
                   edgecolors=edge, linewidths=1.6 if is_hl else 0.9,
                   zorder=4 if is_hl else 3, alpha=0.95)
        txt = ax.text(t["avg_length"], t["avg_speed"], t["team"],
                      fontsize=12.5 if is_hl else 10, fontweight="bold" if is_hl else "normal",
                      color=C_AMBER if is_hl else TEXT_MAIN, zorder=5)
        texts.append(txt)

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#4a5568", lw=0.7),
               expand_points=(1.6, 1.8))

    ax.set_xlabel("Avg. sequence length (passes)", fontsize=13, color=TEXT_MAIN, fontweight="bold", labelpad=12)
    ax.set_ylabel("Avg. sequence speed (m/s)", fontsize=13, color=TEXT_MAIN, fontweight="bold", labelpad=12)
    ax.tick_params(colors=TEXT_SUB, labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6)

    fig.text(0.06, 0.965, "Ecuador 2026 · Team Directness", fontsize=22, fontweight="bold", color="white")
    fig.text(0.06, 0.935, "Avg. possession-sequence speed vs. avg. sequence length · All Competitions",
             fontsize=12.5, color=TEXT_SUB)
    fig.text(0.06, 0.02, "Data via Opta | Ecuador 2026 event data. Sequence = consecutive same-team "
             "events; length = passes, speed = net displacement / duration (min. 3 passes).",
             fontsize=8, color="#6b7684")

    fig.subplots_adjust(left=0.08, right=0.96, top=0.89, bottom=0.09)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    for t in sorted(team_stats, key=lambda t: -t["avg_speed"]):
        print(f"{t['team']:30s} len={t['avg_length']:.2f} speed={t['avg_speed']:.2f} n={t['n']}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/team_directness.png"
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    cid_to_team = build_team_map(files)
    rows = collect_sequences(files, cid_to_team)
    team_stats = aggregate(rows)
    make_plot(team_stats, out)
