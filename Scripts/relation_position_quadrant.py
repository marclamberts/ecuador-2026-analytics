"""
Position vs Relation quadrant: each team plotted by directness (x, a
positional-play indicator -- net possession speed, m/s) against a
relationist blend (y -- short-combination tendency + central/half-space
overload). A companion view to relationism_index.py's composite bar
chart, showing the same underlying signals split across two axes instead
of collapsed into one score. A proxy from event data, not a claim about
actual coaching philosophy.

Usage: python3 relation_position_quadrant.py [out.png]
"""
import sys
import json
import collections

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text

import pi_ratings_lib as pil

LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"

MIN_PASSES = 3
MIN_DURATION_S = 1.0
MAX_DURATION_S = 300.0
MAX_SPEED_MPS = 20.0

NON_TOUCH_TYPES = {17, 18, 19, 27, 28, 30, 32, 34, 37, 40, 43, 58, 65, 70, 71, 79, 84}


def add_logo(fig, width=0.13, margin=0.014):
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


def percentile_rank(values):
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(items)
    ranks = {}
    for i, (k, v) in enumerate(items):
        ranks[k] = i / (n - 1) * 100 if n > 1 else 50.0
    return ranks


def minute_value(e):
    return float(e.get("timeMin") or 0) + float(e.get("timeSec") or 0) / 60.0


def dist_m(x0, y0, x1, y1):
    dx = (x1 - x0) / 100 * 105.0
    dy = (y1 - y0) / 100 * 68.0
    return (dx ** 2 + dy ** 2) ** 0.5


def pass_end_xy(e):
    qmap = {q["qualifierId"]: q.get("value") for q in e.get("qualifier", [])}
    ex, ey = qmap.get(140), qmap.get(141)
    return (float(ex) if ex is not None else e["x"]), (float(ey) if ey is not None else e["y"])


def collect_all(files, team_to_cid):
    cid_to_team = {v: k for k, v in team_to_cid.items()}

    pass_dist_sum = collections.defaultdict(float)
    pass_dist_n = collections.defaultdict(int)
    central_n = collections.defaultdict(int)
    touch_n = collections.defaultdict(int)
    speeds = collections.defaultdict(list)

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        evs = [e for e in data["event"] if e.get("periodId") in (1, 2, 3, 4) and e.get("contestantId")]
        evs.sort(key=lambda e: (e.get("periodId", 0), minute_value(e), e.get("eventId", 0)))

        for e in evs:
            t = cid_to_team.get(e["contestantId"])
            if t is None:
                continue
            if e.get("typeId") == 1:
                x1, y1 = pass_end_xy(e)
                pass_dist_sum[t] += dist_m(e["x"], e["y"], x1, y1)
                pass_dist_n[t] += 1
            if e.get("typeId") in NON_TOUCH_TYPES:
                continue
            x, y = e.get("x"), e.get("y")
            if x is None or y is None or (x == 0 and y == 0):
                continue
            touch_n[t] += 1
            if 33.33 <= y <= 66.67:
                central_n[t] += 1

        seq, cur_cid = [], None
        for e in evs:
            cid = e["contestantId"]
            if cid != cur_cid:
                if seq:
                    _record_seq(seq, cur_cid, cid_to_team, speeds)
                seq, cur_cid = [e], cid
            else:
                seq.append(e)
        if seq:
            _record_seq(seq, cur_cid, cid_to_team, speeds)

    avg_pass_dist = {t: pass_dist_sum[t] / pass_dist_n[t] for t in pass_dist_n if pass_dist_n[t] > 0}
    central_share = {t: central_n[t] / touch_n[t] * 100 for t in touch_n if touch_n[t] > 0}
    avg_speed = {t: sum(v) / len(v) for t, v in speeds.items() if v}

    return avg_pass_dist, central_share, avg_speed


def _record_seq(seq, cid, cid_to_team, speeds):
    team = cid_to_team.get(cid)
    if team is None:
        return
    n_passes = sum(1 for e in seq if e["typeId"] == 1)
    if n_passes < MIN_PASSES:
        return
    first, last = seq[0], seq[-1]
    x0, y0 = first["x"], first["y"]
    x1, y1 = pass_end_xy(last) if last["typeId"] == 1 else (last["x"], last["y"])
    duration_s = (minute_value(last) - minute_value(first)) * 60
    if not (MIN_DURATION_S <= duration_s <= MAX_DURATION_S):
        return
    distance = dist_m(x0, y0, x1, y1)
    speed = distance / duration_s
    if not (0 < speed <= MAX_SPEED_MPS):
        return
    speeds[team].append(speed)


def make_plot(d, out_path):
    files, team_to_cid, points = d["files"], d["team_to_cid"], d["points"]
    avg_pass_dist, central_share, avg_speed = collect_all(files, team_to_cid)
    teams = [t for t in team_to_cid if t in avg_pass_dist and t in central_share and t in avg_speed]

    directness_rank = percentile_rank({t: avg_speed[t] for t in teams})
    short_combo_rank = percentile_rank({t: -avg_pass_dist[t] for t in teams})
    central_rank = percentile_rank({t: central_share[t] for t in teams})
    relation_rank = {t: (short_combo_rank[t] + central_rank[t]) / 2 for t in teams}

    fig, ax = plt.subplots(figsize=(15, 10.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.axhline(50, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.45, zorder=1)
    ax.axvline(50, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.45, zorder=1)

    pts_vals = [points.get(t, 0) for t in teams]
    cmap = LinearSegmentedColormap.from_list("pts", ["#e05c5c", "#ffc247", "#4ade80"])
    pmin, pmax = min(pts_vals), max(pts_vals)
    sizes = [220 + (points.get(t, 0) - pmin) / (pmax - pmin) * 700 if pmax > pmin else 400 for t in teams]

    xs = [directness_rank[t] for t in teams]
    ys = [relation_rank[t] for t in teams]
    sc = ax.scatter(xs, ys, s=sizes, c=pts_vals, cmap=cmap, edgecolors="white",
                    linewidths=1.3, alpha=0.9, zorder=3)

    texts = []
    for i, t in enumerate(teams):
        label = pil.clean_name(t)
        texts.append(ax.text(xs[i], ys[i] + 2.2, label, fontsize=10, color=TEXT_MAIN,
                             fontweight="bold", ha="center", zorder=4))
    adjust_text(texts, ax=ax, expand=(1.4, 2.0), force_text=(0.6, 1.2), force_points=(0.3, 0.5),
               arrowprops=dict(arrowstyle="-", color="#4a5568", lw=0.7))

    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.text(2, 100, "Relationist", fontsize=10.5, color="#4ade80", style="italic", va="top")
    ax.text(98, 100, "Hybrid / Transitional", fontsize=10.5, color="#ffc247", style="italic", va="top", ha="right")
    ax.text(2, 2, "Passive / Structured", fontsize=10.5, color="#9aa4b2", style="italic")
    ax.text(98, 2, "Positional / Direct", fontsize=10.5, color="#5b9bd5", style="italic", ha="right")

    ax.set_xlabel("DIRECTNESS RATING  (possession speed, percentile — positional indicator)", fontsize=12,
                 color=TEXT_MAIN, fontweight="bold", labelpad=12)
    ax.set_ylabel("RELATIONIST RATING  (short combinations + central overload, percentile)", fontsize=12,
                 color=TEXT_MAIN, fontweight="bold", labelpad=12)
    ax.tick_params(colors=TEXT_SUB, labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.5)

    cbar = fig.colorbar(sc, ax=ax, pad=0.012, fraction=0.035)
    cbar.set_label("Points", color=TEXT_MAIN, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB)

    fig.text(0.05, 0.965, "Ecuador 2026  ·  Position vs Relation — Team Quadrants",
             fontsize=20, fontweight="bold", color="white")
    fig.text(0.05, 0.93, "Each team positioned by possession directness vs relationist tendencies "
             "(event-data proxy)  ·  Dot size = points total", fontsize=11.5, color=TEXT_SUB)
    fig.text(0.05, 0.908, "Top-left = relationist  ·  Bottom-right = positional/direct  ·  Proxy from "
             "data, not a claim about coaching philosophy", fontsize=9, color="#6b7684")
    fig.text(0.05, 0.012, "Data via Opta | Ecuador 2026 event data · x = avg. possession-sequence speed "
             "(m/s)  ·  y = avg. of (inverted pass distance, central-third touch share)", fontsize=8,
             color="#6b7684")
    fig.text(0.98, 0.012, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.06, right=0.94, top=0.86, bottom=0.09)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/relation_position_quadrant.png"
    d = pil.load_all()
    make_plot(d, out)
