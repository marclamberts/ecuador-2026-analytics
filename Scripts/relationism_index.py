"""
Relationism Index: a composite proxy score (0-100) for how much a team's
play leans toward "relationist" principles -- fluid, proximity-based
combination play -- versus a more rigid, positional structure. This is a
proxy built from event data, not a claim about a club's actual coaching
philosophy.

Three equally-weighted components, each percentile-ranked 0-100 across
the league:
  1. Short-combination tendency  -- inverse of average completed pass
     distance (shorter passes = more combination-based)
  2. Central overload tendency   -- share of a team's own touches that
     fall in the central third of the pitch width (relationism leans on
     central/half-space overloads rather than fixed wide occupation)
  3. Possession retention        -- average number of passes per
     possession sequence (patient combination play retains the ball
     longer before losing it or shooting)

Usage: python3 relationism_index.py [out.png]
"""
import sys
import json
import collections

import matplotlib.pyplot as plt

import pi_ratings_lib as pil

LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
GREEN = "#4a9e5c"
BLUE = "#5b9bd5"
NEUTRAL = "#3a4658"

MIN_PASSES = 3
MIN_DURATION_S = 1.0
MAX_DURATION_S = 300.0
MAX_SPEED_MPS = 20.0

NON_TOUCH_TYPES = {17, 18, 19, 27, 28, 30, 32, 34, 37, 40, 43, 58, 65, 70, 71, 79, 84}


def add_logo(fig, width=0.11, margin=0.014):
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
    seq_lengths = collections.defaultdict(list)

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        evs = [e for e in data["event"] if e.get("periodId") in (1, 2, 3, 4) and e.get("contestantId")]
        evs.sort(key=lambda e: (e.get("periodId", 0), minute_value(e), e.get("eventId", 0)))

        # touches / pass distance / central share
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

        # possession sequences
        seq, cur_cid = [], None
        for e in evs:
            cid = e["contestantId"]
            if cid != cur_cid:
                if seq:
                    _record_seq(seq, cur_cid, cid_to_team, seq_lengths)
                seq, cur_cid = [e], cid
            else:
                seq.append(e)
        if seq:
            _record_seq(seq, cur_cid, cid_to_team, seq_lengths)

    avg_pass_dist = {t: pass_dist_sum[t] / pass_dist_n[t] for t in pass_dist_n if pass_dist_n[t] > 0}
    central_share = {t: central_n[t] / touch_n[t] * 100 for t in touch_n if touch_n[t] > 0}
    avg_seq_len = {t: sum(v) / len(v) for t, v in seq_lengths.items() if v}

    return avg_pass_dist, central_share, avg_seq_len


def _record_seq(seq, cid, cid_to_team, seq_lengths):
    team = cid_to_team.get(cid)
    if team is None:
        return
    n_passes = sum(1 for e in seq if e["typeId"] == 1)
    if n_passes < MIN_PASSES:
        return
    first, last = seq[0], seq[-1]
    duration_s = (minute_value(last) - minute_value(first)) * 60
    if not (MIN_DURATION_S <= duration_s <= MAX_DURATION_S):
        return
    seq_lengths[team].append(n_passes)


def make_plot(d, out_path):
    files, team_to_cid = d["files"], d["team_to_cid"]
    avg_pass_dist, central_share, avg_seq_len = collect_all(files, team_to_cid)

    teams = [t for t in team_to_cid if t in avg_pass_dist and t in central_share and t in avg_seq_len]

    short_combo_rank = percentile_rank({t: -avg_pass_dist[t] for t in teams})
    central_rank = percentile_rank({t: central_share[t] for t in teams})
    retention_rank = percentile_rank({t: avg_seq_len[t] for t in teams})

    index = {t: (short_combo_rank[t] + central_rank[t] + retention_rank[t]) / 3 for t in teams}
    teams.sort(key=lambda t: -index[t])
    n = len(teams)

    fig, ax = plt.subplots(figsize=(13.5, 0.62 * n + 2.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    mean_idx = sum(index.values()) / n
    y_pos = list(range(n))[::-1]
    for y, t in zip(y_pos, teams):
        val = index[t]
        color = GREEN if val >= mean_idx + 6 else (BLUE if val <= mean_idx - 6 else NEUTRAL)
        ax.barh(y, val, color=color, height=0.62, zorder=3)
        ax.text(val + 1.2, y, f"{val:.0f}", va="center", ha="left", fontsize=10.5,
                color=TEXT_MAIN, fontweight="bold")

    ax.axvline(mean_idx, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.6, zorder=2)
    ax.text(mean_idx, n - 0.1, "league avg", fontsize=8.5, color=TEXT_SUB, ha="center", va="bottom")

    ax.set_yticks(y_pos)
    labels = [f"#{i+1}  {pil.clean_name(t)}" for i, t in enumerate(teams)]
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", colors=TEXT_MAIN, length=0)
    ax.set_xlabel("Relationism Index  (0 = most positional/structured · 100 = most relationist)",
                 fontsize=10.5, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, n - 0.2)

    fig.text(0.05, 0.975, "Ecuador 2026  ·  All Teams  ·  Relationism Index",
             fontsize=19, fontweight="bold", color="white")
    fig.text(0.05, 0.951, "Proxy score: short-combination tendency + central/half-space overload + "
             "possession retention", fontsize=10.5, color=TEXT_SUB)
    fig.text(0.05, 0.022, "Data via Opta | Ecuador 2026 event data · Equal-weighted percentile blend of "
             "avg. pass distance (inverted), central-third touch share, and passes per possession sequence",
             fontsize=7.8, color="#6b7684")
    fig.text(0.05, 0.006, "A proxy built from event data, not a claim about a club's actual coaching "
             "philosophy", fontsize=7.8, color="#6b7684")
    fig.text(0.98, 0.014, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.30, right=0.94, top=0.905, bottom=0.085)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/relationism_index.png"
    d = pil.load_all()
    make_plot(d, out)
