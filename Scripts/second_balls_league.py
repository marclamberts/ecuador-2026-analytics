"""
Set-Piece Second Balls -- League Ranking: for every team's own corners,
free kicks, and long throws into the final third, who wins the loose
ball after the first contact? Same identification method as
set_piece_second_balls.py (single-team pitch map), rolled up into one
win-rate ranking across all 16 teams.

Usage: python3 second_balls_league.py [out.png]
"""
import sys
import glob
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
RED = "#e0765c"
NEUTRAL = "#3a4658"

CORNER_QID = 6
FREEKICK_QID = 5
THROWIN_QID = 107
CONTESTED_TYPES = {44, 7, 8, 12, 49}
FINAL_THIRD_START = 200 / 3
LONG_THROW_MIN_M = 25
WINDOW_EVENTS = 5
WINDOW_MIN = 12 / 60
MAX_CLEAN_PASSES_BETWEEN = 1
MIN_CONTESTS = 15


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


def minute_value(e):
    return float(e.get("timeMin") or 0) + float(e.get("timeSec") or 0) / 60.0


def dist_m(x0, y0, x1, y1):
    dx = (x1 - x0) / 100 * 105.0
    dy = (y1 - y0) / 100 * 68.0
    return (dx ** 2 + dy ** 2) ** 0.5


def collect_second_balls(files, team_to_cid):
    cid_to_team = {v: k for k, v in team_to_cid.items()}
    total = collections.Counter()
    won = collections.Counter()

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        evs = [e for e in data["event"] if e.get("periodId") in (1, 2)]
        evs.sort(key=lambda e: (e["periodId"], minute_value(e), e.get("eventId", 0)))

        for i, e in enumerate(evs):
            if e["typeId"] != 1 or e.get("contestantId") not in cid_to_team:
                continue
            t = cid_to_team[e["contestantId"]]
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            qids = set(qmap.keys())
            x0, y0 = float(e["x"]), float(e["y"])
            x1 = float(qmap.get(140, x0))
            y1 = float(qmap.get(141, y0))
            if x1 < FINAL_THIRD_START:
                continue
            is_corner = CORNER_QID in qids
            is_fk = FREEKICK_QID in qids
            is_long_throw = THROWIN_QID in qids and dist_m(x0, y0, x1, y1) >= LONG_THROW_MIN_M
            if not (is_corner or is_fk or is_long_throw):
                continue

            t0 = minute_value(e)
            window = [ev for ev in evs[i + 1:i + 1 + WINDOW_EVENTS]
                      if minute_value(ev) - t0 <= WINDOW_MIN]
            first_idx = next((j for j, ev in enumerate(window) if ev["typeId"] in CONTESTED_TYPES), None)
            if first_idx is None:
                continue
            sb, clean_passes = None, 0
            for ev in window[first_idx + 1:]:
                if ev["typeId"] in CONTESTED_TYPES:
                    sb = ev
                    break
                if ev["typeId"] == 1 and ev["outcome"] == 1:
                    clean_passes += 1
                    if clean_passes > MAX_CLEAN_PASSES_BETWEEN:
                        break
            if sb is None:
                continue
            total[t] += 1
            if sb.get("contestantId") == e["contestantId"]:
                won[t] += 1

    return total, won


def make_plot(d, out_path):
    files, team_to_cid = d["files"], d["team_to_cid"]
    total, won = collect_second_balls(files, team_to_cid)

    teams = [t for t in team_to_cid if total.get(t, 0) >= MIN_CONTESTS]
    pct = {t: won[t] / total[t] * 100 for t in teams}
    teams.sort(key=lambda t: -pct[t])
    n = len(teams)

    league_avg = sum(won.values()) / sum(total.values()) * 100

    fig, ax = plt.subplots(figsize=(13, 0.62 * n + 2.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y_pos = list(range(n))[::-1]
    for y, t in zip(y_pos, teams):
        val = pct[t]
        color = GREEN if val >= league_avg + 6 else (RED if val <= league_avg - 6 else NEUTRAL)
        ax.barh(y, val, color=color, height=0.62, zorder=3)
        ax.text(val + 1.4, y, f"{val:.0f}%  ({won[t]}/{total[t]})", va="center", ha="left",
                fontsize=9.5, color=TEXT_MAIN, fontweight="bold")

    ax.axvline(league_avg, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.6, zorder=2)
    ax.text(league_avg, n - 0.1, "league avg", fontsize=8.5, color=TEXT_SUB, ha="center", va="bottom")

    ax.set_yticks(y_pos)
    labels = [f"#{i+1}  {pil.clean_name(t)}" for i, t in enumerate(teams)]
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", colors=TEXT_MAIN, length=0)
    ax.set_xlabel("Second balls won  (% of identifiable contests after own set-piece deliveries)",
                 fontsize=10.5, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_xlim(0, max(pct.values()) * 1.22)
    ax.set_ylim(-0.7, n - 0.2)

    fig.text(0.05, 0.975, "Ecuador 2026  ·  All Teams  ·  Set-Piece Second Balls",
             fontsize=19, fontweight="bold", color="white")
    fig.text(0.05, 0.951, "Own corners, free kicks & long throws into the final third  ·  Who wins the "
             f"loose ball?  ·  Min. {MIN_CONTESTS} contests", fontsize=10.5, color=TEXT_SUB)
    fig.text(0.05, 0.022, "Data via Opta | Ecuador 2026 event data · Second contested action (aerial, "
             "tackle, interception, clearance, recovery) within ~12s of the delivery, max 1 clean pass "
             "between", fontsize=7.8, color="#6b7684")
    fig.text(0.05, 0.006, "Often well outside the box once the first header is cleared", fontsize=7.8,
             color="#6b7684")
    fig.text(0.98, 0.014, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.30, right=0.94, top=0.905, bottom=0.085)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/second_balls_league.png"
    d = pil.load_all()
    make_plot(d, out)
