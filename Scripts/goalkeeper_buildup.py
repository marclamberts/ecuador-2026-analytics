"""
Goalkeeper Distribution: how involved is each team's keeper in build-up
play? x = GK passes per game (involvement volume), y = % of those passes
played short (<32m, i.e. into their own defense rather than launched
long) -- a buildup-style indicator. Starting keeper only, identified from
each match's TeamSetup position codes (qualifier 44 == 1).

Usage: python3 goalkeeper_buildup.py [out.png]
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

LONG_BALL_M = 32.0


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


def dist_m(x0, y0, x1, y1):
    dx = (x1 - x0) / 100 * 105.0
    dy = (y1 - y0) / 100 * 68.0
    return (dx ** 2 + dy ** 2) ** 0.5


def pass_end_xy(e):
    qmap = {q["qualifierId"]: q.get("value") for q in e.get("qualifier", [])}
    ex, ey = qmap.get(140), qmap.get(141)
    return (float(ex) if ex is not None else e["x"]), (float(ey) if ey is not None else e["y"])


def find_keepers(data):
    """contestantId -> starting goalkeeper playerId, for this match."""
    keepers = {}
    for e in data.get("event", []):
        if e.get("typeId") != 34:
            continue
        qmap = {q["qualifierId"]: q.get("value") for q in e.get("qualifier", [])}
        pos_str, pid_str = qmap.get(44), qmap.get(30)
        if not pos_str or not pid_str:
            continue
        positions = [p.strip() for p in pos_str.split(",")]
        pids = [p.strip() for p in pid_str.split(",")]
        for pid, pos in zip(pids, positions):
            if pos == "1":
                keepers[e["contestantId"]] = pid
                break
    return keepers


def collect_gk_distribution(files, team_to_cid):
    cid_to_team = {v: k for k, v in team_to_cid.items()}
    games = collections.defaultdict(int)
    gk_passes = collections.defaultdict(int)
    gk_short = collections.defaultdict(int)
    gk_complete = collections.defaultdict(int)

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        keepers = find_keepers(data)
        teams_here = [cid_to_team[c] for c in keepers if c in cid_to_team]
        for t in teams_here:
            games[t] += 1

        for e in data.get("event", []):
            if e.get("typeId") != 1:
                continue
            cid = e.get("contestantId")
            t = cid_to_team.get(cid)
            if t is None or keepers.get(cid) != e.get("playerId"):
                continue
            gk_passes[t] += 1
            if str(e.get("outcome")) == "1":
                gk_complete[t] += 1
            x1, y1 = pass_end_xy(e)
            if dist_m(e["x"], e["y"], x1, y1) < LONG_BALL_M:
                gk_short[t] += 1

    return games, gk_passes, gk_short, gk_complete


def make_plot(d, out_path):
    files, team_to_cid, points = d["files"], d["team_to_cid"], d["points"]
    games, gk_passes, gk_short, gk_complete = collect_gk_distribution(files, team_to_cid)

    teams = [t for t in team_to_cid if games.get(t, 0) > 0 and gk_passes.get(t, 0) > 0]
    per_game = {t: gk_passes[t] / games[t] for t in teams}
    short_pct = {t: gk_short[t] / gk_passes[t] * 100 for t in teams}

    fig, ax = plt.subplots(figsize=(15, 10.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    mean_x = sum(per_game.values()) / len(teams)
    mean_y = sum(short_pct.values()) / len(teams)
    ax.axhline(mean_y, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.45, zorder=1)
    ax.axvline(mean_x, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.45, zorder=1)

    pts_vals = [points.get(t, 0) for t in teams]
    cmap = LinearSegmentedColormap.from_list("pts", ["#e05c5c", "#ffc247", "#4ade80"])
    pmin, pmax = min(pts_vals), max(pts_vals)
    sizes = [220 + (points.get(t, 0) - pmin) / (pmax - pmin) * 700 if pmax > pmin else 400 for t in teams]

    xs = [per_game[t] for t in teams]
    ys = [short_pct[t] for t in teams]
    sc = ax.scatter(xs, ys, s=sizes, c=pts_vals, cmap=cmap, edgecolors="white",
                    linewidths=1.3, alpha=0.9, zorder=3)

    texts = []
    for i, t in enumerate(teams):
        texts.append(ax.text(xs[i], ys[i] + 1.2, pil.clean_name(t), fontsize=10, color=TEXT_MAIN,
                             fontweight="bold", ha="center", zorder=4))
    adjust_text(texts, ax=ax, expand=(1.4, 2.0), force_text=(0.6, 1.2), force_points=(0.3, 0.5),
               arrowprops=dict(arrowstyle="-", color="#4a5568", lw=0.7))

    x_pad = (max(xs) - min(xs)) * 0.12
    y_pad = (max(ys) - min(ys)) * 0.12
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    ax.text(xlo + (xhi - xlo) * 0.015, yhi - (yhi - ylo) * 0.02, "Build-up outlet\n(busy + plays short)",
            fontsize=9.5, color="#4ade80", style="italic", va="top")
    ax.text(xhi - (xhi - xlo) * 0.015, yhi - (yhi - ylo) * 0.02, "Sweeper-keeper,\nstill goes long",
            fontsize=9.5, color="#ffc247", style="italic", va="top", ha="right")
    ax.text(xlo + (xhi - xlo) * 0.015, ylo + (yhi - ylo) * 0.02, "Rarely used,\nplays short when he is",
            fontsize=9.5, color="#5b9bd5", style="italic", va="bottom")
    ax.text(xhi - (xhi - xlo) * 0.015, ylo + (yhi - ylo) * 0.02, "Traditional shot-stopper\n(rare + long)",
            fontsize=9.5, color="#9aa4b2", style="italic", va="bottom", ha="right")

    ax.set_xlabel("GK PASSES PER GAME  (involvement volume)", fontsize=12, color=TEXT_MAIN,
                 fontweight="bold", labelpad=12)
    ax.set_ylabel(f"% OF GK PASSES SHORT  (<{LONG_BALL_M:.0f}m, buildup indicator)", fontsize=12,
                 color=TEXT_MAIN, fontweight="bold", labelpad=12)
    ax.tick_params(colors=TEXT_SUB, labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.5)

    cbar = fig.colorbar(sc, ax=ax, pad=0.012, fraction=0.035)
    cbar.set_label("Points", color=TEXT_MAIN, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB)

    fig.text(0.05, 0.965, "Ecuador 2026  ·  Goalkeeper Distribution — Involvement in Build-up",
             fontsize=20, fontweight="bold", color="white")
    fig.text(0.05, 0.93, "Starting keeper only, all matches  ·  Dot size = points total",
             fontsize=11.5, color=TEXT_SUB)
    fig.text(0.05, 0.908, "Top-right = keeper as extra build-up player  ·  Bottom-left = classic "
             "shot-stopper, rarely involved", fontsize=9, color="#6b7684")
    fig.text(0.05, 0.012, f"Data via Opta | Ecuador 2026 event data · Short pass = completed end-point "
             f"within {LONG_BALL_M:.0f}m of the keeper's touch, straight-line distance", fontsize=8,
             color="#6b7684")
    fig.text(0.98, 0.012, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.06, right=0.94, top=0.86, bottom=0.09)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/goalkeeper_buildup.png"
    d = pil.load_all()
    make_plot(d, out)
