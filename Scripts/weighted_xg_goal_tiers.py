"""
Weighted xG & Goal Tiers -- per match, weighted 70/30 (xG/goals). Each team
plotted by weighted-for vs weighted-against rate; diagonal dashed lines mark
constant weighted goal-difference tiers. Y-axis (against) is inverted so
"good defense" points up and "elite" sits top-right, matching the Cannon
Stats reference layout.

Usage: python3 weighted_xg_goal_tiers.py [out.png]
"""
import glob
import sys
import collections

import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

import pi_ratings_lib as pil

DANGER_DIR = "/Users/marclamberts/Event data/Ecuador 2026/Danger"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
C_AMBER = "#ffc247"
C_RED = "#e0765c"

W_XG = 0.7
W_GOALS = 0.3
TIERS = [-1, -0.5, 0, 0.5, 1, 1.5, 2]


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


def collect_weighted(matches, team_to_cid):
    cid_to_team = {v: k for k, v in team_to_cid.items()}
    csv_lookup = {}
    for path in glob.glob(f"{DANGER_DIR}/*_danger_models.csv"):
        base = path.split("/")[-1].replace("_danger_models.csv", "")
        csv_lookup[base] = path

    xg_for = collections.defaultdict(float)
    xg_against = collections.defaultdict(float)
    goals_for = collections.defaultdict(int)
    goals_against = collections.defaultdict(int)
    games = collections.defaultdict(int)

    for m in matches:
        h, a = m["home"], m["away"]
        if h not in team_to_cid or a not in team_to_cid:
            continue
        key = f"{m['date']}_{h} - {a}"
        path = csv_lookup.get(key)
        if not path:
            continue
        df = pd.read_csv(path)
        xg_h = df.loc[df["contestant_id"] == team_to_cid[h], "xg"].sum()
        xg_a = df.loc[df["contestant_id"] == team_to_cid[a], "xg"].sum()

        xg_for[h] += xg_h
        xg_against[h] += xg_a
        xg_for[a] += xg_a
        xg_against[a] += xg_h

        goals_for[h] += m["home_goals"]
        goals_against[h] += m["away_goals"]
        goals_for[a] += m["away_goals"]
        goals_against[a] += m["home_goals"]

        games[h] += 1
        games[a] += 1

    weighted_for, weighted_against = {}, {}
    for t in games:
        n = games[t]
        weighted_for[t] = W_XG * (xg_for[t] / n) + W_GOALS * (goals_for[t] / n)
        weighted_against[t] = W_XG * (xg_against[t] / n) + W_GOALS * (goals_against[t] / n)

    return weighted_for, weighted_against, games


def tier_label_pos(d_val, x_min, x_max, y_min, y_max):
    """Find where a diagonal (against = for - d_val) crosses the plot
    box near the top, matching the reference's label placement."""
    y_top = y_min + 0.04 * (y_max - y_min)
    x_at_top = y_top + d_val
    if x_min <= x_at_top <= x_max:
        return x_at_top, y_top
    x_right = x_max - 0.02 * (x_max - x_min)
    y_at_right = x_right - d_val
    if y_min <= y_at_right <= y_max:
        return x_right, y_at_right
    x_left = x_min + 0.02 * (x_max - x_min)
    y_at_left = x_left - d_val
    return x_left, y_at_left


def make_plot(d, out_path):
    matches, team_to_cid = d["matches"], d["team_to_cid"]
    weighted_for, weighted_against, games = collect_weighted(matches, team_to_cid)
    teams = [t for t in games if games[t] > 0]

    xs = [weighted_for[t] for t in teams]
    ys = [weighted_against[t] for t in teams]

    fig, ax = plt.subplots(figsize=(14.5, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    pad_x = (max(xs) - min(xs)) * 0.14
    pad_y = (max(ys) - min(ys)) * 0.14
    x_min, x_max = min(xs) - pad_x, max(xs) + pad_x
    y_min, y_max = max(0.0, min(ys) - pad_y), max(ys) + pad_y

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # inverted: 0 at top

    for d_val in TIERS:
        color = C_RED if d_val < 0 else GRID_COLOR
        alpha = 0.55 if d_val < 0 else 0.8
        x0, y0 = x_min, x_min - d_val
        x1, y1 = x_max, x_max - d_val
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.0,
                linestyle=(0, (4, 3)), alpha=alpha, zorder=1)
        lx, ly = tier_label_pos(d_val, x_min, x_max, y_min, y_max)
        label = f"{d_val:+.1f}".rstrip("0").rstrip(".") if d_val != 0 else "0"
        ax.text(lx, ly, label, fontsize=9, color=color, alpha=0.9, zorder=1,
                ha="center", va="center")

    lg_avg_for = sum(xs) / len(xs)
    lg_avg_against = sum(ys) / len(ys)
    ax.axvline(lg_avg_for, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.4, zorder=1)
    ax.axhline(lg_avg_against, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.4, zorder=1)
    # keep clear of the tier labels, which cluster near the top edge
    ax.text(lg_avg_for - 0.012 * (x_max - x_min), y_max - 0.02 * (y_max - y_min), "LG AVG ATTACK",
            fontsize=8, color=TEXT_SUB, va="bottom", ha="right", rotation=90)
    ax.text(x_max - 0.01 * (x_max - x_min), lg_avg_against - 0.02 * (y_max - y_min), "LG AVG\nDEFENSE",
            fontsize=8, color=TEXT_SUB, va="bottom", ha="right")

    ax.scatter(xs, ys, s=260, color=C_AMBER, edgecolors=BG, linewidths=1.4, alpha=0.95, zorder=3)

    texts = []
    for t, x, y in zip(teams, xs, ys):
        texts.append(ax.text(x, y, pil.clean_name(t), fontsize=10.5, color=TEXT_MAIN,
                             fontweight="bold", zorder=4))
    adjust_text(texts, ax=ax, expand=(1.25, 1.6),
               arrowprops=dict(arrowstyle="-", color="#4a5568", lw=0.6))

    ax.annotate("", xy=(x_min + 0.005 * (x_max - x_min), y_min + 0.22 * (y_max - y_min)),
               xytext=(x_min + 0.005 * (x_max - x_min), y_min + 0.32 * (y_max - y_min)),
               arrowprops=dict(arrowstyle="-|>", color=TEXT_SUB, lw=1.3))
    ax.text(x_min + 0.02 * (x_max - x_min), y_min + 0.24 * (y_max - y_min), "Good Defense",
            fontsize=9.5, color=TEXT_SUB, rotation=90, va="bottom")

    ax.annotate("", xy=(x_max - 0.03 * (x_max - x_min), y_max - 0.03 * (y_max - y_min)),
               xytext=(x_max - 0.13 * (x_max - x_min), y_max - 0.03 * (y_max - y_min)),
               arrowprops=dict(arrowstyle="-|>", color=TEXT_SUB, lw=1.3))
    ax.text(x_max - 0.14 * (x_max - x_min), y_max - 0.045 * (y_max - y_min), "Good Attack",
            fontsize=9.5, color=TEXT_SUB, ha="right")

    ax.set_xlabel("WEIGHTED xG AND G — FOR / MATCH", fontsize=11.5, color=TEXT_MAIN,
                 fontweight="bold", labelpad=12)
    ax.set_ylabel("WEIGHTED xG AND G — AGAINST / MATCH", fontsize=11.5, color=TEXT_MAIN,
                 fontweight="bold", labelpad=12)
    ax.tick_params(colors=TEXT_SUB, labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.5)

    fig.text(0.05, 0.965, "Ecuador 2026  ·  Weighted xG & Goal Tiers",
             fontsize=20, fontweight="bold", color="white")
    fig.text(0.05, 0.932, "Weighted xG & Goal Tiers — Per Match  ·  Weighted 70/30",
             fontsize=12, color=TEXT_SUB)
    fig.text(0.05, 0.012, "Data via Opta | Ecuador 2026 event data · xg = mean of 5 calibrated shot "
             "models (danger_score methodology)  ·  Weighted 70% xG / 30% actual goals",
             fontsize=8, color="#6b7684")
    fig.text(0.98, 0.012, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.07, right=0.96, top=0.87, bottom=0.08)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/weighted_xg_goal_tiers.png"
    d = pil.load_all()
    make_plot(d, out)
