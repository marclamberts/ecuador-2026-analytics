"""
Home vs Away PI Rating Gap: each team's final home ELO minus away ELO,
ranked by league position. Positive = stronger at home, negative = "road
warrior" (rare -- stronger away than at home).

Usage: python3 home_away_rating_gap.py [out.png]
"""
import sys

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
GAP_THRESHOLD = 25

ZONE_TOP, ZONE_BOTTOM = 4, 3
ZONE_TOP_COLOR = "#6fcf7a"
ZONE_MID_COLOR = "#6fa8dc"
ZONE_BOTTOM_COLOR = "#e0965c"


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


def zone_color(rank, n_teams):
    if rank <= ZONE_TOP:
        return ZONE_TOP_COLOR
    if rank > n_teams - ZONE_BOTTOM:
        return ZONE_BOTTOM_COLOR
    return ZONE_MID_COLOR


def make_plot(d, out_path):
    history, points = d["history"], d["points"]
    teams = [t for t in d["teams"] if history.get(t)]
    teams.sort(key=lambda t: -points[t])
    n = len(teams)

    gaps = []
    for t in teams:
        h = history[t][-1]
        gaps.append(pil.ELO_SCALE * (h["home_rating"] - h["away_rating"]))

    fig, ax = plt.subplots(figsize=(13, 0.62 * n + 2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y_pos = list(range(n))[::-1]
    for y, gap in zip(y_pos, gaps):
        if gap >= GAP_THRESHOLD:
            color = GREEN
        elif gap <= -GAP_THRESHOLD:
            color = RED
        else:
            color = NEUTRAL
        ax.barh(y, gap, color=color, height=0.62, zorder=3)
        label_x = gap + (2.5 if gap >= 0 else -2.5)
        ha = "left" if gap >= 0 else "right"
        label_color = TEXT_MAIN if abs(gap) >= GAP_THRESHOLD else TEXT_SUB
        ax.text(label_x, y, f"{gap:+.0f}", va="center", ha=ha, fontsize=10.5,
                color=label_color, fontweight="bold")

    ax.set_yticks(y_pos)
    labels = []
    for i, t in enumerate(teams):
        rank = i + 1
        labels.append(f"#{rank}  {pil.clean_name(t)}")
    # capture the return value directly -- get_yticklabels() can hand back
    # ticks re-sorted by y-value, which silently mismatches label->rank
    tick_labels = ax.set_yticklabels(labels, fontsize=10.5)
    for i, tick in enumerate(tick_labels):
        rank = i + 1
        tick.set_color(zone_color(rank, n))

    ax.axvline(0, color=TEXT_SUB, linewidth=1.0, alpha=0.7, zorder=2)
    # split by axis: tick_params(colors=...) would overwrite the per-team
    # zone colors just set on the y tick labels above
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Home PI ELO − Away PI ELO", fontsize=11, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_ylim(-0.7, n - 0.3)
    pad = max(15.0, (max(gaps) - min(gaps)) * 0.12)
    ax.set_xlim(min(gaps) - pad, max(gaps) + pad)

    fig.text(0.05, 0.975, "Ecuador 2026  ·  All Teams  ·  Home vs Away PI Rating Gap",
             fontsize=19, fontweight="bold", color="white")
    fig.text(0.05, 0.945, "Positive = stronger at home  ·  Negative = road warrior",
             fontsize=10.5, color=TEXT_SUB)
    fig.text(0.05, 0.008, "Data via Opta | Ecuador 2026 event data · Pi-rating (Constantinou & Fenton "
             "structure), reimplemented for this dataset", fontsize=8, color="#6b7684")
    fig.text(0.98, 0.008, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.30, right=0.94, top=0.90, bottom=0.06)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/home_away_rating_gap.png"
    d = pil.load_all()
    make_plot(d, out)
