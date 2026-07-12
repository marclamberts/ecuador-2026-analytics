"""
Final PI rating vs actual points -- one dot per team, does the rating
predict real-world performance?

Usage: python3 pi_rating_vs_points.py [out.png]
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

import pi_ratings_lib as pil

LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
C_AMBER = "#ffc247"
C_NAVY = "#2f8fd1"


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


def make_plot(d, out_path):
    history, teams, points = d["history"], d["teams"], d["points"]
    xs, ys, labels = [], [], []
    for t in teams:
        hist = history.get(t)
        if not hist:
            continue
        xs.append(hist[-1]["combined"])
        ys.append(points[t])
        labels.append(pil.clean_name(t))

    xs, ys = np.array(xs), np.array(ys)
    slope, intercept = np.polyfit(xs, ys, 1)
    y_pred = slope * xs + intercept
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    fig, ax = plt.subplots(figsize=(13, 9.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x_line = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 50)
    ax.plot(x_line, slope * x_line + intercept, color=TEXT_SUB, linewidth=1.5,
            linestyle=(0, (5, 4)), alpha=0.8, zorder=1, label=f"Trend (r²={r2:.2f})")

    ax.scatter(xs, ys, s=200, color=C_AMBER, edgecolors="white", linewidths=1.3,
              alpha=0.92, zorder=3)

    texts = []
    for x, y, label in zip(xs, ys, labels):
        texts.append(ax.text(x, y, label, fontsize=10.5, color=TEXT_MAIN, fontweight="bold", zorder=4))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#4a5568", lw=0.7),
               expand=(1.3, 1.5))

    ax.set_xlabel("PI Rating (final, combined)", fontsize=12, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    ax.set_ylabel("Points", fontsize=12, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    ax.tick_params(colors=TEXT_SUB, labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=11, labelcolor=TEXT_MAIN)

    fig.text(0.06, 0.965, "Ecuador 2026  ·  PI Rating vs Points — All Teams",
             fontsize=20, fontweight="bold", color="white")
    fig.text(0.06, 0.93, "Each dot = one team  ·  Final PI rating vs actual points earned this season",
             fontsize=12, color=TEXT_SUB)
    fig.text(0.06, 0.905, f"r² = {r2:.2f}  ·  PI rating explains {r2*100:.0f}% of points variance",
             fontsize=10.5, color=C_AMBER, fontweight="bold")
    fig.text(0.06, 0.02, "Data via Opta | Ecuador 2026 event data · Pi-rating (Constantinou & Fenton "
             "structure), reimplemented for this dataset", fontsize=8, color="#6b7684")
    fig.text(0.98, 0.02, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.08, right=0.96, top=0.87, bottom=0.09)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"r2={r2:.3f}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pi_rating_vs_points.png"
    d = pil.load_all()
    make_plot(d, out)
