"""
Pi-rating progression across the season, raw (non-rescaled) values --
combined rating = (home rating + away rating) / 2, starting at 0.

Usage: python3 pi_rating_progression_raw.py [out.png] [top_n]
"""
import sys
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pi_ratings_lib as pil

LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
GREY = "#3a4658"

HIGHLIGHT_COLORS = ["#ffc247", "#2f8fd1", "#ff8a75", "#7b7fd6", "#4ade80", "#f06fa3"]


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


def make_plot(d, out_path, top_n=6):
    history, teams = d["history"], d["teams"]
    final_combined = {t: history[t][-1]["combined"] for t in teams if history.get(t)}
    ranked = sorted(final_combined, key=lambda t: -final_combined[t])
    top_teams = ranked[:top_n]
    other_teams = ranked[top_n:]

    fig, ax = plt.subplots(figsize=(15, 9.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.axhline(0, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (2, 3)), alpha=0.5, zorder=1)

    for t in other_teams:
        hist = history[t]
        xs = [dt.datetime.strptime(h["date"], "%Y-%m-%d") for h in hist]
        ys = [h["combined"] for h in hist]
        ax.plot(xs, ys, color=GREY, linewidth=0.9, alpha=0.55, zorder=2)

    label_specs = []
    for i, t in enumerate(top_teams):
        hist = history[t]
        xs = [dt.datetime.strptime(h["date"], "%Y-%m-%d") for h in hist]
        ys = [h["combined"] for h in hist]
        color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
        ax.plot(xs, ys, color=color, linewidth=2.6, alpha=0.95, zorder=3,
                label=pil.clean_name(t))
        label_specs.append({"x": xs[-1], "y": ys[-1], "color": color,
                            "text": f"{pil.clean_name(t)}  {ys[-1]:.2f}"})

    label_specs.sort(key=lambda s: -s["y"])
    min_gap = 0.14
    for i in range(1, len(label_specs)):
        if label_specs[i - 1]["y"] - label_specs[i]["y"] < min_gap:
            label_specs[i]["y"] = label_specs[i - 1]["y"] - min_gap
    for spec in label_specs:
        ax.text(spec["x"] + dt.timedelta(days=2), spec["y"], spec["text"],
                color=spec["color"], fontsize=10.5, fontweight="bold", va="center")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(colors=TEXT_SUB, labelsize=10)
    ax.set_ylabel("PI Rating (combined)", fontsize=11.5, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6)

    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", frameon=False, fontsize=10.5,
              labelcolor=TEXT_MAIN, ncol=3, bbox_to_anchor=(0.44, 0.855))

    fig.text(0.06, 0.965, "Ecuador 2026  ·  PI Ratings — Team Strength Progression",
             fontsize=20, fontweight="bold", color="white")
    fig.text(0.06, 0.928, f"Constantinou & Fenton PI-rating model  ·  λ={pil.LAMBDA}, γ={pil.GAMMA}  ·  "
             "Updates after every match", fontsize=11, color=TEXT_SUB)
    fig.text(0.06, 0.905, "Combined rating = (home rating + away rating) / 2  ·  Ratings start at 0",
             fontsize=9, color="#6b7684")
    fig.text(0.06, 0.015, "Data via Opta | Ecuador 2026 event data · Pi-rating (Constantinou & Fenton "
             "structure), reimplemented for this dataset", fontsize=8, color="#6b7684")
    fig.text(0.98, 0.015, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.06, right=0.79, top=0.76, bottom=0.08)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pi_rating_progression_raw.png"
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    d = pil.load_all()
    make_plot(d, out, top_n)
