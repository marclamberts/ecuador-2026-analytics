"""
Rolling 8-Game xGD -- Form Guide: rolling mean of (xG for - xG against)
per game, chronological, one line per team. Uses real shot-level xG from
the Danger/*_danger_models.csv files (mean of 5 calibrated shot models).

Usage: python3 rolling_xgd_form_guide.py [out.png] [top_n] [window]
"""
import sys
import glob
import collections
import datetime as dt

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pi_ratings_lib as pil

DANGER_DIR = "/Users/marclamberts/Event data/Ecuador 2026/Danger"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
GREY = "#3a4658"

HIGHLIGHT_COLORS = ["#ffc247", "#2f8fd1", "#ff8a75", "#7b7fd6", "#4ade80",
                    "#f06fa3", "#5ec9c9"]


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


def collect_match_xg(matches, team_to_cid):
    """Per-match xG for/against per team, in the chronological order of
    `matches`. Returns {team: [(date, xg_for, xg_against), ...]}."""
    csv_lookup = {}
    for path in glob.glob(f"{DANGER_DIR}/*_danger_models.csv"):
        base = path.split("/")[-1].replace("_danger_models.csv", "")
        csv_lookup[base] = path

    per_team = collections.defaultdict(list)
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
        per_team[h].append((m["date"], xg_h, xg_a))
        per_team[a].append((m["date"], xg_a, xg_h))

    return per_team


def rolling_mean(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def make_plot(d, out_path, top_n=6, window=8):
    matches, team_to_cid, points = d["matches"], d["team_to_cid"], d["points"]
    per_team = collect_match_xg(matches, team_to_cid)

    ranked = sorted(per_team, key=lambda t: -points.get(t, 0))
    top_teams = ranked[:top_n]
    other_teams = ranked[top_n:]

    fig, ax = plt.subplots(figsize=(15, 9.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.axhline(0, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (2, 3)), alpha=0.5, zorder=1)

    series = {}
    for t in ranked:
        games = per_team[t]
        xs = [dt.datetime.strptime(g[0], "%Y-%m-%d") for g in games]
        xgd = [g[1] - g[2] for g in games]
        ys = rolling_mean(xgd, window)
        series[t] = (xs, ys)

    for t in other_teams:
        xs, ys = series[t]
        ax.plot(xs, ys, color=GREY, linewidth=0.9, alpha=0.5, zorder=2)

    label_specs = []
    for i, t in enumerate(top_teams):
        xs, ys = series[t]
        color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
        ax.plot(xs, ys, color=color, linewidth=2.6, alpha=0.95, zorder=3,
                label=pil.clean_name(t))
        ax.fill_between(xs, ys, 0, color=color, alpha=0.08, zorder=2)
        label_specs.append({"x": xs[-1], "y": ys[-1], "color": color,
                            "text": f"{pil.clean_name(t)}  {ys[-1]:+.2f}"})

    label_specs.sort(key=lambda s: -s["y"])
    min_gap = 0.16
    for i in range(1, len(label_specs)):
        if label_specs[i - 1]["y"] - label_specs[i]["y"] < min_gap:
            label_specs[i]["y"] = label_specs[i - 1]["y"] - min_gap
    for spec in label_specs:
        ax.text(spec["x"] + dt.timedelta(days=2), spec["y"], spec["text"],
                color=spec["color"], fontsize=10.5, fontweight="bold", va="center")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", colors=TEXT_SUB, labelsize=10)
    ax.set_ylabel(f"Rolling {window}-game xGD (xG for − xG against)", fontsize=11.5,
                 color=TEXT_MAIN, fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6)

    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", frameon=False, fontsize=10.5,
              labelcolor=TEXT_MAIN, ncol=4, bbox_to_anchor=(0.44, 0.855))

    fig.text(0.06, 0.965, f"Ecuador 2026  ·  Rolling {window}-Game xGD — Form Guide",
             fontsize=20, fontweight="bold", color="white")
    fig.text(0.06, 0.928, "Rolling mean of (xG for − xG against) per match, real shot-level xG  ·  "
             f"Top {top_n} highlighted", fontsize=11, color=TEXT_SUB)
    fig.text(0.06, 0.905, "Positive = creating more/better chances than conceding  ·  Others in grey",
             fontsize=9, color="#6b7684")
    fig.text(0.06, 0.015, "Data via Opta | Ecuador 2026 event data · xg = mean of 5 calibrated shot "
             "models (danger_score methodology)", fontsize=8, color="#6b7684")
    fig.text(0.98, 0.015, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.06, right=0.79, top=0.76, bottom=0.08)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rolling_xgd_form_guide.png"
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    d = pil.load_all()
    make_plot(d, out, top_n, window)
