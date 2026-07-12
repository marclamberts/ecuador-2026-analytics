"""
Radar+Distribution comparison chart and Pizza (role template) chart,
built from the pre-aggregated Ecuador 2026 metrics dataset.

Usage: python3 player_templates.py "E. Mero"
"""
import glob
import json
import re
import sys
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mplsoccer import Radar, PyPizza

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


AGG_DIR = "/Users/marclamberts/Event data/Ecuador 2026/Aggregated"
DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
BG = "#0d1117"

MIN_MINUTES = 270
PEN_XG = 0.79
POS_LABEL = {"1": "GK", "2": "DEF", "3": "MID", "4": "FWD"}

RAW_SUM_COLS = [
    "minutes", "matches", "non_penalty_goals", "xg", "shots", "set_piece_shots",
    "psxg_proxy", "key_passes", "xa", "progressive_passes", "progressive_carries",
    "box_touches", "successful_take_ons", "losses", "def_actions", "penalty_shots",
    "aerials", "aerials_won",
]


# ---------------------------------------------------------------- position --

def infer_positions():
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    positions = collections.defaultdict(collections.Counter)
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e["typeId"] != 34:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            ids = (qmap.get(30) or "").split(", ")
            codes = (qmap.get(44) or "").split(", ")
            for pid, code in zip(ids, codes):
                if code in POS_LABEL:
                    positions[pid][code] += 1
    out = {}
    for pid, counter in positions.items():
        out[pid] = POS_LABEL[counter.most_common(1)[0][0]]
    return out


# ------------------------------------------------------------- aggregation --

def load_player_table():
    df = pd.read_csv(f"{AGG_DIR}/player_season_metrics.csv")
    positions = infer_positions()
    df["position"] = df["player_id"].map(positions)

    grouped = df.groupby(["player_id", "player"], as_index=False)[RAW_SUM_COLS].sum()
    grouped["position"] = grouped["player_id"].map(positions)
    grouped["team"] = grouped["player_id"].map(
        df.sort_values("minutes", ascending=False).drop_duplicates("player_id").set_index("player_id")["team"]
    )

    n = grouped["minutes"].replace(0, np.nan) / 90
    grouped["non_penalty_goals_p90"] = grouped["non_penalty_goals"] / n
    grouped["xg_p90"] = grouped["xg"] / n
    grouped["non_penalty_xg"] = grouped["xg"] - grouped["penalty_shots"] * PEN_XG
    grouped["non_penalty_xg_p90"] = grouped["non_penalty_xg"] / n
    grouped["open_play_shots"] = grouped["shots"] - grouped["set_piece_shots"]
    grouped["open_play_shots_p90"] = grouped["open_play_shots"] / n
    grouped["shots_p90"] = grouped["shots"] / n
    grouped["np_xg_per_shot"] = grouped["non_penalty_xg"] / grouped["open_play_shots"].replace(0, np.nan)
    grouped["psxg_proxy_p90"] = grouped["psxg_proxy"] / n
    grouped["key_passes_p90"] = grouped["key_passes"] / n
    grouped["xa_p90"] = grouped["xa"] / n
    grouped["progressive_passes_p90"] = grouped["progressive_passes"] / n
    grouped["progressive_carries_p90"] = grouped["progressive_carries"] / n
    grouped["box_touches_p90"] = grouped["box_touches"] / n
    grouped["take_ons_completed_p90"] = grouped["successful_take_ons"] / n
    grouped["turnovers_p90"] = grouped["losses"] / n
    grouped["def_actions_p90"] = grouped["def_actions"] / n
    grouped["aerials_won_p90"] = grouped["aerials_won"] / n
    grouped["nineties"] = n

    grouped = grouped.replace([np.inf, -np.inf], np.nan)
    return grouped


METRICS = [
    ("non_penalty_goals_p90", "Non-Penalty\nGoals", "{:.2f}"),
    ("non_penalty_xg_p90", "Non-Penalty\nxG", "{:.2f}"),
    ("open_play_shots_p90", "Open Play\nShots", "{:.2f}"),
    ("np_xg_per_shot", "NP xG\nper Shot", "{:.2f}"),
    ("psxg_proxy_p90", "Post-Shot\nxG", "{:.2f}"),
    ("key_passes_p90", "Key\nPasses", "{:.2f}"),
    ("xa_p90", "xA", "{:.2f}"),
    ("progressive_passes_p90", "Progressive\nPasses", "{:.2f}"),
    ("progressive_carries_p90", "Progressive\nCarries", "{:.2f}"),
    ("box_touches_p90", "Box\nTouches", "{:.2f}"),
    ("take_ons_completed_p90", "Dribbles\nCompleted", "{:.2f}"),
    ("turnovers_p90", "Turnovers", "{:.2f}"),
    ("def_actions_p90", "Defensive\nActions", "{:.2f}"),
]


def percentile_of(value, sample):
    sample = np.asarray(sample, dtype=float)
    sample = sample[~np.isnan(sample)]
    if len(sample) == 0 or np.isnan(value):
        return 50.0
    return float((sample < value).sum() / len(sample) * 100)


# ---------------------------------------------------------- chart 1: radar --

def make_radar_distribution(player_name, grouped, out_path):
    me_rows = grouped[grouped["player"] == player_name]
    if me_rows.empty:
        raise SystemExit(f"'{player_name}' not found.")
    me = me_rows.iloc[0]
    peers = grouped[(grouped["position"] == me["position"]) & (grouped["minutes"] >= MIN_MINUTES)]
    if me["player_id"] not in peers["player_id"].values:
        peers = pd.concat([peers, me_rows])

    vals, samples, labels, pcts = [], [], [], []
    for key, label, fmt in METRICS:
        sample = peers[key].dropna().values
        v = me[key]
        if np.isnan(v):
            v = 0.0
        vals.append(v)
        samples.append(sample)
        labels.append(label.replace("\n", " "))
        pcts.append(percentile_of(v, sample))

    low = [min(s.min(), v) * 0.9 if len(s) else 0 for s, v in zip(samples, vals)]
    high = [max(s.max(), v) * 1.08 if len(s) else v * 1.1 or 1 for s, v in zip(samples, vals)]

    fig = plt.figure(figsize=(17, 11))
    fig.patch.set_facecolor(BG)

    lower_is_better = [label for (key, _, _), label in zip(METRICS, labels) if key == "turnovers_p90"]
    radar = Radar(labels, low, high, lower_is_better=lower_is_better,
                   num_rings=4, ring_width=1, center_circle_radius=1)
    rad_ax = fig.add_axes([0.02, 0.06, 0.46, 0.82])
    rad_ax.set_facecolor(BG)
    radar.setup_axis(ax=rad_ax, facecolor=BG)
    radar.draw_circles(ax=rad_ax, facecolor="#161b22", edgecolor="#2c3540")
    radar.draw_radar(vals, ax=rad_ax,
                      kwargs_radar={"facecolor": C_INDIGO, "alpha": 0.55, "edgecolor": C_CORAL,
                                     "linewidth": 2.2, "hatch": "//"},
                      kwargs_rings={"facecolor": C_PURPLE, "alpha": 0.12})
    radar.draw_range_labels(ax=rad_ax, fontsize=8.5, color="#b7bec8")
    radar.draw_param_labels(ax=rad_ax, fontsize=10.5, color="#f0f2f5", fontweight="bold")

    fig.text(0.02, 0.965, player_name, fontsize=30, fontweight="bold", family="sans-serif", color="#ffffff")
    fig.text(0.02, 0.928, f"{me['team']} · Ecuador 2026", fontsize=13, color="#9aa4b2")
    fig.text(0.02, 0.900, f"{me['nineties']:.1f} 90s played · All Competitions", fontsize=11, color="#7b8794")
    small_sample = " · SMALL SAMPLE SIZE — USE WITH CAUTION" if me["nineties"] < 5 else ""
    fig.text(0.02, 0.865, f"{me['position']} TEMPLATE" + small_sample, fontsize=12, fontweight="bold", color=C_CORAL)

    # distribution comparison panel
    rx0, rx1 = 0.53, 0.985
    fig.text(rx0, 0.965, "DISTRIBUTION COMPARISON", fontsize=15, fontweight="bold", color="#ffffff")
    fig.text(rx0, 0.935, f"vs {me['position']}s · {len(peers)} players (≥{MIN_MINUTES} min)", fontsize=11, color="#9aa4b2")

    n_m = len(METRICS)
    top, bottom = 0.905, 0.03
    row_h = (top - bottom) / n_m
    for i, ((key, label, fmt), v, sample, pct) in enumerate(zip(METRICS, vals, samples, pcts)):
        y_top = top - i * row_h
        ax = fig.add_axes([rx0, y_top - row_h + row_h * 0.18, 0.34, row_h * 0.62])
        ax.set_facecolor(BG)
        sample_c = sample[~np.isnan(sample)]
        if len(sample_c) >= 5 and np.std(sample_c) > 0:
            kde = gaussian_kde(sample_c)
            xs = np.linspace(min(sample_c.min(), v), max(sample_c.max(), v), 300)
            dens = kde(xs)
            ax.fill_between(xs, 0, dens, color=C_PURPLE, alpha=0.30, linewidth=0)
            mask = xs <= v
            ax.fill_between(xs[mask], 0, dens[mask], color=C_PINK, alpha=0.55, linewidth=0)
            ax.plot(xs, dens, color=C_PURPLE, linewidth=0.8, alpha=0.6)
            ymax = dens.max() * 1.15
        else:
            xs = np.array([0, 1])
            ymax = 1
        ax.scatter([v], [0], marker="v", s=90, color=C_CORAL, zorder=5, clip_on=False)
        ax.set_ylim(0, ymax)
        ax.set_xlim(xs.min(), xs.max())
        ax.set_yticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#5b6472")
        ax.tick_params(axis="x", labelsize=8, colors="#9aa4b2")
        lo_x, hi_x = xs.min(), xs.max()
        ax.set_xticks([lo_x, (lo_x + hi_x) / 2, hi_x])
        ax.set_xticklabels([f"{lo_x:.2g}", f"{(lo_x + hi_x) / 2:.2g}", f"{hi_x:.2g}"])

        fig.text(rx0, y_top - 0.006, label.replace("\n", " "), fontsize=10.5, fontweight="bold", color="#f0f2f5", va="top")
        pct_color = C_NAVY if pct >= 66 else (C_AMBER if pct >= 33 else C_PINK)
        fig.text(rx0 + 0.365, y_top - row_h * 0.5, fmt.format(v), fontsize=12.5, fontweight="bold",
                 color="#ffffff", ha="left", va="center")
        fig.text(rx0 + 0.40, y_top - row_h * 0.5, f"{pct:.0f}", fontsize=12.5, fontweight="bold",
                 color=pct_color, ha="left", va="center")

    fig.text(rx0 + 0.365, 0.985, "VALUE", fontsize=9, color="#7b8794", fontweight="bold")
    fig.text(rx0 + 0.40, 0.985, "%ILE", fontsize=9, color="#7b8794", fontweight="bold")

    fig.text(0.02, 0.012, "Data via Opta | Ecuador 2026 event data", fontsize=8.5, color="#7b8794")
    fig.text(0.985, 0.012,
             "Non-penalty xG approximated as xG − penalty shots×0.79. Position inferred from formation slots.",
             fontsize=7.5, color="#6b7684", ha="right")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


# ---------------------------------------------------------- chart 2: pizza --

PIZZA_METRICS = [
    ("open_play_shots_p90", "Shots", "shoot"),
    ("non_penalty_xg_p90", "Non-Pen xG", "shoot"),
    ("non_penalty_goals_p90", "Non-Pen Goals", "shoot"),
    ("np_xg_per_shot", "xG per Shot", "shoot"),
    ("key_passes_p90", "Key Passes", "create"),
    ("xa_p90", "xA", "create"),
    ("psxg_proxy_p90", "Post-Shot xG", "create"),
    ("box_touches_p90", "Box Touches", "create"),
    ("progressive_passes_p90", "Progressive Passes", "prog"),
    ("progressive_carries_p90", "Progressive Carries", "prog"),
    ("take_ons_completed_p90", "Dribbles Completed", "prog"),
    ("aerials_won_p90", "Aerials Won", "prog"),
    ("def_actions_p90", "Defensive Actions", "def"),
    ("turnovers_p90", "Turnovers (inv.)", "def"),
]

CAT_COLOR = {"shoot": C_NAVY, "create": C_PURPLE, "prog": C_INDIGO, "def": C_PINK}


def make_pizza(player_name, grouped, out_path):
    me_rows = grouped[grouped["player"] == player_name]
    me = me_rows.iloc[0]
    peers = grouped[(grouped["position"] == me["position"]) & (grouped["minutes"] >= MIN_MINUTES)]
    if me["player_id"] not in peers["player_id"].values:
        peers = pd.concat([peers, me_rows])

    params, values, alt_texts, slice_colors = [], [], [], []
    for key, label, cat in PIZZA_METRICS:
        sample = peers[key].dropna().values
        v = me[key]
        if np.isnan(v):
            v = 0.0
        if key == "turnovers_p90":
            pct = 100 - percentile_of(v, sample)
        else:
            pct = percentile_of(v, sample)
        params.append(label)
        values.append(round(pct, 1))
        alt_texts.append(f"{v:.2f}")
        slice_colors.append(CAT_COLOR[cat])

    baker = PyPizza(
        params=params, min_range=None, max_range=None,
        background_color=BG, straight_line_color="#2c3540", straight_line_lw=1,
        last_circle_color="#6b7684", last_circle_lw=1.5,
        other_circle_color="#2c3540", other_circle_lw=1, other_circle_ls="--",
    )

    fig, ax = baker.make_pizza(
        values, alt_text_values=alt_texts, figsize=(13, 14.5),
        param_location=112, slice_colors=slice_colors,
        value_bck_colors=slice_colors, value_colors=["#ffffff"] * len(params),
        blank_alpha=0.35,
        kwargs_slices=dict(edgecolor=BG, linewidth=2, zorder=2),
        kwargs_params=dict(color="#f0f2f5", fontsize=10.5, fontweight="bold", va="center"),
        kwargs_values=dict(color="#ffffff", fontsize=10.5, fontweight="bold", zorder=3,
                            bbox=dict(edgecolor="none", boxstyle="round,pad=0.25", lw=1)),
    )
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.975, player_name, fontsize=26, fontweight="bold", ha="center", color="#ffffff")
    fig.text(0.5, 0.945, f"{me['team']} · Ecuador 2026", fontsize=13, ha="center", color="#9aa4b2")
    fig.text(0.5, 0.920, f"{me['nineties']:.1f} 90s played · All Competitions", fontsize=10.5, ha="center", color="#7b8794")
    small_sample = me["nineties"] < 5
    if small_sample:
        fig.text(0.5, 0.895, "SMALL SAMPLE SIZE — USE WITH CAUTION", fontsize=11, fontweight="bold",
                 ha="center", color=C_PINK)
    fig.text(0.5, 0.868 if small_sample else 0.895, f"{me['position']} TEMPLATE", fontsize=12,
             fontweight="bold", ha="center", color=C_CORAL)

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=C_NAVY, label="Shooting"),
                    Patch(facecolor=C_PURPLE, label="Creating"),
                    Patch(facecolor=C_INDIGO, label="Progressing"),
                    Patch(facecolor=C_PINK, label="Defending")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.012), fontsize=10.5)

    fig.text(0.02, 0.012, "Data via Opta", fontsize=8, color="#7b8794")
    fig.text(0.985, 0.012, f"vs {len(peers)} {me['position']}s (≥{MIN_MINUTES} min) | Ecuador 2026 event data",
             fontsize=8, color="#7b8794", ha="right")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    player = sys.argv[1] if len(sys.argv) > 1 else "E. Mero"
    out_radar = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/radar_dist_{player.replace(' ', '_')}.png"
    out_pizza = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/pizza_{player.replace(' ', '_')}.png"
    grouped = load_player_table()
    make_radar_distribution(player, grouped, out_radar)
    make_pizza(player, grouped, out_pizza)
