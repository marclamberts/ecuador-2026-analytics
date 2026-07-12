"""
Opponent-Adjusted xG leaderboard: each shot's real xG (Danger CSVs, mean
of 5 calibrated shot models) is scaled by the defensive strength of the
team that conceded it -- shots against a tough defense count for more,
shots against a leaky defense count for less. Defensive strength is the
same percentile-ranked xG-against-per-game measure used in
attack_defense_quadrant.py. League-wide, minimum 600 minutes played.

Multiplier = 0.7 + (opponent defense percentile / 100) * 0.6
  -> 0.7x for the league's weakest defense faced, 1.3x for the strongest,
     1.0x at a league-average opponent.

Usage: python3 player_adjusted_xg.py [out.png] [top_n]
"""
import sys
import glob
import collections

import pandas as pd
import matplotlib.pyplot as plt

import pi_ratings_lib as pil

DANGER_DIR = "/Users/marclamberts/Event data/Ecuador 2026/Danger"
AGG_DIR = "/Users/marclamberts/Event data/Ecuador 2026/Aggregated"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
C_AMBER = "#ffc247"
C_RAW = "#5b6472"

MIN_MINUTES = 600
MULT_LO, MULT_HI = 0.7, 1.3


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


def team_defense_ranks(team_to_cid):
    cid_to_team = {v: k for k, v in team_to_cid.items()}
    xg_against = collections.defaultdict(float)
    games = collections.defaultdict(int)

    for path in sorted(glob.glob(f"{DANGER_DIR}/*_danger_models.csv")):
        df = pd.read_csv(path)
        cids = df["contestant_id"].unique()
        teams_here = [cid_to_team[c] for c in cids if c in cid_to_team]
        if len(teams_here) != 2:
            continue
        t0, t1 = teams_here
        xg0 = df.loc[df["contestant_id"] == team_to_cid[t0], "xg"].sum()
        xg1 = df.loc[df["contestant_id"] == team_to_cid[t1], "xg"].sum()
        xg_against[t0] += xg1
        xg_against[t1] += xg0
        games[t0] += 1
        games[t1] += 1

    xg_against_pg = {t: xg_against[t] / games[t] for t in games if games[t] > 0}
    return percentile_rank({t: -v for t, v in xg_against_pg.items()})


def collect_shots(team_to_cid, defense_rank):
    cid_to_team = {v: k for k, v in team_to_cid.items()}
    rows = []
    for path in sorted(glob.glob(f"{DANGER_DIR}/*_danger_models.csv")):
        df = pd.read_csv(path)
        cids = df["contestant_id"].unique()
        teams_here = [cid_to_team[c] for c in cids if c in cid_to_team]
        if len(teams_here) != 2:
            continue
        t0, t1 = teams_here
        opp_of = {team_to_cid[t0]: t1, team_to_cid[t1]: t0}
        for _, row in df.iterrows():
            shooter_team = cid_to_team.get(row["contestant_id"])
            opp_team = opp_of.get(row["contestant_id"])
            if shooter_team is None or opp_team is None:
                continue
            drank = defense_rank.get(opp_team, 50.0)
            mult = MULT_LO + (drank / 100) * (MULT_HI - MULT_LO)
            rows.append({
                "player_id": row["player_id"], "player": row["player_name"],
                "team": shooter_team, "xg": row["xg"], "adj_xg": row["xg"] * mult,
            })
    return pd.DataFrame(rows)


def make_plot(d, out_path, top_n):
    team_to_cid = d["team_to_cid"]
    defense_rank = team_defense_ranks(team_to_cid)
    shots = collect_shots(team_to_cid, defense_rank)

    per_player = shots.groupby(["player_id", "player", "team"], as_index=False).agg(
        xg=("xg", "sum"), adj_xg=("adj_xg", "sum"), shots=("xg", "size"))

    minutes = pd.read_csv(f"{AGG_DIR}/player_match_metrics.csv").groupby(
        "player_id", as_index=False)["minutes"].sum()
    per_player = per_player.merge(minutes, on="player_id", how="left")
    per_player["minutes"] = per_player["minutes"].fillna(0)
    per_player = per_player[per_player["minutes"] >= MIN_MINUTES]
    per_player = per_player.sort_values("adj_xg", ascending=False).head(top_n)
    per_player = per_player.iloc[::-1].reset_index(drop=True)

    n = len(per_player)
    fig, ax = plt.subplots(figsize=(13.5, 0.5 * n + 2.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y_pos = list(range(n))
    ax.barh(y_pos, per_player["adj_xg"], color=C_AMBER, height=0.6, zorder=3, alpha=0.92)
    ax.scatter(per_player["xg"], y_pos, color=C_RAW, s=55, marker="D", zorder=5,
              edgecolors="white", linewidths=0.8, label="Raw xG")

    for y, adj, raw in zip(y_pos, per_player["adj_xg"], per_player["xg"]):
        label_x = max(adj, raw) + 0.22
        ax.text(label_x, y, f"{adj:.1f}", va="center", ha="left", fontsize=9.5,
                color=TEXT_MAIN, fontweight="bold")

    labels = [f"{row.player}  ·  {pil.clean_name(row.team)}" for row in per_player.itertuples()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", colors=TEXT_MAIN, length=0)
    ax.set_xlabel("Opponent-adjusted xG  (◆ = raw xG for reference)", fontsize=10.5, color=TEXT_MAIN,
                 fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_xlim(0, per_player["adj_xg"].max() * 1.18)
    ax.set_ylim(-0.6, n - 0.4)

    fig.text(0.05, 0.975, "Ecuador 2026  ·  Opponent-Adjusted xG — Top Attackers",
             fontsize=19, fontweight="bold", color="white")
    fig.text(0.05, 0.953, f"Shots scaled by opponent's defensive strength faced  ·  "
             f"Min. {MIN_MINUTES} minutes played  ·  League-wide", fontsize=10.5, color=TEXT_SUB)
    fig.text(0.05, 0.020, f"Data via Opta | Ecuador 2026 event data · xg = mean of 5 calibrated shot "
             f"models · Multiplier = {MULT_LO:.1f}x (weakest defense faced) to {MULT_HI:.1f}x "
             "(strongest), 1.0x at league-average opponent", fontsize=7.8, color="#6b7684")
    fig.text(0.05, 0.005, "Defensive strength = percentile rank of opponent's season xG-against per "
             "game (real shot-level xG)", fontsize=7.8, color="#6b7684")
    fig.text(0.98, 0.013, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.37, right=0.95, top=0.905, bottom=0.09)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/player_adjusted_xg.png"
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    d = pil.load_all()
    make_plot(d, out, top_n)
