"""
Aerial Duels comparison chart (vs same-position peers) for the Ecuador 2026 dataset.

Usage: python3 aerial_duels.py "E. Mero"
"""
import glob
import json
import re
import sys
import collections

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mplsoccer import VerticalPitch

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


DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
BG = "#0d1117"

POS_LABEL = {"1": "GK", "2": "DEF", "3": "MID", "4": "FWD"}
MIN_MINUTES = 270  # ~3 full matches, to keep the comparison sample sane


def build_team_map(files):
    team_cid_sets = collections.defaultdict(list)
    for fn in files:
        m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", fn.split("/")[-1])
        if not m:
            continue
        home, away = m.group(1), m.group(2)
        with open(fn) as f:
            data = json.load(f)
        cids = set(e["contestantId"] for e in data["event"] if "contestantId" in e)
        team_cid_sets[home].append(cids)
        team_cid_sets[away].append(cids)
    team_to_cid = {}
    for team, sets in team_cid_sets.items():
        inter = set.intersection(*sets)
        if len(inter) == 1:
            team_to_cid[team] = next(iter(inter))
    return {v: k for k, v in team_to_cid.items()}


def clock_to_elapsed(period_id, time_min, p1_len, p2_len):
    if period_id == 1:
        return min(time_min, p1_len)
    if period_id == 2:
        return p1_len + max(0, time_min - 45)
    return p1_len + p2_len


def process_match(fn, stats, positions, names, team_of):
    with open(fn) as f:
        data = json.load(f)
    events = data["event"]
    periods = data["matchDetails"]["period"]
    p1_len = next((p["lengthMin"] for p in periods if p["id"] == 1), 45)
    p2_len = next((p["lengthMin"] for p in periods if p["id"] == 2), 45)
    match_total = p1_len + p2_len

    starters = {}   # playerId -> True
    for e in events:
        if e["typeId"] != 34:
            continue
        qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
        ids = (qmap.get(30) or "").split(", ")
        codes = (qmap.get(44) or "").split(", ")
        for pid, code in zip(ids, codes):
            if code in POS_LABEL:
                positions[pid][code] += 1
                if code in ("1", "2", "3", "4") and code != "1" and pid not in starters:
                    pass
        for pid, code in zip(ids, codes):
            if code in ("2", "3", "4", "1"):
                starters[pid] = clock_to_elapsed(1, 0, p1_len, p2_len)  # 0 = kickoff start

    on_time = dict(starters)  # playerId -> elapsed minute they entered
    off_time = {}
    for e in events:
        if e["typeId"] == 18:  # player off
            pid = e.get("playerId")
            if pid:
                off_time[pid] = clock_to_elapsed(e["periodId"], e["timeMin"], p1_len, p2_len)
        elif e["typeId"] == 19:  # player on
            pid = e.get("playerId")
            if pid:
                on_time[pid] = clock_to_elapsed(e["periodId"], e["timeMin"], p1_len, p2_len)
        if e.get("playerId") and e.get("playerName"):
            names[e["playerId"]] = e["playerName"]
            team_of[e["playerId"]] = e.get("contestantId")

    for pid, t_on in on_time.items():
        t_off = off_time.get(pid, match_total)
        minutes = max(0.0, t_off - t_on)
        stats[pid]["minutes"] += minutes

    for e in events:
        if e["typeId"] == 44 and e.get("playerId"):
            pid = e["playerId"]
            stats[pid]["aerials"] += 1
            if e["outcome"] == 1:
                stats[pid]["won"] += 1
            else:
                stats[pid]["lost"] += 1


def build_dataset():
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    cid_to_team = build_team_map(files)
    stats = collections.defaultdict(lambda: {"minutes": 0.0, "aerials": 0, "won": 0, "lost": 0})
    positions = collections.defaultdict(collections.Counter)
    names = {}
    team_of = {}
    for fn in files:
        process_match(fn, stats, positions, names, team_of)

    rows = []
    for pid, s in stats.items():
        if s["minutes"] <= 0:
            continue
        pos_counter = positions.get(pid)
        if not pos_counter:
            continue
        pos_code = pos_counter.most_common(1)[0][0]
        nineties = s["minutes"] / 90
        rows.append({
            "player_id": pid,
            "name": names.get(pid, "?"),
            "team": cid_to_team.get(team_of.get(pid), "Unknown"),
            "position": POS_LABEL[pos_code],
            "minutes": s["minutes"],
            "nineties": nineties,
            "aerials_p90": s["aerials"] / nineties if nineties else 0,
            "won_p90": s["won"] / nineties if nineties else 0,
            "lost_p90": s["lost"] / nineties if nineties else 0,
            "win_pct": (s["won"] / s["aerials"] * 100) if s["aerials"] else 0,
        })
    return rows


def percentile_of(value, sample):
    sample = np.asarray(sample)
    return float((sample < value).sum() / len(sample) * 100)


def half_violin(ax, sample, y0, color, height=0.8):
    sample = np.asarray(sample)
    if len(sample) < 3 or np.std(sample) == 0:
        return
    kde = gaussian_kde(sample)
    xs = np.linspace(sample.min(), sample.max(), 300)
    dens = kde(xs)
    dens = dens / dens.max() * (height / 2)
    ax.fill_between(xs, y0 - dens, y0 + dens, color=color, alpha=0.18, linewidth=0, zorder=1)
    ax.plot(xs, y0 + dens, color=color, alpha=0.35, linewidth=0.8, zorder=1)
    ax.plot(xs, y0 - dens, color=color, alpha=0.35, linewidth=0.8, zorder=1)
    median = np.median(sample)
    ax.plot([median, median], [y0 - height * 0.28, y0 + height * 0.28],
            color="#7b8794", linewidth=1.4, zorder=2)


def make_plot(player_name, shots_events, rows, out_path):
    me = next((r for r in rows if r["name"] == player_name), None)
    if me is None:
        raise SystemExit(f"'{player_name}' not found among players with recorded minutes.")

    peers = [r for r in rows if r["position"] == me["position"] and r["minutes"] >= MIN_MINUTES]
    if me not in peers:
        peers.append(me)

    metrics = [
        ("aerials_p90", "AERIAL DUELS PER 90", "{:.2f}"),
        ("won_p90", "AERIAL DUELS WON PER 90", "{:.2f}"),
        ("lost_p90", "AERIAL DUELS LOST PER 90", "{:.2f}"),
        ("win_pct", "AERIAL DUEL %", "{:.2f}"),
    ]

    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor(BG)

    # -------- left: pitch with aerial events --------
    pitch_ax = fig.add_axes([0.02, 0.06, 0.34, 0.84])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color="#3a4a5c",
                           linewidth=1.2, half=False)
    pitch.draw(ax=pitch_ax)

    n_won = n_lost = 0
    for e in shots_events:
        if e["outcome"] == 1:
            pitch.scatter(e["x"], e["y"], ax=pitch_ax, s=70, color=C_NAVY,
                          edgecolors="white", linewidths=0.8, marker="o", zorder=3, alpha=0.85)
            n_won += 1
        else:
            pitch.scatter(e["x"], e["y"], ax=pitch_ax, s=70, color=C_PINK,
                          marker="x", linewidths=2.0, zorder=3, alpha=0.85)
            n_lost += 1

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_NAVY, markeredgecolor="white",
               markersize=10, label=f"Aerial Won: {n_won}"),
        Line2D([0], [0], marker="x", color=C_PINK, markersize=10, markeredgewidth=2.2,
               label=f"Aerial Lost: {n_lost}"),
    ]
    pitch_ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                     ncol=2, frameon=False, fontsize=10)

    # -------- right: violin comparison panels --------
    rx0, rx1 = 0.42, 0.97
    panel_h = 0.205
    top = 0.90
    for i, (key, label, fmt) in enumerate(metrics):
        y_top = top - i * panel_h
        ax = fig.add_axes([rx0, y_top - panel_h + 0.06, rx1 - rx0, panel_h - 0.09])
        ax.set_facecolor(BG)
        sample = [p[key] for p in peers]
        color = C_PURPLE
        half_violin(ax, sample, 0, color)
        val = me[key]
        pct = percentile_of(val, sample)
        ax.scatter([val], [0], s=140, color=C_CORAL, edgecolors="white",
                   linewidths=1.2, zorder=5)
        ax.plot([val, val], [-0.05, 0.42], color=C_CORAL, linewidth=2.2, zorder=4)

        lo, hi = min(sample + [val]), max(sample + [val])
        pad = (hi - lo) * 0.08 if hi > lo else 1
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#6b7684")
        ax.tick_params(axis="x", labelsize=9, colors="#b7bec8")

        fig.text(rx0, y_top + 0.012, label, fontsize=11.5, fontweight="bold", color="#d7dbe0")
        fig.text(rx0 + 0.235, y_top + 0.012, fmt.format(val), fontsize=13, fontweight="bold", color=C_CORAL)
        fig.text(rx0 + 0.30, y_top + 0.013, f"{pct:.0f}th pctile vs {me['position']}s",
                 fontsize=9.5, color="#9aa4b2")

    fig.text(0.02, 0.965, player_name, fontsize=30, fontweight="bold", family="sans-serif", color="#ffffff")
    fig.text(0.02, 0.932, "Aerial Duels compared to " + me["position"] + "s", fontsize=14,
             fontweight="bold", color="#d7dbe0")
    fig.text(0.02, 0.905, f"{me['team']} · Ecuador 2026 · All Competitions · "
             f"{me['minutes']:.0f} min ({me['nineties']:.1f} 90s)", fontsize=11, color="#9aa4b2")

    fig.text(0.02, 0.02, f"Ecuador 2026 · {n_won} won · {n_lost} lost · "
             f"{n_won / (n_won + n_lost) * 100 if n_won + n_lost else 0:.0f}% win rate",
             fontsize=9, color="#7b8794")
    fig.text(0.97, 0.02,
             f"Data via Opta | vs {len(peers)} {me['position']}s with ≥{MIN_MINUTES} min | Ecuador 2026 event data",
             fontsize=8.5, color="#7b8794", ha="right")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    print(f"{player_name}: pos={me['position']} minutes={me['minutes']:.0f} "
          f"aerials/90={me['aerials_p90']:.2f} won/90={me['won_p90']:.2f} "
          f"lost/90={me['lost_p90']:.2f} win%={me['win_pct']:.1f} peers_n={len(peers)}")


def collect_player_aerials(player_name):
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    events = []
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e.get("playerName") == player_name and e["typeId"] == 44:
                events.append(e)
    return events


if __name__ == "__main__":
    player = sys.argv[1] if len(sys.argv) > 1 else "E. Mero"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/aerial_duels_{player.replace(' ', '_')}.png"
    rows = build_dataset()
    events = collect_player_aerials(player)
    make_plot(player, events, rows, out)
