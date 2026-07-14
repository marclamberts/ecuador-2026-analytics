"""
Receiving pass quality: classifies every completed pass reception as
OPEN / HALF / CLOSED based on how much the receiver has to change
direction with their first action after the ball arrives.

There is no tracking data (no defender positions, no body-orientation
tags) in this Opta/StatsPerform feed, so this is a proxy built entirely
from event coordinates:

  - The receiver is found with the same "next same-team touch within a
    short window" heuristic the core pipeline uses for received_passes
    (see ecuador_2026_expanded_metrics_one_cell.ipynb).
  - The incoming vector is the pass's own direction (start -> end).
  - The outgoing vector is the receiver's next on-ball action (pass end,
    or shot direction, or -- if the first touch is a take-on/clearance/
    aerial with no end coordinate -- their following pass/shot).
  - The angle between the two vectors buckets the reception:
      < 60 deg   -> OPEN   (kept the same direction, received facing play)
      60-120 deg -> HALF   (redirected at an angle, side-on / half-turn)
      >= 120 deg -> CLOSED (had to turn away / play backward)
    An immediate Dispossessed (typeId 50) on the first touch is always
    CLOSED regardless of angle -- there was no time to do anything else.
  - If no outgoing vector can be found at all, the reception is HALF
    (indeterminate) rather than guessed in either direction.

Usage: python3 receiving_pass_quality.py [out.png] [top_n]
Writes Aggregated/receiving_pass_quality.csv and a leaderboard PNG.
"""
import sys
import glob
import json
import math

import pandas as pd
import matplotlib.pyplot as plt

import pi_ratings_lib as pil

DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"
AGG_DIR = f"{DATA_DIR}/Aggregated"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"

BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
C_NAVY = "#2f8fd1"    # open
C_AMBER = "#ffc247"   # half
C_CORAL = "#ff8a75"   # closed

MIN_MINUTES = 270
MIN_RECEPTIONS = 20

PASS_TYPES = {1, 2}
SHOT_TYPES = {13, 14, 15, 16}
DISPOSSESSED = 50
TOUCH_EXCLUDE = {18, 19, 30, 32, 34, 37, 40, 70, 71, 90, 91}
Q_END_X, Q_END_Y = 140, 141

WINDOW_EVENTS = 8
WINDOW_MIN = 0.18  # ~11 seconds, same window used for received_passes

OPEN_MAX = 60.0
HALF_MAX = 120.0


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


def safe_float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def minute_value(e):
    return e.get("timeMin", 0) + e.get("timeSec", 0) / 60.0


def qual_map(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", [])}


def pass_vector(e):
    qmap = qual_map(e)
    ex = safe_float(qmap.get(Q_END_X), e["x"])
    ey = safe_float(qmap.get(Q_END_Y), e["y"])
    return (ex - e["x"], ey - e["y"])


def outgoing_vector(e):
    tid = e.get("typeId")
    if tid in PASS_TYPES:
        return pass_vector(e)
    if tid in SHOT_TYPES:
        return (100.0 - e["x"], 50.0 - e["y"])
    return None


def vector_angle(v1, v2):
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1.0 or n2 < 1.0:
        return None
    cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def find_receiver(events, i):
    e = events[i]
    t0 = minute_value(e)
    for j in range(i + 1, min(i + 1 + WINDOW_EVENTS, len(events))):
        nxt = events[j]
        if minute_value(nxt) - t0 > WINDOW_MIN:
            break
        if (nxt.get("contestantId") == e.get("contestantId")
                and nxt.get("playerName")
                and nxt.get("playerName") != e.get("playerName")
                and nxt.get("typeId") not in TOUCH_EXCLUDE):
            return j
    return None


def classify_reception(events, pass_idx, rec_idx):
    rec = events[rec_idx]
    receiver, team_cid = rec["playerName"], rec["contestantId"]

    if rec.get("typeId") == DISPOSSESSED:
        return {"receiver": receiver, "team_cid": team_cid, "category": "closed", "angle": None}

    out_vec = outgoing_vector(rec)
    if out_vec is None:
        t0 = minute_value(rec)
        for j in range(rec_idx + 1, min(rec_idx + 1 + WINDOW_EVENTS, len(events))):
            n2 = events[j]
            if minute_value(n2) - t0 > WINDOW_MIN:
                break
            if n2.get("playerName") != receiver or n2.get("contestantId") != team_cid:
                continue
            if n2.get("typeId") == DISPOSSESSED:
                return {"receiver": receiver, "team_cid": team_cid, "category": "closed", "angle": None}
            out_vec = outgoing_vector(n2)
            if out_vec is not None:
                break
        if out_vec is None:
            return {"receiver": receiver, "team_cid": team_cid, "category": "half", "angle": None}

    in_vec = pass_vector(events[pass_idx])
    angle = vector_angle(in_vec, out_vec)
    if angle is None:
        return {"receiver": receiver, "team_cid": team_cid, "category": "half", "angle": None}
    if angle < OPEN_MAX:
        category = "open"
    elif angle < HALF_MAX:
        category = "half"
    else:
        category = "closed"
    return {"receiver": receiver, "team_cid": team_cid, "category": category, "angle": angle}


def collect():
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    rows = []
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        events = data["event"]
        for i, e in enumerate(events):
            if e.get("typeId") != 1 or e.get("outcome") != 1 or not e.get("playerName"):
                continue
            rec_idx = find_receiver(events, i)
            if rec_idx is None:
                continue
            rows.append(classify_reception(events, i, rec_idx))
    df = pd.DataFrame(rows)
    cid_to_team = {v: k for k, v in pil.build_team_map(files).items()}
    df["team"] = df["team_cid"].map(cid_to_team)
    return df.drop(columns="team_cid")


def summarize(df):
    counts = df.groupby(["team", "receiver", "category"]).size().unstack(fill_value=0)
    for c in ("open", "half", "closed"):
        if c not in counts.columns:
            counts[c] = 0
    counts["receptions"] = counts[["open", "half", "closed"]].sum(axis=1)
    for c in ("open", "half", "closed"):
        counts[f"{c}_pct"] = counts[c] / counts["receptions"] * 100
    summary = counts.reset_index().rename(columns={"receiver": "player"})

    core = pd.read_csv(f"{AGG_DIR}/player_season_core.csv")[
        ["team", "player", "player_id", "minutes"]
    ]
    summary = summary.merge(core, on=["team", "player"], how="left")
    for c in ("open", "half", "closed"):
        summary[f"{c}_p90"] = summary[c] / summary["minutes"] * 90

    cols = ["team", "player", "player_id", "minutes", "receptions",
            "open", "half", "closed", "open_pct", "half_pct", "closed_pct",
            "open_p90", "half_p90", "closed_p90"]
    return summary[cols].sort_values("receptions", ascending=False)


def make_plot(summary, out_path, top_n):
    pool = summary[(summary["minutes"] >= MIN_MINUTES) & (summary["receptions"] >= MIN_RECEPTIONS)]
    top = pool.sort_values("open_pct", ascending=False).head(top_n)
    top = top.iloc[::-1].reset_index(drop=True)

    n = len(top)
    fig, ax = plt.subplots(figsize=(13.5, 0.5 * n + 2.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y_pos = list(range(n))
    left = [0.0] * n
    for cat, color, label in (("open", C_NAVY, "Open"), ("half", C_AMBER, "Half"), ("closed", C_CORAL, "Closed")):
        vals = top[f"{cat}_pct"].tolist()
        ax.barh(y_pos, vals, left=left, color=color, height=0.62, zorder=3, label=label)
        left = [l + v for l, v in zip(left, vals)]

    for y, row in zip(y_pos, top.itertuples()):
        ax.text(101.5, y, f"{int(row.receptions)} rec.", va="center", ha="left",
                fontsize=9, color=TEXT_SUB)

    labels = [f"{row.player}  ·  {pil.clean_name(row.team)}" for row in top.itertuples()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", colors=TEXT_MAIN, length=0)
    ax.set_xlabel("Share of receptions  (OPEN = kept direction, HALF = redirected, CLOSED = turned away / dispossessed)",
                  fontsize=10, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.6, n - 0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.012), frameon=False,
              labelcolor=TEXT_MAIN, fontsize=9.5, ncol=3)

    fig.text(0.05, 0.975, "Ecuador 2026  ·  Receiving Pass Quality — Open / Half / Closed",
             fontsize=19, fontweight="bold", color="white")
    fig.text(0.05, 0.953, f"Ranked by share of OPEN receptions  ·  Min. {MIN_MINUTES} minutes, "
             f"{MIN_RECEPTIONS} receptions  ·  League-wide", fontsize=10.5, color=TEXT_SUB)
    fig.text(0.05, 0.020, "Data via Opta | Ecuador 2026 event data · Proxy metric: bucketed by the "
             "angle between the incoming pass and the receiver's next on-ball action "
             "(no tracking/defender data available) · immediate Dispossessed = Closed",
             fontsize=7.8, color="#6b7684")
    fig.text(0.98, 0.013, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.28, right=0.93, top=0.905, bottom=0.09)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/receiving_pass_quality.png"
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    df = collect()
    summary = summarize(df)
    summary.to_csv(f"{AGG_DIR}/receiving_pass_quality.csv", index=False)
    print("Saved:", f"{AGG_DIR}/receiving_pass_quality.csv")

    make_plot(summary, out, top_n)
