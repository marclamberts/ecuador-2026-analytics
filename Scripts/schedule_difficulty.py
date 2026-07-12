"""
Remaining Schedule Difficulty: for each team's 13 remaining fixtures,
average the *effective* strength of the opponent they'll actually face --
using the opponent's away rating when the opponent travels, or their
home rating when the team travels to them (Pi-rating model, ELO scale).
Higher = harder run-in.

Usage: python3 schedule_difficulty.py [out.png]
"""
import sys

import matplotlib.pyplot as plt

import pi_ratings_lib as pil

LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
HARD_COLOR = "#e0765c"
EASY_COLOR = "#6fcf7a"
MID_COLOR = "#3a4658"


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


def remaining_fixtures(teams, matches):
    played = {(m["home"], m["away"]) for m in matches}
    return [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]


def make_plot(d, out_path):
    teams, matches, points = d["teams"], d["matches"], d["points"]
    history = d["history"]
    home_rating = {t: history[t][-1]["home_rating"] if history.get(t) else 0.0 for t in teams}
    away_rating = {t: history[t][-1]["away_rating"] if history.get(t) else 0.0 for t in teams}

    fixtures = remaining_fixtures(teams, matches)
    opp_strengths = {t: [] for t in teams}
    home_ct = {t: 0 for t in teams}
    away_ct = {t: 0 for t in teams}
    for h, a in fixtures:
        opp_strengths[h].append(pil.to_elo(away_rating[a]))
        home_ct[h] += 1
        opp_strengths[a].append(pil.to_elo(home_rating[h]))
        away_ct[a] += 1

    avg_difficulty = {t: sum(v) / len(v) for t, v in opp_strengths.items() if v}
    teams_sorted = sorted(avg_difficulty, key=lambda t: -avg_difficulty[t])
    n = len(teams_sorted)

    league_avg = sum(avg_difficulty.values()) / n

    fig, ax = plt.subplots(figsize=(13.5, 0.62 * n + 2.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y_pos = list(range(n))[::-1]
    for y, t in zip(y_pos, teams_sorted):
        val = avg_difficulty[t]
        color = HARD_COLOR if val >= league_avg + 4 else (EASY_COLOR if val <= league_avg - 4 else MID_COLOR)
        ax.barh(y, val, color=color, height=0.62, zorder=3, left=0)
        ax.text(val + 3, y, f"{val:.0f}  ({home_ct[t]}H/{away_ct[t]}A)", va="center", ha="left",
                fontsize=9.5, color=TEXT_MAIN, fontweight="bold")

    ax.axvline(league_avg, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.6, zorder=2)
    ax.text(league_avg, n - 0.1, "league avg", fontsize=8.5, color=TEXT_SUB, ha="center", va="bottom")

    ax.set_yticks(y_pos)
    labels = [f"#{i+1}  {pil.clean_name(t)}" for i, t in enumerate(teams_sorted)]
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", colors=TEXT_MAIN, length=0)
    ax.set_xlabel("Avg. effective opponent strength, remaining 13 games  (Pi-rating ELO scale)",
                 fontsize=10.5, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)
    lo = min(avg_difficulty.values())
    hi = max(avg_difficulty.values())
    pad = (hi - lo) * 0.18
    ax.set_xlim(lo - pad, hi + pad * 2.2)
    ax.set_ylim(-0.7, n - 0.2)

    fig.text(0.05, 0.975, "Ecuador 2026  ·  All Teams  ·  Remaining Schedule Difficulty",
             fontsize=19, fontweight="bold", color="white")
    fig.text(0.05, 0.951, "Hardest run-in at top  ·  Red = tougher than average  ·  Green = easier than "
             "average", fontsize=10.5, color=TEXT_SUB)
    fig.text(0.05, 0.022, "Data via Opta | Ecuador 2026 event data · Opponent strength = their Pi-rating "
             "ELO in the venue they'll actually play (away rating if they travel, home rating if they "
             "host)", fontsize=7.8, color="#6b7684")
    fig.text(0.05, 0.006, "13 remaining fixtures per team, from the standard LigaPro double round-robin "
             "schedule", fontsize=7.8, color="#6b7684")
    fig.text(0.98, 0.014, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.30, right=0.94, top=0.905, bottom=0.085)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/schedule_difficulty.png"
    d = pil.load_all()
    make_plot(d, out)
