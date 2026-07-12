"""
Full Season Projection: Monte Carlo simulation of the remaining fixtures
(13 games per team, 104 total -- Ecuador LigaPro plays a 16-team double
round-robin, 30 rounds; 136 of 240 fixtures are already played) using the
Pi-rating model already built for this league.

For each of the 104 remaining fixtures, expected goal difference comes
from the same g() compression function used throughout (pi_ratings_lib),
evaluated with each team's rating at that point in the simulated
timeline. Match outcomes are drawn from a normal approximation around
that expected value, with the residual std fitted from this season's 136
already-played matches (actual goal diff vs. model's pre-match expected
diff). Ratings are then updated with the sampled result exactly as in
the real model, so simulated form still evolves during the run. Repeated
4,000 times to get a distribution of final points/rank per team.

Playoff (top 8) and relegation (bottom 2) cutoffs follow the standard
Ecuador LigaPro aggregate-table format -- an assumption about the
competition structure, not something confirmed inside the event data
itself.

Usage: python3 season_projection.py [out.png] [n_sims]
"""
import sys
import random
import collections
import statistics

import matplotlib.pyplot as plt

import pi_ratings_lib as pil

LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
PLAYOFF_COLOR = "#6fcf7a"
MID_COLOR = "#6fa8dc"
RELEGATION_COLOR = "#e0765c"
CURRENT_MARK = "#ffc247"

PLAYOFF_SPOTS = 8
RELEGATION_SPOTS = 2
MARGIN_CHOICES = [1, 2, 3, 4]
MARGIN_WEIGHTS = [50, 30, 15, 5]

N_SIMS_DEFAULT = 4000


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


def fit_residual_std(matches, teams):
    home_rating = {t: 0.0 for t in teams}
    away_rating = {t: 0.0 for t in teams}
    resid = []
    for m in matches:
        h, a = m["home"], m["away"]
        exp = pil.g(home_rating[h] - away_rating[a])
        actual = m["home_goals"] - m["away_goals"]
        resid.append(actual - exp)
        err = actual - exp
        rh_new = home_rating[h] + pil.LAMBDA * err
        ra_new = away_rating[a] - pil.LAMBDA * err
        ra_home_new = away_rating[h] + pil.GAMMA * (rh_new - home_rating[h])
        rh_away_new = home_rating[a] + pil.GAMMA * (ra_new - away_rating[a])
        home_rating[h] = rh_new
        away_rating[h] = ra_home_new
        home_rating[a] = rh_away_new
        away_rating[a] = ra_new
    return statistics.pstdev(resid)


def norm_cdf(x):
    return 0.5 * (1 + _erf(x / 2 ** 0.5))


def _erf(x):
    # Abramowitz-Stegun approximation, good to ~1e-7
    sign = 1 if x >= 0 else -1
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (2.718281828459045 ** (-x * x))
    return sign * y


def remaining_fixtures(teams, matches):
    played = {(m["home"], m["away"]) for m in matches}
    remaining = [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]
    return remaining


def run_simulations(d, std, n_sims, seed=42):
    rng = random.Random(seed)
    teams = d["teams"]
    base_points = d["points"]
    base_home = {t: d["history"][t][-1]["home_rating"] if d["history"].get(t) else 0.0 for t in teams}
    base_away = {t: d["history"][t][-1]["away_rating"] if d["history"].get(t) else 0.0 for t in teams}
    fixtures = remaining_fixtures(teams, d["matches"])

    final_points = collections.defaultdict(list)
    title_ct = collections.Counter()
    playoff_ct = collections.Counter()
    relegation_ct = collections.Counter()

    for _ in range(n_sims):
        points = dict(base_points)
        home_rating = dict(base_home)
        away_rating = dict(base_away)
        sim_fixtures = fixtures[:]
        rng.shuffle(sim_fixtures)

        for h, a in sim_fixtures:
            exp = pil.g(home_rating[h] - away_rating[a])
            p_home = 1 - norm_cdf((0.5 - exp) / std)
            p_away = norm_cdf((-0.5 - exp) / std)
            p_draw = max(0.0, 1 - p_home - p_away)
            total = p_home + p_draw + p_away
            r = rng.random() * total
            if r < p_home:
                margin = rng.choices(MARGIN_CHOICES, weights=MARGIN_WEIGHTS)[0]
                actual = margin
                points[h] += 3
            elif r < p_home + p_draw:
                actual = 0
                points[h] += 1
                points[a] += 1
            else:
                margin = rng.choices(MARGIN_CHOICES, weights=MARGIN_WEIGHTS)[0]
                actual = -margin
                points[a] += 3

            err = actual - exp
            rh_new = home_rating[h] + pil.LAMBDA * err
            ra_new = away_rating[a] - pil.LAMBDA * err
            ra_home_new = away_rating[h] + pil.GAMMA * (rh_new - home_rating[h])
            rh_away_new = home_rating[a] + pil.GAMMA * (ra_new - away_rating[a])
            home_rating[h] = rh_new
            away_rating[h] = ra_home_new
            home_rating[a] = rh_away_new
            away_rating[a] = ra_new

        ranked = sorted(teams, key=lambda t: (-points[t], rng.random()))
        for t in teams:
            final_points[t].append(points[t])
        for rank, t in enumerate(ranked, start=1):
            if rank == 1:
                title_ct[t] += 1
            if rank <= PLAYOFF_SPOTS:
                playoff_ct[t] += 1
            if rank > len(teams) - RELEGATION_SPOTS:
                relegation_ct[t] += 1

    return final_points, title_ct, playoff_ct, relegation_ct


def percentile(values, p):
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[idx]


def make_plot(d, out_path, n_sims):
    std = fit_residual_std(d["matches"], d["teams"])
    final_points, title_ct, playoff_ct, relegation_ct = run_simulations(d, std, n_sims)

    teams = d["teams"]
    median_pts = {t: statistics.median(final_points[t]) for t in teams}
    teams_sorted = sorted(teams, key=lambda t: -median_pts[t])
    n = len(teams_sorted)

    fig, ax = plt.subplots(figsize=(14, 0.64 * n + 2.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    all_p10 = [percentile(final_points[t], 10) for t in teams_sorted]
    all_p90 = [percentile(final_points[t], 90) for t in teams_sorted]
    xlim_lo = min(all_p10) - 8
    xlim_hi = max(all_p90) * 1.34
    ax.set_xlim(xlim_lo, xlim_hi)
    label_frac = (max(all_p90) - xlim_lo) / (xlim_hi - xlim_lo) + 0.025

    y_pos = list(range(n))[::-1]
    for y, t in zip(y_pos, teams_sorted):
        vals = final_points[t]
        p10, p90 = percentile(vals, 10), percentile(vals, 90)
        med = median_pts[t]
        rank = n - y
        color = PLAYOFF_COLOR if rank <= PLAYOFF_SPOTS else (
            RELEGATION_COLOR if rank > n - RELEGATION_SPOTS else MID_COLOR)

        ax.plot([p10, p90], [y, y], color=color, linewidth=6, alpha=0.55, zorder=2,
                solid_capstyle="round")
        ax.scatter([med], [y], color=color, s=90, zorder=4, edgecolors="white", linewidths=1.2)
        ax.scatter([d["points"][t]], [y], color=CURRENT_MARK, s=55, marker="D", zorder=5,
                  edgecolors="#0d1117", linewidths=0.8)

        ax.text(label_frac, y, f"med {med:.0f}  ·  playoff {playoff_ct[t]/n_sims*100:.0f}%"
                + (f"  ·  reloc {relegation_ct[t]/n_sims*100:.0f}%" if relegation_ct[t] > 0 else ""),
                va="center", ha="left", fontsize=9, color=TEXT_SUB, transform=ax.get_yaxis_transform())

    ax.set_yticks(y_pos)
    labels = [f"#{i+1}  {pil.clean_name(t)}" for i, t in enumerate(teams_sorted)]
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", colors=TEXT_MAIN, length=0)
    ax.set_xlabel("Projected final points  (10th-90th percentile band, ◆ = current points, "
                 "● = median projection)", fontsize=10.5, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_ylim(-0.7, n - 0.2)

    fig.text(0.05, 0.975, "Ecuador 2026  ·  Full Season Projection",
             fontsize=19, fontweight="bold", color="white")
    fig.text(0.05, 0.951, f"Monte Carlo simulation (N={n_sims:,}) of the remaining 13 games per team, "
             "using the Pi-rating model", fontsize=10.5, color=TEXT_SUB)
    fig.text(0.05, 0.930, "Green = projected playoff zone (top 8)  ·  Red = projected relegation (bottom "
             "2)  ·  Assumes standard LigaPro double round-robin format", fontsize=8.5, color="#6b7684")
    fig.text(0.05, 0.024, f"Data via Opta | Ecuador 2026 event data · Outcome model: normal approx. "
             f"around Pi-rating expected goal diff (fitted std={std:.2f} from 136 played matches)",
             fontsize=7.6, color="#6b7684")
    fig.text(0.05, 0.007, "Playoff/relegation cutoffs are an assumption about league structure, not "
             "confirmed in the data", fontsize=7.6, color="#6b7684")
    fig.text(0.98, 0.015, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.24, right=0.97, top=0.885, bottom=0.09)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/season_projection.png"
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else N_SIMS_DEFAULT
    d = pil.load_all()
    make_plot(d, out, n_sims)
