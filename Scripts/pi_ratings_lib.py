"""
Shared Pi-rating engine (Constantinou & Fenton, 2013) for Ecuador 2026.

Each team carries two ratings -- a home rating and an away rating. After
every match, both are updated based on how much the actual goal
difference surprised the model (scaled by lambda), and each team's
*other* rating is partially nudged toward that same movement (scaled by
gamma), since home form carries some information about away form and
vice versa.

    g(x)      = C * log10(1 + |x|) * sign(x)      -- compresses goal
                differences/rating gaps so a 4-goal win doesn't move
                the rating 4x as much as a 1-goal win
    expected  = g(home_rating[H] - away_rating[A])
    error     = (home_goals - away_goals) - expected
    home_rating[H] += lambda * error
    away_rating[A] -= lambda * error
    away_rating[H] += gamma * (new_home_rating[H] - old_home_rating[H])
    home_rating[A] += gamma * (new_away_rating[A] - old_away_rating[A])

This is a from-scratch reimplementation matched to the published
structure (not a reproduction of any specific prior codebase), tuned
with the same lambda/gamma the reference charts used.
"""
import glob
import json
import math
import re
import collections

DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"

LAMBDA = 0.06
GAMMA = 0.5
C = 3.0
ELO_BASE = 1500
ELO_SCALE = 150

PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name):
    return PREFIX_RE.sub("", name)


def g(x):
    if x == 0:
        return 0.0
    return C * math.log10(1 + abs(x)) * (1 if x > 0 else -1)


def build_team_map(files):
    """team display name -> contestantId, for every team in the league."""
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
    return team_to_cid


def load_matches(files):
    """Chronological list of match dicts: date, home/away names+cids, goals."""
    matches = []
    for fn in files:
        base = fn.split("/")[-1]
        m = re.match(r"(\d{4}-\d{2}-\d{2})_(.+) - (.+)\.json$", base)
        if not m:
            continue
        date, home_name, away_name = m.group(1), m.group(2), m.group(3)
        with open(fn) as f:
            data = json.load(f)
        scores = data.get("matchDetails", {}).get("scores", {}).get("total")
        if not scores:
            continue
        cids = set(e["contestantId"] for e in data["event"] if "contestantId" in e)
        if len(cids) != 2:
            continue
        matches.append({
            "date": date, "home": home_name, "away": away_name,
            "home_goals": scores["home"], "away_goals": scores["away"],
        })
    matches.sort(key=lambda m: m["date"])
    return matches


def run_pi_ratings(matches, teams):
    """Returns per-team history: list of dicts with date, opponent,
    home_rating, away_rating, combined -- recorded after each of that
    team's matches, in chronological order."""
    home_rating = {t: 0.0 for t in teams}
    away_rating = {t: 0.0 for t in teams}
    history = collections.defaultdict(list)
    points = {t: 0 for t in teams}

    for m in matches:
        h, a = m["home"], m["away"]
        if h not in home_rating or a not in home_rating:
            continue
        hg, ag = m["home_goals"], m["away_goals"]

        rh_home_old = home_rating[h]
        ra_away_old = away_rating[a]
        expected = g(rh_home_old - ra_away_old)
        actual = hg - ag
        error = actual - expected

        rh_home_new = rh_home_old + LAMBDA * error
        ra_away_new = ra_away_old - LAMBDA * error
        ra_home_new = away_rating[h] + GAMMA * (rh_home_new - rh_home_old)
        rh_away_new = home_rating[a] + GAMMA * (ra_away_new - ra_away_old)

        home_rating[h] = rh_home_new
        away_rating[h] = ra_home_new
        home_rating[a] = rh_away_new
        away_rating[a] = ra_away_new

        if hg > ag:
            points[h] += 3
        elif hg < ag:
            points[a] += 3
        else:
            points[h] += 1
            points[a] += 1

        history[h].append({
            "date": m["date"], "opponent": a, "home_goals": hg, "away_goals": ag,
            "home_rating": home_rating[h], "away_rating": away_rating[h],
            "combined": (home_rating[h] + away_rating[h]) / 2,
            "points": points[h],
        })
        history[a].append({
            "date": m["date"], "opponent": h, "home_goals": hg, "away_goals": ag,
            "home_rating": home_rating[a], "away_rating": away_rating[a],
            "combined": (home_rating[a] + away_rating[a]) / 2,
            "points": points[a],
        })

    return history, points


def to_elo(combined_rating):
    return ELO_BASE + ELO_SCALE * combined_rating


def load_all(data_dir=DATA_DIR):
    files = sorted(glob.glob(f"{data_dir}/*.json"))
    team_to_cid = build_team_map(files)
    teams = list(team_to_cid.keys())
    matches = load_matches(files)
    history, points = run_pi_ratings(matches, teams)
    return {
        "files": files, "team_to_cid": team_to_cid, "teams": teams,
        "matches": matches, "history": history, "points": points,
    }
