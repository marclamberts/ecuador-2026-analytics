"""
Team- and lineup-specific nonstationary possession-clock MDPs for football,
fit from Opta/Stats Perform F24-style event feeds.

See README.md in this directory for the full methodology writeup. In brief:
each team's possessions are modeled as episodes of a Markov decision process
whose state is (pitch zone, possession-clock bucket), whose actions are the
coarse attempted decision (advance / sideways / back / cross / shot), and
whose transition probabilities are fit with a three-level Bayesian
hierarchy (league -> team -> lineup) using conjugate Dirichlet-Multinomial
updates, then recombined into a single team-average MDP with an
exposure-weighted mixture over lineups.

Event typeId reference used here (matches the convention already
established by the other Scripts/*.py models in this repo):
    1  Pass                 12 Clearance            34 Team Set Up
    2  Offside Pass         13 Miss                 44 Aerial
    3  Take On              14 Post                 49 Ball Recovery
    4  Foul                 15 Attempt Saved
    7  Tackle                16 Goal
    8  Interception          18 Player off (sub off)
                             19 Player on  (sub on)
"""
from __future__ import annotations

import bisect
import collections
import datetime as dt
import json
import pathlib
import re
from dataclasses import dataclass, field

PASS = 1
OFFSIDE_PASS = 2
TAKE_ON = 3
FOUL = 4
TACKLE = 7
INTERCEPTION = 8
CLEARANCE = 12
SHOT_TYPES = {13, 14, 15, 16}
GOAL_TYPE = 16
PLAYER_OFF = 18
PLAYER_ON = 19
TEAM_SET_UP = 34
AERIAL = 44
BALL_RECOVERY = 49

CONTINUATION_TYPES = {PASS, TAKE_ON, AERIAL, BALL_RECOVERY}
RESET_TYPES = {CLEARANCE, TACKLE, INTERCEPTION, FOUL, OFFSIDE_PASS}

END_X_QID, END_Y_QID = 140, 141
LINEUP_QID = 30  # list of 11 starting player ids, Team Set Up event

N_COLS, N_ROWS = 6, 4  # pitch-zone grid (length x width), x=100 = opponent goal
CLOCK_EDGES = [5, 10, 15, 25]  # possession-clock bucket edges, seconds
N_CLOCK = len(CLOCK_EDGES) + 1
N_ZONES = N_COLS * N_ROWS
N_STATES = N_ZONES * N_CLOCK

ACTIONS = ["advance", "sideways", "back", "cross", "shot"]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

GOAL = "GOAL"
TURNOVER = "TURNOVER"
NOGOAL_SHOT = "SHOT_NOGOAL"
TERMINALS = [GOAL, TURNOVER, NOGOAL_SHOT]

MAX_POSSESSION_SECONDS = 60.0

PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name: str) -> str:
    return PREFIX_RE.sub("", name)


def match_teams_from_filename(path: pathlib.Path) -> tuple[str, str]:
    m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", path.name)
    if not m:
        return "Unknown Home", "Unknown Away"
    return m.group(1), m.group(2)


def qmap_of(event: dict) -> dict:
    return {int(q["qualifierId"]): q.get("value") for q in event.get("qualifier", [])}


def zone_of(x: float, y: float) -> int:
    col = min(max(int(x / 100.0 * N_COLS), 0), N_COLS - 1)
    row = min(max(int(y / 100.0 * N_ROWS), 0), N_ROWS - 1)
    return col * N_ROWS + row


def clock_bucket(seconds: float) -> int:
    return bisect.bisect_right(CLOCK_EDGES, seconds)


def state_idx(zone: int, clock: int) -> int:
    return zone * N_CLOCK + clock


def classify_action(dx: float) -> str:
    """Coarse decision label from the change in x (toward opponent goal)."""
    if dx > 8:
        return "advance"
    if dx < -8:
        return "back"
    return "sideways"


def event_seconds(e: dict) -> float:
    """Fractional seconds within the event's period, from timeMin/timeSec."""
    return float(e.get("timeMin", 0)) * 60.0 + float(e.get("timeSec", 0))


def event_ts(e: dict) -> dt.datetime | None:
    ts = e.get("timeStamp")
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class Transition:
    team: str
    lineup: tuple
    state: int
    action: str
    outcome: object  # int next-state index, or one of TERMINALS
    weight: float = 1.0


@dataclass
class Episode:
    team: str
    lineup: tuple
    transitions: list = field(default_factory=list)
    start_zone: int = 0
    terminal: str = TURNOVER


def load_match(path: pathlib.Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_matches(data_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(data_dir.glob("*.json"))


def starting_lineups(events: list[dict]) -> dict[str, tuple]:
    """contestantId -> sorted tuple of the 11 starting player ids, from the
    Team Set Up (typeId 34) events."""
    lineups: dict[str, tuple] = {}
    for e in events:
        if e.get("typeId") != TEAM_SET_UP:
            continue
        cid = e.get("contestantId")
        if not cid or cid in lineups:
            continue
        qm = qmap_of(e)
        raw = qm.get(LINEUP_QID)
        if not raw:
            continue
        players = tuple(sorted(p.strip() for p in raw.split(",") if p.strip()))
        if len(players) >= 10:
            lineups[cid] = players
    return lineups


def build_lineup_timeline(events: list[dict]) -> dict[str, list[tuple]]:
    """contestantId -> list of (start_ts, end_ts_or_None, lineup_tuple),
    updated at every substitution (typeId 18/19)."""
    starters = starting_lineups(events)
    timeline: dict[str, list] = {cid: [] for cid in starters}
    current: dict[str, set] = {cid: set(p) for cid, p in starters.items()}
    seg_start: dict[str, dt.datetime | None] = {cid: None for cid in starters}

    ordered = sorted(
        (e for e in events if e.get("typeId") in (PLAYER_OFF, PLAYER_ON)),
        key=lambda e: (e.get("periodId", 0), event_seconds(e)),
    )
    for e in ordered:
        cid = e.get("contestantId")
        pid = e.get("playerId")
        if cid not in current or not pid:
            continue
        ts = event_ts(e)
        if seg_start[cid] is not None and ts is not None:
            timeline[cid].append((seg_start[cid], ts, tuple(sorted(current[cid]))))
        if e["typeId"] == PLAYER_OFF:
            current[cid].discard(pid)
        else:
            current[cid].add(pid)
        seg_start[cid] = ts

    for cid in starters:
        timeline[cid].insert(0, (None, seg_start[cid] or None, starters[cid]))
        timeline[cid].append((seg_start[cid], None, tuple(sorted(current[cid]))))
    return timeline


def lineup_at(timeline: list[tuple], ts: dt.datetime | None) -> tuple:
    if ts is None or not timeline:
        return timeline[0][2] if timeline else ()
    for start, end, lineup in timeline:
        if (start is None or ts >= start) and (end is None or ts < end):
            return lineup
    return timeline[-1][2]


def extract_episodes(match: dict, weight: float = 1.0) -> list[Episode]:
    """Segment one match's event stream into per-possession episodes and
    their (state, action, outcome) transitions. `weight` is the per-match
    recency weight (README.md Section 3, "borrowing through time") applied
    to every transition drawn from this match. See README.md Section 8 for
    the simplifying assumptions behind the segmentation itself."""
    events = match["event"]
    timelines = build_lineup_timeline(events)

    relevant = [
        e
        for e in events
        if e.get("typeId") in CONTINUATION_TYPES | RESET_TYPES | SHOT_TYPES
        and e.get("x") is not None
        and e.get("contestantId")
    ]
    relevant.sort(key=lambda e: (e.get("periodId", 0), event_seconds(e)))

    episodes: list[Episode] = []
    owner = None
    chain: Episode | None = None
    cur_state = None
    chain_start_sec = None
    last_period = None

    def finalize(terminal: str):
        nonlocal chain, owner, cur_state
        if chain is not None and chain.transitions:
            chain.terminal = terminal
            episodes.append(chain)
        chain, owner, cur_state = None, None, None

    for e in relevant:
        period = e.get("periodId", 0)
        if last_period is not None and period != last_period:
            finalize(TURNOVER)
        last_period = period

        typ = e["typeId"]
        cid = e["contestantId"]
        secs = event_seconds(e)
        x, y = float(e["x"]), float(e["y"])

        if typ in RESET_TYPES:
            if chain is not None and owner != cid:
                finalize(TURNOVER)
            continue

        if typ in SHOT_TYPES:
            if chain is None or owner != cid:
                finalize(TURNOVER)
                owner = cid
                cur_state = state_idx(zone_of(x, y), 0)
                chain_start_sec = secs
                chain = Episode(team=cid, lineup=lineup_at(timelines.get(cid, []), event_ts(e)))
                chain.start_zone = zone_of(x, y)
            outcome = GOAL if typ == GOAL_TYPE else NOGOAL_SHOT
            chain.transitions.append(Transition(cid, chain.lineup, cur_state, "shot", outcome, weight))
            finalize(outcome)
            continue

        # PASS / TAKE_ON / AERIAL / BALL_RECOVERY
        if chain is None or owner != cid:
            finalize(TURNOVER)
            owner = cid
            cur_state = state_idx(zone_of(x, y), 0)
            chain_start_sec = secs
            chain = Episode(team=cid, lineup=lineup_at(timelines.get(cid, []), event_ts(e)))
            chain.start_zone = zone_of(x, y)
            if typ != PASS:
                continue  # duel/recovery just establishes possession start
            # a completed opening pass still needs to record its transition below

        elapsed = max(0.0, secs - chain_start_sec) if chain_start_sec is not None else 0.0
        clk = clock_bucket(elapsed)
        succeeded = int(e.get("outcome", 0)) == 1

        if typ == PASS:
            qm = qmap_of(e)
            end_x = float(qm.get(END_X_QID, x))
            end_y = float(qm.get(END_Y_QID, y))
            dx = end_x - x
            is_wide_channel = y <= 21 or y >= 79
            action = "cross" if (is_wide_channel and end_x >= 78 and succeeded) else classify_action(dx)
            if succeeded:
                nxt_zone = zone_of(end_x, end_y)
                nxt_state = state_idx(nxt_zone, clk)
                chain.transitions.append(Transition(cid, chain.lineup, cur_state, action, nxt_state, weight))
                cur_state = nxt_state
            else:
                chain.transitions.append(Transition(cid, chain.lineup, cur_state, action, TURNOVER, weight))
                finalize(TURNOVER)
                continue
        else:  # TAKE_ON, AERIAL, BALL_RECOVERY continuing a live chain
            action = classify_action(0.0)  # treated as a lateral/holding action
            if succeeded:
                nxt_zone = zone_of(x, y)
                nxt_state = state_idx(nxt_zone, clk)
                chain.transitions.append(Transition(cid, chain.lineup, cur_state, action, nxt_state, weight))
                cur_state = nxt_state
            else:
                chain.transitions.append(Transition(cid, chain.lineup, cur_state, action, TURNOVER, weight))
                finalize(TURNOVER)
                continue

        if elapsed > MAX_POSSESSION_SECONDS:
            finalize(TURNOVER)

    finalize(TURNOVER)
    return episodes


def build_team_to_cid(files: list[pathlib.Path]) -> dict[str, str]:
    """Clean team display name -> contestantId, resolved by intersecting the
    contestantId set of every match a team appears in (same technique used
    by Scripts/pi_ratings_lib.py)."""
    team_cid_sets: dict[str, list[set]] = collections.defaultdict(list)
    for path in files:
        home, away = match_teams_from_filename(path)
        match = load_match(path)
        cids = {e["contestantId"] for e in match["event"] if "contestantId" in e}
        team_cid_sets[clean_name(home)].append(cids)
        team_cid_sets[clean_name(away)].append(cids)
    team_to_cid = {}
    for team, sets in team_cid_sets.items():
        inter = set.intersection(*sets)
        if len(inter) == 1:
            team_to_cid[team] = next(iter(inter))
    return team_to_cid


MATCH_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_")
RECENCY_RHO = 0.98  # per-week decay applied to older matchdays


def match_date(path: pathlib.Path) -> dt.date | None:
    m = MATCH_DATE_RE.match(path.name)
    if not m:
        return None
    return dt.date.fromisoformat(m.group(1))


def load_all_episodes(data_dir: pathlib.Path, limit: int | None = None) -> tuple[list[Episode], dict[str, str]]:
    """Returns (episodes, contestantId -> clean team name)."""
    files = find_matches(data_dir)
    if limit:
        files = files[:limit]
    id_to_name = {cid: name for name, cid in build_team_to_cid(files).items()}

    dates = [d for d in (match_date(p) for p in files) if d is not None]
    latest = max(dates) if dates else None

    episodes: list[Episode] = []
    for path in files:
        match = load_match(path)
        d = match_date(path)
        weight = RECENCY_RHO ** ((latest - d).days / 7.0) if (d and latest) else 1.0
        episodes.extend(extract_episodes(match, weight=weight))
    return episodes, id_to_name
