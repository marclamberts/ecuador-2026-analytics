"""
Monte Carlo possession simulator for the fitted MDPs (README.md Section 5),
plus the on-policy vs altered-policy ("directness") comparison of
README.md Section 6.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

import mdp_model as mm
from hierarchy import dirichlet_sample

MAX_STEPS = 20


def start_zone_distribution(episodes: list[mm.Episode], cid: str) -> dict[int, float]:
    counts: Counter = Counter()
    for ep in episodes:
        if ep.team == cid:
            counts[ep.start_zone] += 1.0
    total = sum(counts.values())
    return {z: c / total for z, c in counts.items()} if total > 0 else {}


def _sample_categorical(rng: np.random.Generator, dist: dict):
    keys = list(dist.keys())
    probs = np.array(list(dist.values()), dtype=float)
    probs = probs / probs.sum()
    idx = rng.choice(len(keys), p=probs)
    return keys[idx]


def rollout(
    cid: str,
    start_zone: int,
    policy: dict,
    alpha_table: dict,
    draw_cache: dict,
    rng: np.random.Generator,
) -> tuple[str, int]:
    """One simulated possession. Returns (terminal_outcome, n_steps)."""
    s = mm.state_idx(start_zone, 0)
    for step in range(MAX_STEPS):
        pol = policy.get((cid, s))
        if not pol:
            return mm.TURNOVER, step
        action = _sample_categorical(rng, pol)

        cell = (cid, s, action)
        if cell not in draw_cache:
            alpha = alpha_table.get(cell)
            draw_cache[cell] = dirichlet_sample(alpha, rng) if alpha else {}
        draw = draw_cache[cell]
        if not draw:
            return mm.TURNOVER, step + 1

        outcome = _sample_categorical(rng, draw)
        if outcome in mm.TERMINALS:
            return outcome, step + 1
        s = outcome
    return mm.TURNOVER, MAX_STEPS


def simulate_policy(
    cid: str,
    alpha_table: dict,
    policy: dict,
    start_dist: dict[int, float],
    rng: np.random.Generator,
    n_worlds: int = 30,
    episodes_per_world: int = 400,
) -> dict:
    """Runs `n_worlds` posterior draws x `episodes_per_world` rollouts each,
    and returns per-world summary statistics (so the spread across worlds
    reflects parameter uncertainty, not just Monte Carlo rollout noise)."""
    if not start_dist:
        return {"goal_rate": [], "shot_rate": [], "turnover_rate": [], "mean_steps": []}

    goal_rates, shot_rates, turnover_rates, mean_steps_list = [], [], [], []
    for _ in range(n_worlds):
        draw_cache: dict = {}
        outcomes = []
        steps_list = []
        for _ in range(episodes_per_world):
            z0 = _sample_categorical(rng, start_dist)
            outcome, steps = rollout(cid, z0, policy, alpha_table, draw_cache, rng)
            outcomes.append(outcome)
            steps_list.append(steps)
        n = len(outcomes)
        goal_rates.append(sum(o == mm.GOAL for o in outcomes) / n)
        shot_rates.append(sum(o in (mm.GOAL, mm.NOGOAL_SHOT) for o in outcomes) / n)
        turnover_rates.append(sum(o == mm.TURNOVER for o in outcomes) / n)
        mean_steps_list.append(sum(steps_list) / n)

    return {
        "goal_rate": goal_rates,
        "shot_rate": shot_rates,
        "turnover_rate": turnover_rates,
        "mean_steps": mean_steps_list,
    }


def _episode_stats(outcomes: list, steps_list: list) -> dict:
    n = len(outcomes)
    return {
        "goal_rate": sum(o == mm.GOAL for o in outcomes) / n,
        "shot_rate": sum(o in (mm.GOAL, mm.NOGOAL_SHOT) for o in outcomes) / n,
        "turnover_rate": sum(o == mm.TURNOVER for o in outcomes) / n,
        "mean_steps": sum(steps_list) / n,
    }


def simulate_policy_pair(
    cid: str,
    alpha_table: dict,
    policy_a: dict,
    policy_b: dict,
    start_dist: dict[int, float],
    rng: np.random.Generator,
    n_worlds: int = 30,
    episodes_per_world: int = 400,
) -> dict:
    """Like simulate_policy, but runs both policies against the *same*
    per-world dynamics draw (shared draw_cache) so the only thing that
    differs between them within a world is the decision rule. Rare-event
    metrics like goal_rate are noisy enough that comparing two
    independently-drawn simulations (two separate simulate_policy calls)
    can flip sign between runs on Monte Carlo noise alone; pairing removes
    that shared source of variance from the comparison and leaves only the
    effect actually attributable to the policy change."""
    if not start_dist:
        empty = {"goal_rate": [], "shot_rate": [], "turnover_rate": [], "mean_steps": []}
        return {"a": empty, "b": empty, "delta_goal_rate": []}

    stats_a, stats_b, delta_goal = [], [], []
    for _ in range(n_worlds):
        draw_cache: dict = {}
        outcomes_a, steps_a, outcomes_b, steps_b = [], [], [], []
        for _ in range(episodes_per_world):
            z0 = _sample_categorical(rng, start_dist)
            o, s = rollout(cid, z0, policy_a, alpha_table, draw_cache, rng)
            outcomes_a.append(o)
            steps_a.append(s)
        for _ in range(episodes_per_world):
            z0 = _sample_categorical(rng, start_dist)
            o, s = rollout(cid, z0, policy_b, alpha_table, draw_cache, rng)
            outcomes_b.append(o)
            steps_b.append(s)
        sa, sb = _episode_stats(outcomes_a, steps_a), _episode_stats(outcomes_b, steps_b)
        stats_a.append(sa)
        stats_b.append(sb)
        delta_goal.append(sb["goal_rate"] - sa["goal_rate"])

    def stack(stats_list, key):
        return [s[key] for s in stats_list]

    return {
        "a": {k: stack(stats_a, k) for k in ("goal_rate", "shot_rate", "turnover_rate", "mean_steps")},
        "b": {k: stack(stats_b, k) for k in ("goal_rate", "shot_rate", "turnover_rate", "mean_steps")},
        "delta_goal_rate": delta_goal,
    }


def summarize(samples: list[float]) -> tuple[float, float, float]:
    """(mean, 2.5th pct, 97.5th pct) of the raw per-world samples -- the
    predictive spread across simulated seasons, e.g. "if we ran one season,
    what range of goal rates might we see"."""
    if not samples:
        return (float("nan"),) * 3
    arr = np.array(samples)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def mean_ci(samples: list[float]) -> tuple[float, float, float]:
    """(mean, lo, hi) 95% CI of the *mean* via the normal approximation
    (mean +/- 1.96 * SEM) -- how confidently we know the average effect,
    not the predictive spread of any one simulated world. Use this rather
    than summarize() when comparing a per-world paired difference, where
    the individual-world spread is much wider than the uncertainty in the
    average effect it's estimating."""
    if not samples:
        return (float("nan"),) * 3
    arr = np.array(samples)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    sem = float(arr.std(ddof=1)) / np.sqrt(len(arr))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def apply_directness_policy(team_policy: dict, cid: str, shift: float = 0.15) -> dict:
    """README.md Section 6's altered ("directness") policy: in the middle
    and attacking thirds, shifts probability mass from sideways/back toward
    advance/cross, leaving the fitted dynamics untouched."""
    altered = dict(team_policy)
    for (c, s), dist in team_policy.items():
        if c != cid:
            continue
        zone = s // mm.N_CLOCK
        col = zone // mm.N_ROWS
        if col < mm.N_COLS // 3:  # defensive third: leave build-up policy alone
            continue
        new_dist = dict(dist)
        take = 0.0
        for a in ("sideways", "back"):
            if a in new_dist:
                delta = new_dist[a] * shift
                new_dist[a] -= delta
                take += delta
        if take <= 0:
            continue
        targets = [a for a in ("advance", "cross") if a in new_dist] or list(new_dist)
        for a in targets:
            new_dist[a] += take / len(targets)
        altered[(c, s)] = new_dist
    return altered
