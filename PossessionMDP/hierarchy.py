"""
Three-level Bayesian hierarchical Dirichlet-Multinomial pooling
(league -> team -> lineup) for the transition and policy distributions of
the possession MDPs, plus the lineup-to-team-average transition-weighting
scheme. See README.md Sections 3-4 for the derivation this implements.
"""
from __future__ import annotations

import collections
from collections import Counter, defaultdict

import numpy as np

import mdp_model as mm

# Dirichlet concentration hyperparameters (pseudo-observation counts at each
# level of the cascade). Larger kappa => stronger shrinkage toward the
# parent level.
KAPPA_BASE = 2.0
KAPPA_TEAM = 8.0
KAPPA_LINEUP = 8.0
MIN_TEAM_AVG_CONCENTRATION = 3.0


def aggregate_counts(episodes: list[mm.Episode]):
    """Builds sparse, weighted transition and policy counts.

    Returns:
        transition_counts[(cid, lineup)][s][a] -> Counter(outcome -> weight)
        policy_counts[(cid, lineup)][s]        -> Counter(action  -> weight)
    """
    transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    policy_counts = defaultdict(lambda: defaultdict(Counter))
    for ep in episodes:
        key = (ep.team, ep.lineup)
        for tr in ep.transitions:
            transition_counts[key][tr.state][tr.action][tr.outcome] += tr.weight
            policy_counts[key][tr.state][tr.action] += tr.weight
    return transition_counts, policy_counts


def _normalize(alpha: dict) -> dict:
    total = sum(alpha.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in alpha.items()}


def _shrink(counts: dict, base: dict, kappa: float) -> dict:
    """Dirichlet posterior alpha = kappa * base_mean + observed_counts."""
    support = set(counts) | set(base)
    if not support:
        return {}
    return {o: kappa * base.get(o, 0.0) + counts.get(o, 0.0) for o in support}


def hierarchical_transition_alpha(transition_counts):
    """Cascades league -> team -> lineup Dirichlet posteriors for the
    transition (dynamics) distributions. Returns three dicts of alpha
    vectors keyed (s, a), (cid, s, a), and (cid, lineup, s, a)."""
    league_counts = defaultdict(lambda: defaultdict(Counter))
    team_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for (cid, lineup), by_s in transition_counts.items():
        for s, by_a in by_s.items():
            for a, ctr in by_a.items():
                for outcome, w in ctr.items():
                    league_counts[s][a][outcome] += w
                    team_counts[cid][s][a][outcome] += w

    league_alpha = {}
    for s, by_a in league_counts.items():
        for a, ctr in by_a.items():
            support = set(ctr)
            base = {o: 1.0 / len(support) for o in support} if support else {}
            league_alpha[(s, a)] = _shrink(ctr, base, KAPPA_BASE)

    team_alpha = {}
    for cid, by_s in team_counts.items():
        for s, by_a in by_s.items():
            for a, ctr in by_a.items():
                base = _normalize(league_alpha.get((s, a), {}))
                team_alpha[(cid, s, a)] = _shrink(ctr, base, KAPPA_TEAM)

    lineup_alpha = {}
    for (cid, lineup), by_s in transition_counts.items():
        for s, by_a in by_s.items():
            for a, ctr in by_a.items():
                base = _normalize(team_alpha.get((cid, s, a), {}))
                lineup_alpha[(cid, lineup, s, a)] = _shrink(ctr, base, KAPPA_LINEUP)

    return league_alpha, team_alpha, lineup_alpha


def hierarchical_policy(policy_counts):
    """League -> team cascade for the action-choice policy pi(a|s)."""
    league_counts = defaultdict(Counter)
    team_counts = defaultdict(lambda: defaultdict(Counter))
    for (cid, lineup), by_s in policy_counts.items():
        for s, ctr in by_s.items():
            for a, w in ctr.items():
                league_counts[s][a] += w
                team_counts[cid][s][a] += w

    league_policy = {}
    for s, ctr in league_counts.items():
        support = set(ctr)
        base = {a: 1.0 / len(support) for a in support} if support else {}
        league_policy[s] = _normalize(_shrink(ctr, base, KAPPA_BASE))

    team_policy = {}
    for cid, by_s in team_counts.items():
        for s, ctr in by_s.items():
            base = league_policy.get(s, {})
            team_policy[(cid, s)] = _normalize(_shrink(ctr, base, KAPPA_TEAM))

    return team_policy


def lineup_exposure_weights(transition_counts):
    """w_L: each lineup's share of its team's total (weighted) transition
    volume -- the exposure weight from README.md Section 4."""
    lineup_totals = defaultdict(float)
    team_totals = defaultdict(float)
    for (cid, lineup), by_s in transition_counts.items():
        total = sum(w for by_a in by_s.values() for ctr in by_a.values() for w in ctr.values())
        lineup_totals[(cid, lineup)] = total
        team_totals[cid] += total
    return {
        key: (total / team_totals[key[0]] if team_totals[key[0]] > 0 else 0.0)
        for key, total in lineup_totals.items()
    }


def team_average_alpha(transition_counts, lineup_alpha, team_alpha, weights):
    """The team-average MDP of README.md Section 4: combines lineup-specific
    posterior means P_L(s'|s,a), weighted by w_L * n_L(s,a), into a single
    team-average distribution per (cid, s, a). The result is expressed as a
    synthetic Dirichlet alpha (mean matched to the weighted mixture, total
    concentration matched to the combined lineup evidence) so the simulator
    can still draw posterior samples from it exactly as it would from any
    other level of the hierarchy."""
    by_cid = defaultdict(set)
    for cid, lineup in transition_counts:
        by_cid[cid].add(lineup)

    result = {}
    for cid, lineups in by_cid.items():
        cells = set()
        for lineup in lineups:
            for s, by_a in transition_counts[(cid, lineup)].items():
                cells.update((s, a) for a in by_a)

        for (s, a) in cells:
            numer = Counter()
            denom = 0.0
            for lineup in lineups:
                n_L = sum(transition_counts[(cid, lineup)].get(s, {}).get(a, {}).values())
                if n_L <= 0:
                    continue
                w_L = weights.get((cid, lineup), 0.0)
                p_L = _normalize(lineup_alpha.get((cid, lineup, s, a), {}))
                if not p_L:
                    continue
                weight = w_L * n_L
                denom += weight
                for outcome, p in p_L.items():
                    numer[outcome] += weight * p

            if denom > 0:
                mean = {o: v / denom for o, v in numer.items()}
                strength = max(denom, MIN_TEAM_AVG_CONCENTRATION)
                result[(cid, s, a)] = {o: p * strength for o, p in mean.items()}
            else:
                result[(cid, s, a)] = team_alpha.get((cid, s, a), {})
    return result


def dirichlet_mean(alpha: dict) -> dict:
    return _normalize(alpha)


def dirichlet_sample(alpha: dict, rng: np.random.Generator) -> dict:
    if not alpha:
        return {}
    outcomes = list(alpha.keys())
    concentrations = np.array([max(alpha[o], 1e-6) for o in outcomes])
    draw = rng.dirichlet(concentrations)
    return dict(zip(outcomes, draw))
