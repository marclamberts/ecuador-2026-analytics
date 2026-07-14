"""
The 13-submodel Goalkeeper Value Model: shot-stopping, claiming,
sweeping, distribution, risk, and availability, rolled into a single
composite Goalkeeper Value Index.

See ../README.md for the season-folder layout this expects and the
methodology writeup in ``Lamberts Goalkeeper Model/README.md`` for the
full formulas and caveats.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loader import SeasonData

MIN_MINUTES_FOR_RANKING_DEFAULT = 450.0  # ~5 full matches
BIG_CHANCE_XG = 0.30

MATCH_KEEP_COLS = [
    "season", "match_file", "date", "match", "team_id", "team",
    "player_id", "player", "minutes",
    "saves", "claims", "keeper_pickups", "keeper_sweeper_actions",
    "punches", "smothers", "penalties_faced", "gk_actions",
    "passes", "completed_passes", "accurate_long_passes",
    "accurate_short_medium_passes", "avg_pass_length_sum",
    "xt", "pass_value_over_expected", "pass_completion_oe",
    "progressive_passes", "fouls", "cards", "losses",
    "ga_on", "gf_on",
]

# Composite index weights. Shot-stopping is the core of the job and gets
# the largest share; sweeping/claiming and distribution are meaningful
# secondary skills; risk and availability are modifiers.
SUBMODEL_WEIGHTS = {
    "shot_stopping_gpae": 0.16,
    "save_difficulty_weighted": 0.12,
    "big_chance_denial": 0.10,
    "shot_stopping_reliability": 0.08,
    "claiming_command": 0.09,
    "sweeper_activity": 0.07,
    "distribution_involvement": 0.05,
    "distribution_accuracy": 0.08,
    "progressive_distribution": 0.07,
    "error_risk": 0.08,
    "penalty_save_ability": 0.04,
    "discipline_risk": 0.03,
    "availability": 0.03,
}
assert abs(sum(SUBMODEL_WEIGHTS.values()) - 1.0) < 1e-9

SUBMODEL_DEFINITIONS = [
    ("shot_stopping_gpae", "Shot-Stopping (Goals Prevented)", "Σ PSxG (on-target shots faced) − goals conceded, per 90.", SUBMODEL_WEIGHTS["shot_stopping_gpae"]),
    ("save_difficulty_weighted", "Save Difficulty-Weighted Stopping", "Σ xGOT of shots actually saved (harder saves count for more), per 90.", SUBMODEL_WEIGHTS["save_difficulty_weighted"]),
    ("big_chance_denial", "Big-Chance / High-Danger Denial", "Bayesian-shrunk save rate on faced shots with xG ≥ 0.30.", SUBMODEL_WEIGHTS["big_chance_denial"]),
    ("shot_stopping_reliability", "Shot-Stopping Reliability", "Inverse of match-to-match std dev of GPAE per 90 (consistency, not just magnitude).", SUBMODEL_WEIGHTS["shot_stopping_reliability"]),
    ("claiming_command", "Claiming & Command of the Box", "Bayesian-shrunk (claims+punches+smothers) rate per cross faced into the box.", SUBMODEL_WEIGHTS["claiming_command"]),
    ("sweeper_activity", "Sweeper Activity", "(keeper sweeper actions + keeper pickups) per 90.", SUBMODEL_WEIGHTS["sweeper_activity"]),
    ("distribution_involvement", "Distribution Involvement", "Total passes attempted per 90 (build-up outlet volume).", SUBMODEL_WEIGHTS["distribution_involvement"]),
    ("distribution_accuracy", "Distribution Accuracy", "Pass value over expected (xPass-model-based) per 90.", SUBMODEL_WEIGHTS["distribution_accuracy"]),
    ("progressive_distribution", "Progressive Distribution Value", "xT generated from the keeper's own passing, per 90.", SUBMODEL_WEIGHTS["progressive_distribution"]),
    ("error_risk", "Error / Risk Cost", "2×(typeId-51 error events per 90) + (turnovers on own passes per 90), inverted.", SUBMODEL_WEIGHTS["error_risk"]),
    ("penalty_save_ability", "Penalty Save Ability", "Bayesian-shrunk penalty save rate (prior weight 8 penalties).", SUBMODEL_WEIGHTS["penalty_save_ability"]),
    ("discipline_risk", "Discipline Risk", "Fouls per 90 + 3×cards per 90, inverted (availability risk).", SUBMODEL_WEIGHTS["discipline_risk"]),
    ("availability", "Availability / Durability", "Minutes played ÷ team's total possible minutes this season.", SUBMODEL_WEIGHTS["availability"]),
]


def pctile(series: pd.Series, invert: bool = False) -> pd.Series:
    """0-100 percentile rank, higher is always better after inversion."""
    s = series.astype(float)
    if invert:
        s = -s
    ranked = s.rank(pct=True, method="average") * 100.0
    return ranked.fillna(ranked.mean() if ranked.notna().any() else 50.0)


def zscore(series: pd.Series, invert: bool = False) -> pd.Series:
    """Population z-score ((x - mean) / std) within the ranked pool, higher
    is always better after inversion. Std of 0 (a constant column) maps to
    an all-zero z-score rather than dividing by zero."""
    s = series.astype(float)
    if invert:
        s = -s
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    z = (s - s.mean()) / std
    return z.fillna(0.0)


def _load_gk_matches(season: SeasonData) -> pd.DataFrame:
    gk = season.goalkeepers()
    gk = gk[[c for c in MATCH_KEEP_COLS if c in gk.columns]].reset_index(drop=True)
    return gk


def _shots_faced_by_team(shots: pd.DataFrame, gk: pd.DataFrame) -> pd.DataFrame:
    """For each (match_file, defending team_id), aggregate shots taken by the opponent."""
    teams_per_match = gk.groupby("match_file")["team_id"].apply(lambda s: sorted(set(s))).to_dict()
    rows = []
    for match_file, team_ids in teams_per_match.items():
        match_shots = shots[shots["match_file"] == match_file]
        if match_shots.empty or len(team_ids) != 2:
            continue
        for defending_team in team_ids:
            faced = match_shots[match_shots["contestant_id"] != defending_team]
            if faced.empty:
                rows.append({"match_file": match_file, "team_id": defending_team})
                continue
            on_target = faced[faced["is_on_target"] == 1]
            saved = on_target[on_target["is_goal"] == 0]
            big_chance = faced[faced["xg"] >= BIG_CHANCE_XG]
            big_chance_goals = big_chance["is_goal"].sum()
            penalties = faced[faced["is_penalty"] == 1]
            penalty_saves = penalties[penalties["is_goal"] == 0]
            rows.append({
                "match_file": match_file,
                "team_id": defending_team,
                "shots_faced": len(faced),
                "on_target_faced": len(on_target),
                "goals_conceded_run_of_play": int(faced["is_goal"].sum()),
                "psxg_faced": on_target["psxg"].sum(),
                "xgot_saved_sum": saved["xgot"].sum(),
                "big_chances_faced": len(big_chance),
                "big_chance_goals_conceded": int(big_chance_goals),
                "penalties_faced_shots": len(penalties),
                "penalty_saves": len(penalty_saves),
                "penalty_goals_conceded": int(penalties["is_goal"].sum()),
            })
    return pd.DataFrame(rows)


def _crosses_faced_by_team(season: SeasonData, gk: pd.DataFrame) -> pd.DataFrame:
    tm = season.team_match[["match_file", "team_id", "crosses_into_box"]].rename(
        columns={"crosses_into_box": "own_crosses_into_box"}
    )
    teams_per_match = gk.groupby("match_file")["team_id"].apply(lambda s: sorted(set(s))).to_dict()
    rows = []
    for match_file, team_ids in teams_per_match.items():
        if len(team_ids) != 2:
            continue
        sub = tm[tm["match_file"] == match_file].set_index("team_id")["own_crosses_into_box"].to_dict()
        for defending_team in team_ids:
            opponent = [t for t in team_ids if t != defending_team][0]
            rows.append({
                "match_file": match_file,
                "team_id": defending_team,
                "crosses_faced": sub.get(opponent, 0.0),
            })
    return pd.DataFrame(rows)


def _build_match_table(season: SeasonData) -> pd.DataFrame:
    gk = _load_gk_matches(season)
    shot_agg = _shots_faced_by_team(season.shots, gk)
    cross_agg = _crosses_faced_by_team(season, gk)
    err_agg = season.error_events_by_keeper(set(gk["player_id"]))

    df = gk.merge(shot_agg, on=["match_file", "team_id"], how="left")
    df = df.merge(cross_agg, on=["match_file", "team_id"], how="left")
    df = df.merge(err_agg, on=["match_file", "player_id"], how="left")

    numeric_fill = [
        "shots_faced", "on_target_faced", "goals_conceded_run_of_play", "psxg_faced",
        "xgot_saved_sum", "big_chances_faced", "big_chance_goals_conceded",
        "penalties_faced_shots", "penalty_saves", "penalty_goals_conceded",
        "crosses_faced", "errors",
    ]
    for c in numeric_fill:
        df[c] = df[c].fillna(0.0)

    df["goals_conceded"] = df["ga_on"].fillna(df["goals_conceded_run_of_play"])
    df["gpae"] = df["psxg_faced"] - df["goals_conceded_run_of_play"]
    df["claim_actions"] = df["claims"] + df["punches"] + df["smothers"]
    df["sweeper_actions_total"] = df["keeper_sweeper_actions"] + df["keeper_pickups"]
    return df


def _p90(total: pd.Series, minutes: pd.Series) -> pd.Series:
    mins = minutes.replace(0, np.nan)
    return (total / mins * 90.0).fillna(0.0)


def _bayesian_rate(successes: pd.Series, attempts: pd.Series, prior_rate: float, prior_weight: float) -> pd.Series:
    return (successes + prior_rate * prior_weight) / (attempts + prior_weight)


def _build_season_table(match_df: pd.DataFrame) -> pd.DataFrame:
    # Group by team_id and player_id, never by the team/player NAME text
    # columns: the aggregator's name resolution is occasionally wrong for a
    # given match row (team mislabeled as the opponent; player name falls
    # back to the raw player_id string when it wasn't captured for that
    # match's lineup/substitution event), while the ids are stable.
    # Grouping by name would silently fragment a keeper's season into bogus
    # single-match sub-groups under the wrong label. team_id/player_id are
    # reliable because every other join in this model (shots faced, crosses
    # faced) already keys off them. Every player_id in this dataset maps to
    # exactly one team_id (no mid-season transfers to worry about losing).
    grp_cols = ["season", "team_id", "player_id"]
    agg = match_df.groupby(grp_cols, as_index=False).agg(
        matches=("match_file", "nunique"),
        minutes=("minutes", "sum"),
        shots_faced=("shots_faced", "sum"),
        on_target_faced=("on_target_faced", "sum"),
        goals_conceded=("goals_conceded", "sum"),
        psxg_faced=("psxg_faced", "sum"),
        gpae=("gpae", "sum"),
        xgot_saved_sum=("xgot_saved_sum", "sum"),
        big_chances_faced=("big_chances_faced", "sum"),
        big_chance_goals_conceded=("big_chance_goals_conceded", "sum"),
        penalties_faced_shots=("penalties_faced_shots", "sum"),
        penalty_saves=("penalty_saves", "sum"),
        claims=("claims", "sum"),
        punches=("punches", "sum"),
        smothers=("smothers", "sum"),
        claim_actions=("claim_actions", "sum"),
        crosses_faced=("crosses_faced", "sum"),
        keeper_sweeper_actions=("keeper_sweeper_actions", "sum"),
        keeper_pickups=("keeper_pickups", "sum"),
        sweeper_actions_total=("sweeper_actions_total", "sum"),
        passes=("passes", "sum"),
        completed_passes=("completed_passes", "sum"),
        accurate_long_passes=("accurate_long_passes", "sum"),
        pass_value_over_expected=("pass_value_over_expected", "sum"),
        xt=("xt", "sum"),
        errors=("errors", "sum"),
        losses=("losses", "sum"),
        fouls=("fouls", "sum"),
        cards=("cards", "sum"),
    )

    # Canonical display names: the majority-vote team/player name across
    # that id's match rows (recovers a clean label even though individual
    # rows can be mislabeled or fall back to the raw id string).
    canonical_team = match_df.groupby("team_id")["team"].agg(lambda s: s.mode().iloc[0]).rename("team")
    canonical_player = match_df.groupby("player_id")["player"].agg(lambda s: s.mode().iloc[0]).rename("player")
    agg = agg.merge(canonical_team, on="team_id", how="left")
    agg = agg.merge(canonical_player, on="player_id", how="left")

    # --- Submodel 1: Shot-Stopping (Goals Prevented Above Expected) ---
    agg["gpae_p90"] = _p90(agg["gpae"], agg["minutes"])

    # --- Submodel 2: Save-Difficulty-Weighted Stopping ---
    agg["weighted_save_value_p90"] = _p90(agg["xgot_saved_sum"], agg["minutes"])

    # --- Submodel 3: Big-Chance / High-Danger Denial ---
    league_big_chance_save_rate = 1 - (
        match_df["big_chance_goals_conceded"].sum() / max(match_df["big_chances_faced"].sum(), 1)
    )
    saves_big_chance = agg["big_chances_faced"] - agg["big_chance_goals_conceded"]
    agg["big_chance_save_rate"] = _bayesian_rate(
        saves_big_chance, agg["big_chances_faced"], league_big_chance_save_rate, prior_weight=6.0
    )

    # --- Submodel 4: Shot-Stopping Reliability ---
    match_df_gpae90 = match_df.copy()
    match_df_gpae90["gpae_p90_match"] = _p90(match_df_gpae90["gpae"], match_df_gpae90["minutes"])
    volatility = match_df_gpae90.groupby(grp_cols)["gpae_p90_match"].std().rename("gpae_volatility")
    agg = agg.merge(volatility, on=grp_cols, how="left")
    agg["gpae_volatility"] = agg["gpae_volatility"].fillna(agg["gpae_volatility"].median())

    # --- Submodel 5: Claiming & Command of the Box ---
    agg["claim_actions_p90"] = _p90(agg["claim_actions"], agg["minutes"])
    agg["claim_rate"] = _bayesian_rate(
        agg["claim_actions"], agg["crosses_faced"], agg["claim_actions"].sum() / max(agg["crosses_faced"].sum(), 1),
        prior_weight=10.0,
    )

    # --- Submodel 6: Sweeper Activity ---
    agg["sweeper_actions_p90"] = _p90(agg["sweeper_actions_total"], agg["minutes"])

    # --- Submodel 7: Distribution Involvement ---
    agg["passes_p90"] = _p90(agg["passes"], agg["minutes"])

    # --- Submodel 8: Distribution Accuracy (value-weighted) ---
    agg["pass_value_over_expected_p90"] = _p90(agg["pass_value_over_expected"], agg["minutes"])

    # --- Submodel 9: Progressive Distribution Value (xT from own passing) ---
    agg["xt_p90"] = _p90(agg["xt"], agg["minutes"])

    # --- Submodel 10: Error / Risk Cost ---
    agg["errors_p90"] = _p90(agg["errors"], agg["minutes"])
    agg["losses_p90"] = _p90(agg["losses"], agg["minutes"])
    agg["risk_cost_p90"] = agg["errors_p90"] * 2.0 + agg["losses_p90"]

    # --- Submodel 11: Penalty Save Ability ---
    league_pen_save_rate = 1 - (
        match_df["penalty_goals_conceded"].sum() / max(match_df["penalties_faced_shots"].sum(), 1)
    )
    agg["penalty_save_rate"] = _bayesian_rate(
        agg["penalty_saves"], agg["penalties_faced_shots"], league_pen_save_rate, prior_weight=8.0
    )

    # --- Submodel 12: Discipline Risk ---
    agg["fouls_p90"] = _p90(agg["fouls"], agg["minutes"])
    agg["cards_p90"] = _p90(agg["cards"], agg["minutes"])
    agg["discipline_cost_p90"] = agg["fouls_p90"] + agg["cards_p90"] * 3.0

    # --- Submodel 13: Availability / Durability ---
    team_max_minutes = match_df.groupby("team_id")["match_file"].nunique() * 90.0
    agg["team_possible_minutes"] = agg["team_id"].map(team_max_minutes)
    agg["availability_pct"] = (agg["minutes"] / agg["team_possible_minutes"]).clip(upper=1.0).fillna(0.0)

    return agg


_PCTILE_MAP = {
    "shot_stopping_gpae": ("gpae_p90", False),
    "save_difficulty_weighted": ("weighted_save_value_p90", False),
    "big_chance_denial": ("big_chance_save_rate", False),
    "shot_stopping_reliability": ("gpae_volatility", True),
    "claiming_command": ("claim_rate", False),
    "sweeper_activity": ("sweeper_actions_p90", False),
    "distribution_involvement": ("passes_p90", False),
    "distribution_accuracy": ("pass_value_over_expected_p90", False),
    "progressive_distribution": ("xt_p90", False),
    "error_risk": ("risk_cost_p90", True),
    "penalty_save_ability": ("penalty_save_rate", False),
    "discipline_risk": ("discipline_cost_p90", True),
    "availability": ("availability_pct", False),
}


def _score_submodels(season_df: pd.DataFrame, min_minutes: float) -> pd.DataFrame:
    ranked = season_df[season_df["minutes"] >= min_minutes].copy()
    if ranked.empty:
        ranked = season_df.copy()

    scores = pd.DataFrame(index=ranked.index)
    for submodel, (col, invert) in _PCTILE_MAP.items():
        scores[f"{submodel}_score"] = pctile(ranked[col], invert=invert)
        scores[f"{submodel}_zscore"] = zscore(ranked[col], invert=invert)

    composite = pd.Series(0.0, index=ranked.index)
    composite_z = pd.Series(0.0, index=ranked.index)
    for submodel, weight in SUBMODEL_WEIGHTS.items():
        composite += scores[f"{submodel}_score"] * weight
        composite_z += scores[f"{submodel}_zscore"] * weight
    scores["goalkeeper_value_index"] = composite
    scores["goalkeeper_value_index_pctile"] = pctile(composite)
    scores["goalkeeper_value_index_zscore"] = composite_z

    out = pd.concat([ranked, scores], axis=1)
    return out.sort_values("goalkeeper_value_index", ascending=False)


@dataclass
class GoalkeeperValueModelResult:
    """Output of build_goalkeeper_value_model(): the raw match table and
    the scored, ranked season table, with a convenience .save()."""

    match_df: pd.DataFrame
    season_df: pd.DataFrame
    min_minutes: float

    def save(self, out_dir) -> None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.match_df.to_csv(out_dir / "goalkeeper_match_value.csv", index=False)
        self.season_df.to_csv(out_dir / "goalkeeper_season_value_model.csv", index=False)
        pd.DataFrame(
            SUBMODEL_DEFINITIONS, columns=["submodel", "name", "formula", "composite_weight"]
        ).to_csv(out_dir / "submodel_definitions.csv", index=False)


def build_goalkeeper_value_model(
    season_dir,
    min_minutes: float = MIN_MINUTES_FOR_RANKING_DEFAULT,
) -> GoalkeeperValueModelResult:
    """Automatically load a season data folder and build the composite
    Goalkeeper Value Index.

    ``season_dir`` can be a path to a season folder (auto-loaded via
    ``SeasonData.load``) or an already-loaded ``SeasonData`` instance.
    """
    season = season_dir if isinstance(season_dir, SeasonData) else SeasonData.load(season_dir)

    match_df = _build_match_table(season)
    season_raw = _build_season_table(match_df)
    scored = _score_submodels(season_raw, min_minutes)

    return GoalkeeperValueModelResult(match_df=match_df, season_df=scored, min_minutes=min_minutes)
