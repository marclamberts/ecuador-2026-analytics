"""
Goalkeeper Value Model: 13 submodels covering shot-stopping, claiming,
sweeping, distribution, risk, and availability, rolled into a single
composite Goalkeeper Value Index.

Data sources (all Ecuador 2026 season):
  - Aggregated/player_match_metrics.csv  -> per-keeper match rows
    (position_group == 'GK'), raw GK action counts, passing/xT metrics
  - Danger/*_danger_models.csv           -> per-shot psxg/xgot/xg model
    output (trained shot model), used for goals-prevented and
    difficulty-weighted stopping
  - Aggregated/team_match_metrics.csv    -> team cross volume, used to
    derive crosses faced (opponent's crosses_into_box in the same match)
  - Event/*.json                         -> typeId 51 ("error") events,
    not broken out as a column in the aggregator, parsed directly

Every Ecuador 2026 match in this dataset has exactly one goalkeeper per
team for the full 90 (no in-match keeper substitutions), so match-level
joins are exact -- no minute-window attribution is needed.

Usage: python3 build_goalkeeper_value_model.py
"""

from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
AGG_DIR = ROOT / "Aggregated"
DANGER_DIR = ROOT / "Danger"
EVENT_DIR = ROOT / "Event"
OUT_DIR = ROOT / "Goalkeeper Value Model"

MIN_MINUTES_FOR_RANKING = 450.0  # ~5 full matches
BIG_CHANCE_XG = 0.30
ERROR_TYPE_ID = 51

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


def load_gk_matches() -> pd.DataFrame:
    pm = pd.read_csv(AGG_DIR / "player_match_metrics.csv")
    gk = pm[pm["position_group"] == "GK"].copy()
    keep_cols = [
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
    gk = gk[[c for c in keep_cols if c in gk.columns]].reset_index(drop=True)
    return gk


def load_shots() -> pd.DataFrame:
    files = [f for f in glob.glob(str(DANGER_DIR / "*_danger_models.csv")) if "eredivisie" not in f]
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["match_file"] = pathlib.Path(f).name.replace("_danger_models.csv", ".json")
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def shots_faced_by_team(shots: pd.DataFrame, gk: pd.DataFrame) -> pd.DataFrame:
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


def crosses_faced_by_team(gk: pd.DataFrame) -> pd.DataFrame:
    tm = pd.read_csv(AGG_DIR / "team_match_metrics.csv")
    tm = tm[["match_file", "team_id", "crosses_into_box"]].rename(
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


def error_events_by_keeper(gk_ids: set) -> pd.DataFrame:
    rows = []
    for f in EVENT_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for e in data.get("event", []) or []:
            if e.get("typeId") == ERROR_TYPE_ID and e.get("playerId") in gk_ids:
                rows.append({"match_file": f.name, "player_id": e.get("playerId"), "errors": 1})
    if not rows:
        return pd.DataFrame(columns=["match_file", "player_id", "errors"])
    df = pd.DataFrame(rows)
    return df.groupby(["match_file", "player_id"], as_index=False)["errors"].sum()


def build_match_table() -> pd.DataFrame:
    gk = load_gk_matches()
    shots = load_shots()
    shot_agg = shots_faced_by_team(shots, gk)
    cross_agg = crosses_faced_by_team(gk)
    err_agg = error_events_by_keeper(set(gk["player_id"]))

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


def p90(total: pd.Series, minutes: pd.Series) -> pd.Series:
    mins = minutes.replace(0, np.nan)
    return (total / mins * 90.0).fillna(0.0)


def bayesian_rate(successes: pd.Series, attempts: pd.Series, prior_rate: float, prior_weight: float) -> pd.Series:
    return (successes + prior_rate * prior_weight) / (attempts + prior_weight)


def build_season_table(match_df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = ["season", "team", "player_id", "player"]
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

    # --- Submodel 1: Shot-Stopping (Goals Prevented Above Expected) ---
    agg["gpae_p90"] = p90(agg["gpae"], agg["minutes"])

    # --- Submodel 2: Save-Difficulty-Weighted Stopping ---
    # sum of xgot (post-shot on-target difficulty) across shots the keeper
    # actually saved, per 90 -- rewards saving hard shots over easy ones.
    agg["weighted_save_value_p90"] = p90(agg["xgot_saved_sum"], agg["minutes"])

    # --- Submodel 3: Big-Chance / High-Danger Denial ---
    # Bayesian-shrunk save rate on shots with xg >= 0.30 faced.
    league_big_chance_save_rate = 1 - (
        match_df["big_chance_goals_conceded"].sum() / max(match_df["big_chances_faced"].sum(), 1)
    )
    saves_big_chance = agg["big_chances_faced"] - agg["big_chance_goals_conceded"]
    agg["big_chance_save_rate"] = bayesian_rate(
        saves_big_chance, agg["big_chances_faced"], league_big_chance_save_rate, prior_weight=6.0
    )

    # --- Submodel 4: Shot-Stopping Reliability ---
    match_df_gpae90 = match_df.copy()
    match_df_gpae90["gpae_p90_match"] = p90(match_df_gpae90["gpae"], match_df_gpae90["minutes"])
    volatility = match_df_gpae90.groupby(grp_cols)["gpae_p90_match"].std().rename("gpae_volatility")
    agg = agg.merge(volatility, on=grp_cols, how="left")
    agg["gpae_volatility"] = agg["gpae_volatility"].fillna(agg["gpae_volatility"].median())

    # --- Submodel 5: Claiming & Command of the Box ---
    agg["claim_actions_p90"] = p90(agg["claim_actions"], agg["minutes"])
    agg["claim_rate"] = bayesian_rate(
        agg["claim_actions"], agg["crosses_faced"], agg["claim_actions"].sum() / max(agg["crosses_faced"].sum(), 1),
        prior_weight=10.0,
    )

    # --- Submodel 6: Sweeper Activity ---
    agg["sweeper_actions_p90"] = p90(agg["sweeper_actions_total"], agg["minutes"])

    # --- Submodel 7: Distribution Involvement ---
    agg["passes_p90"] = p90(agg["passes"], agg["minutes"])

    # --- Submodel 8: Distribution Accuracy (value-weighted) ---
    agg["pass_value_over_expected_p90"] = p90(agg["pass_value_over_expected"], agg["minutes"])

    # --- Submodel 9: Progressive Distribution Value (xT from own passing) ---
    agg["xt_p90"] = p90(agg["xt"], agg["minutes"])

    # --- Submodel 10: Error / Risk Cost ---
    agg["errors_p90"] = p90(agg["errors"], agg["minutes"])
    agg["losses_p90"] = p90(agg["losses"], agg["minutes"])
    agg["risk_cost_p90"] = agg["errors_p90"] * 2.0 + agg["losses_p90"]  # errors weighted heavier than generic losses

    # --- Submodel 11: Penalty Save Ability ---
    league_pen_save_rate = 1 - (
        match_df["penalty_goals_conceded"].sum() / max(match_df["penalties_faced_shots"].sum(), 1)
    )
    agg["penalty_save_rate"] = bayesian_rate(
        agg["penalty_saves"], agg["penalties_faced_shots"], league_pen_save_rate, prior_weight=8.0
    )

    # --- Submodel 12: Discipline Risk ---
    agg["fouls_p90"] = p90(agg["fouls"], agg["minutes"])
    agg["cards_p90"] = p90(agg["cards"], agg["minutes"])
    agg["discipline_cost_p90"] = agg["fouls_p90"] + agg["cards_p90"] * 3.0

    # --- Submodel 13: Availability / Durability ---
    team_max_minutes = match_df.groupby("team")["match_file"].nunique() * 90.0
    agg["team_possible_minutes"] = agg["team"].map(team_max_minutes)
    agg["availability_pct"] = (agg["minutes"] / agg["team_possible_minutes"]).clip(upper=1.0).fillna(0.0)

    return agg


def score_submodels(season: pd.DataFrame) -> pd.DataFrame:
    ranked = season[season["minutes"] >= MIN_MINUTES_FOR_RANKING].copy()
    if ranked.empty:
        ranked = season.copy()

    pctile_map = {
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

    scores = pd.DataFrame(index=ranked.index)
    for submodel, (col, invert) in pctile_map.items():
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


def write_submodel_definitions() -> None:
    rows = [
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
    df = pd.DataFrame(rows, columns=["submodel", "name", "formula", "composite_weight"])
    df.to_csv(OUT_DIR / "submodel_definitions.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    match_df = build_match_table()
    match_df.to_csv(OUT_DIR / "goalkeeper_match_value.csv", index=False)

    season = build_season_table(match_df)
    scored = score_submodels(season)
    scored.to_csv(OUT_DIR / "goalkeeper_season_value_model.csv", index=False)

    write_submodel_definitions()

    print(f"Wrote {len(match_df)} keeper-match rows -> {OUT_DIR / 'goalkeeper_match_value.csv'}")
    print(f"Wrote {len(scored)} ranked keeper-season rows -> {OUT_DIR / 'goalkeeper_season_value_model.csv'}")
    print(scored[["player", "team", "minutes", "matches", "goalkeeper_value_index", "goalkeeper_value_index_pctile", "goalkeeper_value_index_zscore"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
