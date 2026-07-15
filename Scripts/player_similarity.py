"""
Player similarity model for the Ecuador 2026 dataset (Metrics -> Similarity).

Builds one profile per player from Aggregated/player_season_core.csv by
minutes-weighting each player's rows (a player can appear multiple times,
split by opponent/window), then ranks every other player in the same role
group by cosine similarity over the engineered style/output indices
(attacking_index, progression_index, defensive_index, ...) plus a curated
set of per-90 and percentage metrics.

Players below MIN_MINUTES are dropped from the candidate pool -- too small
a sample to trust the profile.

Usage:
    python3 player_similarity.py                 # writes Aggregated/player_similarity.csv
    python3 player_similarity.py "E. Mero"        # also prints that player's top matches
    python3 player_similarity.py "E. Mero" 15      # top 15 instead of the default 10
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG_DIR = os.path.join(REPO_ROOT, "Aggregated")
CORE_PATH = os.path.join(AGG_DIR, "player_season_core.csv")
OUT_PATH = os.path.join(AGG_DIR, "player_similarity.csv")

MIN_MINUTES = 450
TOP_N_DEFAULT = 10

INDEX_COLS = [
    "attacking_index", "progression_index", "defensive_index",
    "creative_engine_index", "ball_winner_index", "finisher_index",
    "chance_quality_index", "needle_mover_index", "chaos_creator_index",
    "press_resistance_index", "rest_defence_index",
    "territory_domination_index", "connector_index", "directness_index",
    "vertical_threat_index", "defensive_range_index",
    "goalkeeper_activity_index", "crossing_value_index",
    "box_delivery_index", "set_piece_creator_index",
    "dead_ball_threat_index", "corner_threat_index",
    "second_phase_corner_index", "disruption_index",
    "passing_added_index", "defensive_duel_index",
    "high_press_defending_index", "box_defender_index",
    "defensive_security_index", "waltzing_all_round_index",
    "box_presence_index", "transition_index",
]
RATE_COLS = [
    "xg_p90", "threat_p90", "xt_p90", "progressive_passes_p90",
    "def_actions_p90", "xdisruption_p90", "rapm_regularized_p90",
    "pass_pct", "xpass_completion_pct", "security_pct",
]
FEATURE_COLS = INDEX_COLS + RATE_COLS


def load_player_profiles():
    df = pd.read_csv(CORE_PATH)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)

    weighted = df.copy()
    for col in feature_cols:
        weighted[col] = weighted[col].fillna(0) * weighted["minutes"]

    grouped = weighted.groupby("player_id")
    total_minutes = grouped["minutes"].sum().rename("total_minutes")
    total_matches = grouped["matches"].sum().rename("total_matches")
    summed_features = grouped[feature_cols].sum()
    profile = summed_features.div(total_minutes.replace(0, np.nan), axis=0)
    profile = profile.join(total_minutes).join(total_matches)

    primary_rows = (
        df.sort_values("minutes", ascending=False)
        .groupby("player_id")
        .first()[["player", "team", "role", "archetype"]]
    )
    profile = profile.join(primary_rows)
    profile = profile.dropna(subset=feature_cols, how="all")
    profile = profile[profile["total_minutes"] >= MIN_MINUTES].copy()
    profile[feature_cols] = profile[feature_cols].fillna(0)
    return profile, feature_cols


def zscore_within_role(profile, feature_cols):
    z = profile.copy()
    for role, idx in profile.groupby("role").groups.items():
        block = profile.loc[idx, feature_cols]
        std = block.std(ddof=0).replace(0, np.nan)
        z.loc[idx, feature_cols] = ((block - block.mean()) / std).fillna(0)
    return z


def cosine_sim_matrix(mat):
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = mat / norm
    return unit @ unit.T


def build_similarity_table(profile, z, feature_cols, top_n):
    rows = []
    for role, idx in profile.groupby("role").groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        mat = z.loc[idx, feature_cols].to_numpy()
        sim = cosine_sim_matrix(mat)
        np.fill_diagonal(sim, -np.inf)
        players = profile.loc[idx, ["player", "team", "role"]].reset_index()
        for i, pid in enumerate(idx):
            order = np.argsort(sim[i])[::-1][:top_n]
            for rank, j in enumerate(order, start=1):
                if not np.isfinite(sim[i, j]):
                    continue
                rows.append({
                    "player_id": pid,
                    "player": players.loc[i, "player"],
                    "team": players.loc[i, "team"],
                    "role": role,
                    "rank": rank,
                    "similar_player_id": idx[j],
                    "similar_player": players.loc[j, "player"],
                    "similar_team": players.loc[j, "team"],
                    "similarity": round(float(sim[i, j]), 4),
                })
    return pd.DataFrame(rows)


def main():
    profile, feature_cols = load_player_profiles()
    z = zscore_within_role(profile, feature_cols)

    query = sys.argv[1] if len(sys.argv) > 1 else None
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else TOP_N_DEFAULT

    table = build_similarity_table(profile, z, feature_cols, top_n)
    table.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(table)} rows ({profile.shape[0]} players, "
          f"MIN_MINUTES={MIN_MINUTES}) to {OUT_PATH}")

    if query:
        matches = table[table["player"].str.lower() == query.lower()]
        if matches.empty:
            print(f"No player matching '{query}' with >= {MIN_MINUTES} minutes.")
            return
        print(f"\nMost similar to {query} "
              f"({matches['team'].iloc[0]}, {matches['role'].iloc[0]}):")
        for _, r in matches.sort_values("rank").iterrows():
            print(f"  {r['rank']:>2}. {r['similar_player']:<20} "
                  f"{r['similar_team']:<28} sim={r['similarity']:.3f}")


if __name__ == "__main__":
    main()
