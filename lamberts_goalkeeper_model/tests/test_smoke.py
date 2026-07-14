"""
Smoke test: builds the model against this repo's own Ecuador 2026 data
(the repo root is a valid season folder) and checks basic invariants.
Not a substitute for the value-by-value equivalence check that was run
manually against the previously-committed CSVs when this package was
extracted from Scripts/build_goalkeeper_value_model.py -- this just
guards against future regressions breaking the shape of the output.

Run from the repo root: python3 -m pytest lamberts_goalkeeper_model/tests
"""

import pathlib

import pytest

from lamberts_goalkeeper_model import SeasonDataError, build_goalkeeper_value_model

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not (REPO_ROOT / "Aggregated").exists(), reason="repo season data not present in this checkout")
def test_build_against_repo_data():
    result = build_goalkeeper_value_model(REPO_ROOT)

    assert len(result.match_df) > 0
    assert len(result.season_df) > 0
    assert (result.match_df["minutes"] >= 0).all()

    season = result.season_df
    assert (season["minutes"] >= result.min_minutes).all()
    assert season["goalkeeper_value_index"].between(0, 100).all()
    assert season["goalkeeper_value_index_pctile"].between(0, 100).all()

    for submodel in ("shot_stopping_gpae", "claiming_command", "availability"):
        assert f"{submodel}_score" in season.columns
        assert season[f"{submodel}_score"].between(0, 100).all()

    # Ranked by composite index, descending.
    assert list(season["goalkeeper_value_index"]) == sorted(season["goalkeeper_value_index"], reverse=True)


def test_missing_season_dir_raises_specific_error(tmp_path):
    with pytest.raises(SeasonDataError, match="player_match_metrics.csv"):
        build_goalkeeper_value_model(tmp_path)


def test_missing_position_group_column_raises(tmp_path):
    agg = tmp_path / "Aggregated"
    agg.mkdir()
    import pandas as pd

    pd.DataFrame({"season": [], "match_file": [], "date": [], "match": [], "team_id": [], "team": [],
                  "player_id": [], "player": [], "minutes": []}).to_csv(agg / "player_match_metrics.csv", index=False)

    with pytest.raises(SeasonDataError, match="position_group"):
        build_goalkeeper_value_model(tmp_path)
