import json

from setpieces.teams import team_ids_from_filenames


def test_team_ids_from_filenames(tmp_path):
    match_a = {
        "event": [
            {"contestantId": "t_A", "typeId": 1},
            {"contestantId": "t_B", "typeId": 1},
        ]
    }
    match_b = {
        "event": [
            {"contestantId": "t_A", "typeId": 1},
            {"contestantId": "t_C", "typeId": 1},
        ]
    }
    match_c = {
        "event": [
            {"contestantId": "t_B", "typeId": 1},
            {"contestantId": "t_C", "typeId": 1},
        ]
    }
    path_a = tmp_path / "2026-01-01_Team A - Team B.json"
    path_b = tmp_path / "2026-01-08_Team C - Team A.json"
    path_c = tmp_path / "2026-01-15_Team B - Team C.json"
    path_a.write_text(json.dumps(match_a))
    path_b.write_text(json.dumps(match_b))
    path_c.write_text(json.dumps(match_c))

    team_to_id = team_ids_from_filenames([str(path_a), str(path_b), str(path_c)])
    assert team_to_id["Team A"] == "t_A"
    assert team_to_id["Team B"] == "t_B"
    assert team_to_id["Team C"] == "t_C"
