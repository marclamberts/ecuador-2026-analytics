"""
Automatic loader for a season data folder.

Expects the folder to follow the season-data convention used across the
Lamberts analytics pipeline::

    <season_dir>/
      Aggregated/
        player_match_metrics.csv   (must include a position_group column)
        team_match_metrics.csv
      Danger/
        YYYY-MM-DD_*_danger_models.csv   (one per match; per-shot xg/psxg/xgot)
      Event/
        *.json                            (raw event feed, one per match)

``SeasonData.load()`` reads everything eagerly and validates the
required columns are present, so a malformed season folder fails fast
with a specific error instead of surfacing as a confusing ``KeyError``
deep inside the model code. Danger files are matched with a
date-prefixed glob so that unrelated multi-season/multi-league cache
files sitting in the same folder (e.g. a stray ``all_x_danger_models.csv``)
are not mistaken for a single match's shot file.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import pandas as pd

REQUIRED_PLAYER_MATCH_COLS = {
    "season", "match_file", "date", "match", "team_id", "team",
    "player_id", "player", "minutes", "position_group",
}
REQUIRED_TEAM_MATCH_COLS = {"match_file", "team_id", "crosses_into_box"}
REQUIRED_SHOT_COLS = {
    "match_file", "contestant_id", "is_on_target", "is_goal", "is_penalty",
    "xg", "psxg", "xgot",
}

DANGER_FILE_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*_danger_models.csv"

ERROR_TYPE_ID = 51


class SeasonDataError(RuntimeError):
    """Raised when a season folder is missing an expected file or column."""


@dataclass
class SeasonData:
    """Everything the goalkeeper value model needs, loaded from one season folder."""

    season_dir: pathlib.Path
    player_match: pd.DataFrame = field(repr=False)
    team_match: pd.DataFrame = field(repr=False)
    shots: pd.DataFrame = field(repr=False)
    event_files: list = field(repr=False)

    @classmethod
    def load(cls, season_dir) -> "SeasonData":
        season_dir = pathlib.Path(season_dir).resolve()
        agg_dir = season_dir / "Aggregated"
        danger_dir = season_dir / "Danger"
        event_dir = season_dir / "Event"

        player_match = cls._read_csv(agg_dir / "player_match_metrics.csv", REQUIRED_PLAYER_MATCH_COLS)
        team_match = cls._read_csv(agg_dir / "team_match_metrics.csv", REQUIRED_TEAM_MATCH_COLS)
        shots = cls._load_shots(danger_dir)

        event_files = sorted(event_dir.glob("*.json"))
        if not event_files:
            raise SeasonDataError(f"No event JSON files found under {event_dir}")

        return cls(
            season_dir=season_dir, player_match=player_match, team_match=team_match,
            shots=shots, event_files=event_files,
        )

    @staticmethod
    def _read_csv(path: pathlib.Path, required_cols) -> pd.DataFrame:
        if not path.exists():
            raise SeasonDataError(f"Expected file not found: {path}")
        df = pd.read_csv(path)
        missing = required_cols - set(df.columns)
        if missing:
            raise SeasonDataError(f"{path} is missing required columns: {sorted(missing)}")
        return df

    @staticmethod
    def _load_shots(danger_dir: pathlib.Path) -> pd.DataFrame:
        if not danger_dir.exists():
            raise SeasonDataError(f"Expected directory not found: {danger_dir}")
        files = sorted(danger_dir.glob(DANGER_FILE_GLOB))
        if not files:
            raise SeasonDataError(
                f"No date-prefixed *_danger_models.csv files found under {danger_dir} "
                f"(expected filenames like 'YYYY-MM-DD_Home - Away_danger_models.csv')"
            )
        frames = []
        for f in files:
            d = pd.read_csv(f)
            missing = REQUIRED_SHOT_COLS - set(d.columns)
            if missing:
                raise SeasonDataError(f"{f} is missing required columns: {sorted(missing)}")
            d["match_file"] = f.name.replace("_danger_models.csv", ".json")
            frames.append(d)
        return pd.concat(frames, ignore_index=True)

    def goalkeepers(self) -> pd.DataFrame:
        """Per-match rows for players tagged as goalkeepers in the lineup."""
        return self.player_match[self.player_match["position_group"] == "GK"].copy()

    def error_events_by_keeper(self, player_ids: set) -> pd.DataFrame:
        """Direct parse of typeId-51 ('error') events for a set of player ids."""
        rows = []
        for f in self.event_files:
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            for e in data.get("event", []) or []:
                if e.get("typeId") == ERROR_TYPE_ID and e.get("playerId") in player_ids:
                    rows.append({"match_file": f.name, "player_id": e.get("playerId"), "errors": 1})
        if not rows:
            return pd.DataFrame(columns=["match_file", "player_id", "errors"])
        df = pd.DataFrame(rows)
        return df.groupby(["match_file", "player_id"], as_index=False)["errors"].sum()
