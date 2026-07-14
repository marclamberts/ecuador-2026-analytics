# lamberts-goalkeeper-model

Installable Python package version of the Lamberts Goalkeeper Model. Point
it at a season data folder and it automatically loads everything it needs
and builds the composite Goalkeeper Value Index — no manual wiring of
file paths.

## Install

```bash
pip install -e ./lamberts_goalkeeper_model
```

## Expected season folder layout

`SeasonData.load()` auto-discovers these files under a single season
directory (this is the same layout this repo already uses at its root):

```
<season_dir>/
  Aggregated/
    player_match_metrics.csv   # must include a position_group column
    team_match_metrics.csv
  Danger/
    YYYY-MM-DD_*_danger_models.csv   # one per match, date-prefixed
  Event/
    *.json                            # raw event feed, one per match
```

Files that don't match the expected shape raise a `SeasonDataError` with
a specific message (missing file, missing column) instead of failing
deep inside the model with a confusing `KeyError`.

## Usage

### As a library

```python
from lamberts_goalkeeper_model import build_goalkeeper_value_model

result = build_goalkeeper_value_model("/path/to/season")
result.match_df       # one row per keeper-match
result.season_df      # one row per ranked keeper-season, with all 13
                       # submodel scores/z-scores and the composite index
result.save("/path/to/output")   # writes the 3 CSVs
```

Pass `min_minutes=` to change the ranking cutoff (default 450, ~5 full
matches). You can also load once and reuse the `SeasonData` object:

```python
from lamberts_goalkeeper_model import SeasonData, build_goalkeeper_value_model

season = SeasonData.load("/path/to/season")
result = build_goalkeeper_value_model(season, min_minutes=270)
```

### From the command line

```bash
lamberts-goalkeeper-model /path/to/season --out-dir /path/to/output
```

## What it outputs

- `goalkeeper_match_value.csv` — one row per keeper-match, raw counts.
- `goalkeeper_season_value_model.csv` — one row per ranked keeper-season:
  13 submodel percentile scores (`{submodel}_score`), 13 z-scores
  (`{submodel}_zscore`), and the composite `goalkeeper_value_index` /
  `goalkeeper_value_index_pctile` / `goalkeeper_value_index_zscore`.
- `submodel_definitions.csv` — the 13 submodels, their formulas, and
  their weight in the composite index.

This is the same 13-submodel methodology documented in
`../Lamberts Goalkeeper Model/README.md` — see that file for the full
methodology writeup, caveats, and the visuals built on top of this
output. This package only produces the data; visuals and the written
analysis are separate scripts in `../Scripts/` that read these CSVs.
