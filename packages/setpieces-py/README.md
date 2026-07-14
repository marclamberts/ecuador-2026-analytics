# setpieces (Python)

Extract and analyze football set-piece events -- corners, free kicks,
throw-ins, and penalties -- from Opta MA1-style event data. Also ships a
generalization of the corner-delivery-zone and second-ball-contest
analysis originally built for the Ecuador 2026 dataset in this repo.

No dependencies for the core API. Plotting is optional (matplotlib).

## Install

Clone this repo, then from `packages/setpieces-py`:

```bash
pip install .
# with plotting helpers:
pip install ".[plot]"
```

Or straight from GitHub without cloning first:

```bash
pip install "git+https://github.com/marclamberts/ecuador-2026-analytics.git#subdirectory=packages/setpieces-py"
```

## Data format

Each match is one JSON file shaped like:

```json
{
  "matchDetails": {"scores": {"total": {"home": 2, "away": 1}}},
  "event": [
    {
      "typeId": 1,
      "periodId": 1,
      "timeMin": 23, "timeSec": 5,
      "contestantId": "abc123",
      "playerName": "J. Player",
      "outcome": 1,
      "x": 100.0, "y": 45.2,
      "qualifier": [{"qualifierId": 6, "value": null}]
    }
  ]
}
```

This is the standard Opta MA1 feed shape (`typeId`/`qualifierId` codes,
0-100 normalized `x`/`y`). See `setpieces.codes` for the exact codes used
to identify corners (qualifier 6), free kicks (5), throw-ins (107),
penalties (9), and the shot/contested-action type IDs.

## Quick start

```python
import glob
import setpieces as sp

match = sp.load_match("Event/2026-04-12_CD Cuenca - CSD Independiente del Valle.json")

events = sp.extract_set_pieces(match)  # corners, free kicks, throw-ins, penalties
corners = [e for e in events if e.kind == "corner"]

# Delivery-zone breakdown (near/central/far post x six-yard/edge-of-box,
# mirrored onto one attacking side)
pct = sp.zone_percentages(corners)

# Penalty conversion
print(sp.penalty_summary(events))

# Second-ball contests after a team's own set pieces
raw_events = sp.sorted_events(match)
team_id = corners[0].team_id
contests = sp.find_second_ball_contests(raw_events, events, team_id)
won = sum(1 for c in contests if c.won)
print(f"{won}/{len(contests)} second balls won")
```

### Across many matches

```python
files = sorted(glob.glob("Event/*.json"))
team_to_id = sp.team_ids_from_filenames(files)  # relies on this repo's
                                                 # "DATE_Home - Away.json" naming

all_corners = []
for path in files:
    match = sp.load_match(path)
    all_corners += [e for e in sp.extract_set_pieces(match, kinds=["corner"])]
```

### Plotting (optional)

```python
import setpieces.plot as spplot

ax = spplot.plot_zone_grid(pct, title="Corner delivery zones")
ax = spplot.plot_second_ball_map(contests, title="Second-ball contests")
```

## API

- `load_match(path)`, `load_matches(paths)`, `sorted_events(match)`
- `extract_set_pieces(match, kinds=None) -> list[SetPieceEvent]`
- `zones.classify_zone(end_x, end_y, start_y, params=...)`, `zone_breakdown(events)`, `zone_percentages(events)`
- `find_second_ball_contests(events, set_pieces, team_id, ...) -> list[SecondBallContest]`
- `penalty_summary(events) -> dict`
- `team_ids_from_filenames(paths) -> dict[str, str]` (convenience for this repo's filename convention)
- `distance_m(x0, y0, x1, y1)` -- real-world metres between two 0-100 pitch coordinates

## Tests

```bash
pip install ".[dev]"
pytest
```
