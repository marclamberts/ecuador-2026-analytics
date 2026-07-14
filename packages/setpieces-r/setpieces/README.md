# setpieces (R)

Extract and analyze football set-piece events -- corners, free kicks,
throw-ins, and penalties -- from Opta MA1-style event data. Also ships
an R port of the corner-delivery-zone and second-ball-contest analysis
originally built for the Ecuador 2026 dataset in this repo.

Only `jsonlite` is required for the core API. Plotting is optional
(`ggplot2`).

## Install

Clone this repo, then from R:

```r
install.packages("jsonlite")  # required dependency
install.packages("packages/setpieces-r/setpieces", repos = NULL, type = "source")
```

Or straight from GitHub without cloning first (requires `remotes`):

```r
install.packages("remotes")
remotes::install_github("marclamberts/ecuador-2026-analytics", subdir = "packages/setpieces-r/setpieces")
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
0-100 normalized `x`/`y`). See `R/codes.R` for the exact codes used to
identify corners (qualifier 6), free kicks (5), throw-ins (107),
penalties (9), and the shot/contested-action type IDs.

## Quick start

```r
library(setpieces)

match <- load_match("Event/2026-04-12_CD Cuenca - CSD Independiente del Valle.json")

events <- extract_set_pieces(match)  # corners, free kicks, throw-ins, penalties
corners <- Filter(function(e) e$kind == "corner", events)

# Delivery-zone breakdown (near/central/far post x six-yard/edge-of-box,
# mirrored onto one attacking side)
pct <- zone_percentages(corners)

# Penalty conversion
print(penalty_summary(events))

# Second-ball contests after a team's own set pieces
raw_events <- sorted_events(match)
team_id <- corners[[1]]$team_id
contests <- find_second_ball_contests(raw_events, events, team_id)
won <- sum(vapply(contests, function(c) c$won, logical(1)))
cat(sprintf("%d/%d second balls won\n", won, length(contests)))
```

### Across many matches

```r
files <- sort(list.files("Event", pattern = "\\.json$", full.names = TRUE))
team_to_id <- team_ids_from_filenames(files)  # relies on this repo's
                                               # "DATE_Home - Away.json" naming

all_corners <- do.call(c, lapply(files, function(path) {
  match <- load_match(path)
  extract_set_pieces(match, kinds = "corner")
}))
```

### Plotting (optional)

```r
install.packages("ggplot2")

plot_zone_grid(pct, title = "Corner delivery zones")
plot_second_ball_map(contests, title = "Second-ball contests")
```

## API

- `load_match(path)`, `load_matches(paths)`, `sorted_events(match)`
- `extract_set_pieces(match, kinds = NULL)`, `set_pieces_to_df(events)`
- `classify_zone(end_x, end_y, start_y, params = default_zone_params)`, `zone_breakdown(events)`, `zone_percentages(events)`
- `find_second_ball_contests(events, set_pieces, team_id, ...)`
- `penalty_summary(events)`
- `team_ids_from_filenames(paths)` (convenience for this repo's filename convention)
- `distance_m(x0, y0, x1, y1)` -- real-world metres between two 0-100 pitch coordinates

## Tests

```r
install.packages(c("testthat", "jsonlite"))
devtools::test()
```
