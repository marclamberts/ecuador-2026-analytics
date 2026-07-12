# Headed Clearance Landing Insights

This folder turns the headed-clearance landing model into analysis outputs.

## Core files

- `headed_clearance_insights.csv`: one row per headed clearance with model prediction, actual landing, over-expectation metrics, and next-action outcome flags.
- `headed_clearance_player_summary.csv`: player-level summary.
- `headed_clearance_team_summary.csv`: team-level summary.
- `headed_clearance_zone_summary.csv`: summary by actual landing zone.

## Main metrics

- `length_oe`: actual clearance length minus expected clearance length.
- `territory_oe`: actual landing x minus predicted landing x. Positive means further upfield than expected.
- `wide_oe`: actual landing distance from the pitch centerline minus expected distance from centerline. Positive means wider than expected.
- `danger_zone_avoidance`: 1 when the model expected a defensive-central landing but the actual clearance avoided it.
- `same_team_first_touch_rate`: how often the clearing team got the first recorded touch within 10 seconds.
- `opponent_shot_10s_rate`: how often the opponent shot within 10 seconds.
- `clearance_value_oe`: weighted composite combining territory, length, width, first-touch outcome, danger-zone avoidance, and danger after the clearance.

## Interpretation

Use `clearance_value_oe` as a directional analysis score, not as a universal truth metric. It is built for comparing headed clearances in this Ecuador 2026 event feed and should be reviewed alongside the maps and next-action context.
