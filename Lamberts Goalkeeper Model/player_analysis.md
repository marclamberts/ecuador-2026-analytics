# Player Analysis — Lamberts Goalkeeper Model

Ecuador 2026 season. 29 distinct goalkeepers were observed; 18 qualify for ranking (≥450 minutes played). Grounded entirely in `goalkeeper_season_value_model.csv` — every number below is pulled from that file, not estimated, and every strength/weakness claim is cross-checked against `visuals/player_strength_weakness_board.png` and `visuals/submodel_specialists_board.png` rather than eyeballed from raw stats.

> **Note on a data correction:** an earlier version of this document was built before a bug fix to the model's season-aggregation step (see `lamberts_goalkeeper_model`'s `model.py`) — the aggregator's team/player *name* fields are occasionally wrong for a given match row (mislabeled team, or the player name falling back to a raw id string), which silently fragmented most keepers' seasons into bogus single-match sub-groups keyed off the wrong label. Grouping now keys off `team_id`/`player_id` instead, which are reliable. That fix changed minutes, matches, and every submodel for nearly every keeper in the pool (affected 21 of 29 identities, ~95% of match rows) and reshuffled the ranking substantially — e.g. F. Ferrero went from 719 minutes/7 matches to 1734 minutes/17 matches. This version reflects the corrected data.

## Headline ranking

| # | Player | Team | Index | Pctile | Min / Mtc | Strength | Weakness |
|---|---|---|---|---|---|---|---|
| 1 | J. Angulo | Leones FC | 72.2 | 100 | 828 / 8 | Big-Chance Denial (100) | Discipline (6) |
| 2 | F. Ferrero | CD Cuenca | 69.8 | 94 | 1734 / 17 | Distribution Vol. (94) | Reliability (22) |
| 3 | L. Nazareno | Leones FC | 67.0 | 89 | 949 / 9 | Save Difficulty (94) | Availability (17) |
| 4 | G. Napa | Guayaquil City FC | 66.4 | 83 | 1583 / 16 | Goals Prevented (100) | Distribution Acc. (6) |
| 5 | J. Contreras | Barcelona SC | 64.6 | 78 | 1652 / 16 | Goals Prevented (83) | Claiming/Command (11) |
| 6 | B. Heras | Delfín SC | 62.8 | 72 | 1230 / 12 | Sweeper Activity (100) | Claiming/Command (6) |
| 7 | G. Valle | LDU Quito | 59.4 | 67 | 1440 / 14 | Claiming/Command (83) | Progressive Value (22) |
| 8 | P. Ortíz | CS Emelec | 56.9 | 61 | 1540 / 15 | Discipline (94) | Ball Security (28) |
| 9 | R. Romo | CD Universidad Católica | 55.8 | 56 | 1411 / 14 | Reliability (100) | Big-Chance Denial (6) |
| 10 | D. Cabezas | Libertad FC | 52.3 | 50 | 1732 / 17 | Sweeper Activity (94) | Ball Security (17) |
| 11 | R. Formento | Mushuc Runa SC | 50.4 | 44 | 1763 / 17 | Distribution Acc. (100) | Reliability (6) |
| 12 | R. Silva | Orense SC | 48.9 | 39 | 1660 / 16 | Discipline (89) | Reliability (17) |
| 13 | R. Rodríguez | CSD Macará | 46.3 | 33 | 1408 / 14 | Reliability (72) | Penalty Saves (8) |
| 14 | S. Razzeto | CD Técnico Universitario | 43.4 | 28 | 1426 / 14 | Ball Security (94) | Distribution Vol. (6) |
| 15 | A. Quintana | CSD Independiente del Valle | 36.9 | 22 | 1311 / 13 | Claiming/Command (100) | Goals Prevented (11) |
| 16 | F. Zambrano | Manta FC | 35.8 | 17 | 1621 / 16 | Availability (78) | Big-Chance Denial (11) |
| 17 | H. Piedra | SD Aucas | 33.8 | 11 | 1749 / 17 | Ball Security (100) | Save Difficulty (6) |
| 18 | J. Cárdenas | Delfín SC | 27.1 | 6 | 511 / 5 | Sweeper Activity (78) | Goals Prevented (6) |

## Reading the top of the table

**J. Angulo leads outright (72.2)**, with a perfect Big-Chance Denial score (100) and one of the two best penalty-save records in the pool — but his weakest dimension, Discipline (6th percentile), reflects 3 cards across just 8 matches, a real availability risk if it continues. His 132 shots faced is now a reasonably solid sample (up from a thin 79 before the data fix), though still below the pool median (~171).

**F. Ferrero at #2 (69.8) has the largest sample in the pool that isn't padded by weak performance** — 1734 minutes across 17 matches, the joint-most matches played, with a strong all-around profile: he's the "elite all-round" reference point in `visuals/shot_stopping_vs_secondary_skills.png` (above-average on both shot-stopping and every secondary category simultaneously, at his best on Distribution Volume). His only real gap is Reliability (22nd percentile) — his shot-stopping output swings more match to match than most.

**G. Napa is the pool's most decorated specialist** — he leads 3 of the 13 submodels outright (Goals Prevented, Save Difficulty, Distribution Volume), more than anyone else, per `visuals/submodel_specialists_board.png`. But he's also the clearest volume-without-quality outlier in the league: among keepers with above-median pass volume, he has the *worst* pass value over expected in the entire dataset (-0.36 per 90, `visuals/distribution_volume_vs_quality.png`). He's constantly involved in build-up but that involvement is actively subtracting value relative to what an average pass in his situations should produce — worth checking on video whether that's forced long clearances under pressure (tactically defensible even if the model scores it low) or genuine distribution risk.

## The submodel specialists (leader per category)

| Submodel | Leader | Score |
|---|---|---|
| Goals Prevented | G. Napa ★ | 100 |
| Save Difficulty | G. Napa ★ | 100 |
| Big-Chance Denial | J. Angulo ★ | 100 |
| Reliability | R. Romo ★ | 100 |
| Penalty Saves | J. Angulo ★ | 97 |
| Claiming/Command | A. Quintana | 100 |
| Sweeper Activity | B. Heras | 100 |
| Distribution Vol. | G. Napa ★ | 100 |
| Distribution Acc. | R. Formento ★ | 100 |
| Progressive Value | R. Formento ★ | 100 |
| Ball Security | H. Piedra | 100 |
| Discipline | R. Romo ★ | 100 |
| Availability | F. Ferrero | 78 |

★ = leads more than one submodel. G. Napa leads the most (3); J. Angulo, R. Romo, and R. Formento each lead 2. No keeper hit 100 on Availability this season — F. Ferrero's 78th percentile is the closest anyone got to playing every possible minute.

## Two profiles worth flagging specifically

**"Busy but leaky" — A. Quintana and R. Formento.** Both have a genuine standout secondary skill (Quintana leads Claiming/Command outright at 100; Formento leads both Distribution Accuracy and Progressive Value at 100 each) that gets outweighed by the model's largest single weight, shot-stopping. Quintana's Goals Prevented score (11th percentile) is his worst dimension; Formento's shot-stopping submodels average in the low 30s despite his distribution numbers being the best in the league. `visuals/shot_stopping_vs_secondary_skills.png` places both clearly in the "weak shot-stopping, strong elsewhere" quadrant — a flag for video review on positioning/shot-stopping technique specifically, not a verdict on their overall goalkeeping.

**F. Ferrero — elite all-round.** The only keeper simultaneously above the pool median on both shot-stopping and every other category combined (see the same chart, top-right quadrant). Combined with the largest reliable sample in the pool (17 matches), he's the season's most complete goalkeeper by this model even though he ranks #2 on the composite index — J. Angulo's #1 spot rests on a higher peak (perfect Big-Chance Denial) rather than broader coverage.

## Small-sample caveats

- **J. Cárdenas (5 matches, 511 minutes, 73 shots faced) is the thinnest sample in the ranked pool** — he only just clears the 450-minute threshold, and his last-place ranking (27.1) should be read with that in mind. Compare `visuals/sample_size_confidence.png` for how his shots-faced total stacks up against the rest of the pool.
- **R. Rodríguez's weakest submodel is Penalty Saves (8th percentile)** off a small number of penalties faced this season — `visuals/bayesian_shrinkage.png` shows how much the shrinkage pulls a score like this toward the league mean, but with only a handful of attempts the submodel still carries limited information about him specifically.
- Several keepers' Big-Chance Denial and Ball Security scores are decided by single-digit shot counts — `visuals/big_chance_waffle_grid.png` makes this concrete: B. Heras's perfect-looking "3/3 big chances saved" record (implied by his overall profile) is a much smaller sample than G. Napa's "11/17."

## How to use this

Start from `goalkeeper_value_rankings.png` for the headline order, cross-check any name you're relying on against `sample_size_confidence.png` for how much data backs it, then use the per-player pizza chart in `player_visuals/` for the full 13-submodel profile before making a call. `composite_score_decomposition.png` is the fastest way to see *why* two similarly-ranked keepers got there through different routes, and `shot_stopping_pitch_maps.png` / `rolling_form_momentum.png` (in the advanced visuals set) show *where* danger comes from and *when* in the season a keeper's form actually shifted, rather than just the season-long average.
