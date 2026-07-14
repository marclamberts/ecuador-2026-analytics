# Lamberts Goalkeeper Model

A composite value model for Ecuador 2026 goalkeepers built from 13 submodels
covering shot-stopping, claiming/sweeping, distribution, risk, and
availability. Built by `Scripts/build_goalkeeper_value_model.py`.

## Visuals

Built by `Scripts/create_goalkeeper_value_visuals.py` (requires
`goalkeeper_season_value_model.csv` and `submodel_definitions.csv` to
already exist — run the build script first).

- `visuals/goalkeeper_value_rankings.png`: league-wide ranking of all
  ranked keepers on both the percentile-based `goalkeeper_value_index`
  and the z-score-based `goalkeeper_value_index_zscore`, colored above
  vs. below the pool mean.
- `visuals/submodel_weights.png`: the 13 submodel weights, color-coded
  by category (Shot-Stopping / Claiming & Sweeping / Distribution /
  Risk & Availability).
- `player_visuals/{player}_pizza.png`: one pizza chart per ranked
  keeper, one slice per submodel (percentile score, colored by
  category), with the raw metric value labeled on each slice and the
  composite index/percentile in the header.

## Core files

- `goalkeeper_match_value.csv`: one row per keeper-match with raw counts
  (shots faced, PSxG faced, goals conceded, claims, sweeper actions,
  distribution, errors, etc.).
- `goalkeeper_season_value_model.csv`: one row per keeper-season with all
  13 submodel scores in two parallel forms — a 0-100 percentile
  (`{submodel}_score`) and a standardized z-score (`{submodel}_zscore`)
  — plus two composite indexes built the same way from each: the
  percentile-based `goalkeeper_value_index` (with its own percentile,
  `goalkeeper_value_index_pctile`) and the z-score-based
  `goalkeeper_value_index_zscore`. Ranking is restricted to keepers with
  >= 450 minutes (~5 full matches) to avoid small-sample noise; keepers
  below that threshold are dropped from this file (their raw stats are
  still in the match-level file).
- `submodel_definitions.csv`: the 13 submodels, their formulas, and their
  weight in the composite index.

## Data sources

- `Aggregated/player_match_metrics.csv` (`position_group == 'GK'`) for raw
  per-match keeper actions, minutes, and passing/xT metrics.
- `Danger/*_danger_models.csv` for trained per-shot `xg`/`psxg`/`xgot`
  model output, used for shot-stopping value.
- `Aggregated/team_match_metrics.csv` for opponent cross volume, used to
  derive crosses faced.
- `Event/*.json` `typeId == 51` ("error") events, parsed directly since
  the aggregator doesn't break these out as their own column.

Every match in this dataset has exactly one goalkeeper per team for the
full 90 minutes (no in-match keeper changes), so all joins are exact
match-level joins rather than minute-window attribution.

## The 13 submodels

| Submodel | What it measures | Formula |
|---|---|---|
| `shot_stopping_gpae` | Goals Prevented Above Expected | Σ PSxG (on-target shots faced) − goals conceded, per 90 |
| `save_difficulty_weighted` | Credit for saving *hard* shots | Σ xGOT of shots actually saved, per 90 |
| `big_chance_denial` | Stopping the most dangerous chances | Bayesian-shrunk save rate on faced shots with xG ≥ 0.30 |
| `shot_stopping_reliability` | Consistency, not just magnitude | inverse of match-to-match std dev of GPAE per 90 |
| `claiming_command` | Commanding the box on crosses | Bayesian-shrunk (claims+punches+smothers) rate per cross faced |
| `sweeper_activity` | Proactive off-line defending | (keeper sweeper actions + keeper pickups) per 90 |
| `distribution_involvement` | Build-up outlet volume | passes attempted per 90 |
| `distribution_accuracy` | Passing quality above expected | pass value over expected (xPass model), per 90 |
| `progressive_distribution` | Value added by own passing | xT generated from the keeper's own passes, per 90 |
| `error_risk` | Cost of mistakes (inverted) | 2×(typeId-51 errors per 90) + turnovers on own passes per 90 |
| `penalty_save_ability` | Penalty-save skill | Bayesian-shrunk penalty save rate (prior weight: 8 penalties) |
| `discipline_risk` | Availability risk from cards/fouls (inverted) | fouls per 90 + 3×cards per 90 |
| `availability` | Durability as the starter | minutes played ÷ team's total possible minutes |

Each submodel's raw metric is converted two ways among ranked keepers
(higher is always better after inversion where noted):

- **Percentile** (`{submodel}_score`, 0-100): where a keeper ranks
  relative to the pool. Robust to outliers, but compresses gaps between
  closely-bunched keepers and expands gaps between keepers who happen to
  sit at sparsely-populated points in the distribution.
- **Z-score** (`{submodel}_zscore`, standardized: `(x - mean) / population std`,
  sign-flipped for inverted metrics): how many standard deviations from
  the pool average a keeper is on that specific metric. Preserves actual
  magnitude of separation, so one keeper being *far* ahead on a submodel
  shows up as a large z-score even if it's still just "1st of 15" in
  percentile terms — but it's more sensitive to outliers than the
  percentile version, and a submodel with a near-constant metric across
  all keepers is defined to have all z-scores of 0 rather than dividing
  by a near-zero standard deviation.

Both are blended into a composite index using the weights in
`submodel_definitions.csv` (shot-stopping submodels carry the largest
combined weight, since that's the core of the position; discipline and
availability are minor modifiers): `goalkeeper_value_index` from the
percentile scores, and `goalkeeper_value_index_zscore` from the
z-scores. The two composites usually agree on the top/bottom of the
ranking but can disagree on ordering in the middle of the pack — that
disagreement itself is informative (a keeper who's "solidly above
average" everywhere vs. one who's "1st place by a hair" on a couple of
submodels can land differently on each scale).

## Caveats

- `psxg`/`xgot` come from the trained shot model in `Danger/`, not a
  bespoke goalkeeper positioning model — it captures shot placement and
  danger, not GK-specific reaction/positioning data (not available in the
  base event feed).
- `crosses_faced` is the opponent's total `crosses_into_box` for that
  match, not filtered to crosses actually contested by the keeper.
- Small-sample submodels (`big_chance_denial`, `penalty_save_ability`)
  use Bayesian shrinkage toward the league rate; interpret keepers with
  few big chances/penalties faced cautiously even after shrinkage.
- Composite weights are an analytical judgment call, not fitted to an
  external outcome (e.g. win probability) — treat `goalkeeper_value_index`
  as a transparent, adjustable blend, not a ground-truth rating.
