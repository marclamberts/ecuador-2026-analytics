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

### Methodology / diagnostic visuals

Built by `Scripts/create_goalkeeper_model_diagnostics.py` — these explain
*how the model works and whether its inputs are sound*, as opposed to the
player-profile visuals above.

- `visuals/shot_stopping_diagnostic.png`: PSxG faced vs. goals conceded
  per keeper — what `shot_stopping_gpae` is actually measuring, and who's
  over/underperforming the shot quality they faced.
- `visuals/shot_model_calibration.png`: a reliability curve (mean
  predicted PSxG vs. actual goal rate, in 8 quantile bins) for the
  trained shot model every shot-stopping submodel depends on, plus its
  PSxG distribution. Validates the model behind the model.
- `visuals/submodel_correlation_heatmap.png`: pairwise Spearman
  correlation across all 13 submodel scores — checks whether they
  capture distinct goalkeeping skills or are redundant. (Notably,
  `shot_stopping_gpae` and `sweeper_activity` correlate at 0.78, and
  `error_risk` correlates negatively with several shot-stopping/command
  submodels — worth knowing before trusting the composite as
  "independent" dimensions.)
- `visuals/bayesian_shrinkage.png`: raw vs. shrunk save rate for
  `big_chance_denial` and `penalty_save_ability`, plotted against sample
  size, showing how the shrinkage pulls small samples toward the league
  mean.
- `visuals/composite_score_decomposition.png`: each ranked keeper's
  `goalkeeper_value_index`, stacked-bar decomposed into the 4 category
  contributions (Shot-Stopping / Claiming & Sweeping / Distribution /
  Risk & Availability) that built it.
- `visuals/percentile_vs_zscore_agreement.png`: rank-by-percentile-index
  vs. rank-by-zscore-index for every keeper — shows exactly which
  keepers' ranking depends on which composite method you trust.

Built by `Scripts/create_goalkeeper_model_explainer_visuals.py`:

- `visuals/model_architecture_overview.png`: a one-page infographic —
  data sources → 13 submodels grouped into 4 weighted categories →
  composite index — plus the sample size behind it. The fastest way to
  explain the whole model to someone who hasn't read this README.
- `visuals/submodel_score_distributions.png`: the *raw* metric behind
  each submodel (not the percentile score — percentile ranks are
  uniform by construction, so they can't show whether a submodel
  actually separates keepers). Each row is one submodel, min-max scaled
  to its own range purely for layout; actual min/max values are labeled
  at each end. Clustered dots mean keepers are genuinely close on that
  metric; spread-out dots mean real separation exists in the data.
- `visuals/minutes_threshold_sensitivity.png`: how many keepers would
  qualify for ranking at every minutes cutoff from 0 to 1200, justifying
  why 450 minutes (~5 matches) was chosen — low enough to keep 15
  keepers, high enough to filter out single-match cameos.
- `visuals/sample_size_confidence.png`: Goalkeeper Value Index plotted
  against shots faced (the sample size behind every shot-stopping
  submodel). Keepers on the left have noisier shot-stopping scores even
  after Bayesian shrinkage — worth checking before treating small
  rank differences as meaningful.

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
