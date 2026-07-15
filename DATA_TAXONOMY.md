# Ecuador 2026 — Event Data Taxonomy

Maps this repo's event data, models, and outputs onto the Waltzing Analytics
event-data structure: **Data** (Players / Performance / Tactics) and
**Metrics** (Scouting / Recruitment / Similarity / Benchmarking / GK).

Source: Opta/StatsPerform, LigaPro Serie A Ecuador 2026 — 136 matches,
208,136 events, 425 players, 16 teams (`Aggregated/metrics_meta.json`).

```
Event Data
├── Data
│   ├── Players      → Prematch, Post match, Set pieces, GK
│   ├── Performance   → Physical
│   └── Tactics       → Style
└── Metrics
    ├── Scouting
    ├── Recruitment model
    ├── Similarity
    ├── Benchmarking
    └── GK
```

## Data

### Players

Raw and per-player event data, split by match phase.

| Branch | Where it lives | Notes |
|---|---|---|
| Prematch | `Scripts/Ecuador 2026 Matches.csv` | Fixture list / schedule used to drive per-match pulls. |
| Post match | `Event/*.json` (one file per fixture, `matchDetails` + `event`) | Raw Opta event stream per match — the base layer everything else is built from. |
| Set pieces | `Aggregated/metric_dictionary.csv` (`family = set_pieces`, 59 metrics: corners, first/second phase, set-piece passes) · `Scripts/corner_routines.py`, `set_piece_second_balls.py`, `long_throwins.py` | Corner and throw-in delivery, second-ball recovery, first/second-phase xG. |
| GK | `Danger Model/xg_output/` (PSxG, multi-outcome save models, shot suppression) · `Scripts/goalkeeper_buildup.py` | Goalkeeper-specific raw actions (saves, sweeper actions, pickups) — see also Metrics → GK below for the modeled/aggregated view. |

Player-level visuals live in `Ecuador Viz/` (currently built out for `E_Mero`)
and `Visuals/Player_E_Mero/`.

### Performance

#### Physical

| Where it lives | Notes |
|---|---|
| `Aggregated/metric_dictionary.csv` (`family = progression`, 43 metrics) | Carries, progressive distance, accelerations — the closest proxy to physical output available from event (non-tracking) data. |
| `Scripts/progressive_carries_final_third.py`, `carries_wide_to_box.py`, `switches.py` | Ball-progression and off-the-ball movement proxies. |

No tracking-data physical metrics (distance covered, sprints, top speed)
exist in this repo — everything under Physical is event-derived (carries,
accelerations, progression), not GPS/tracking-based.

### Tactics

#### Style

| Where it lives | Notes |
|---|---|
| `Scripts/team_directness.py`, `buildup_patterns.py`, `pass_network.py`, `pass_network_clusters.py`, `field_tilt_trend.py`, `zone_control.py`, `relationism_index.py`, `relation_position_quadrant.py` | Team style outputs: directness, buildup shape, passing network structure, territorial control, "relationism" positional index. |
| `Aggregated/metric_dictionary.csv` (`family = possession`, 68 metrics) | Passing volume/accuracy, buildup involvement — feeds the style scripts above. |
| `Ecuador Team Viz/*_ecuador2026.png`, `*_independiente_del_valle.png` | Rendered style outputs at league level (`ecuador2026`) and club level (`independiente_del_valle`, the pilot team). |

## Metrics

| Branch | Where it lives | Notes |
|---|---|---|
| Scouting | `Scripts/player_templates.py`, `Scripts/Scouting_Report_Universal.ipynb` | Templated player scouting reports built on top of `Aggregated/player_season_metrics.csv`. |
| Recruitment model | `GDA/` (`gda_player_summary.csv`, `gda_action_values.csv`, `gda_zone_values.csv`, `gda_model_meta.json`) | Goal Difference Added = on-pitch GD + Markov possession action value; closest thing here to a recruitment/valuation signal. `Cross Models/models/*.pkl` (cross completion, chance creation, delivery value, goal contribution) are role-specific recruitment sub-models. |
| Similarity | `Scripts/player_similarity.py` → `Aggregated/player_similarity.csv` | Cosine similarity over the engineered per-90/index metrics in `player_season_core.csv`, computed within each role group (Finisher/Ball-Winner/Creator/Progressor/Two-Way Connector), minutes-weighted per player, min. 450 minutes. `python3 player_similarity.py "E. Mero"` prints the top matches for one player. |
| Benchmarking | `Aggregated/player_season_metrics.csv`, `team_season_metrics.csv`, `metric_dictionary.csv` (`_pctile` suffixed metrics), `Scripts/passing_metrics_compared.py`, `home_away_rating_gap.py`, `schedule_difficulty.py`, `season_projection.py` | League-wide percentile ranks and cross-team/cross-player comparisons. |
| GK | `Danger Model/xg_output/` (`model_psxg*.pkl`, `model_multi_outcome.pkl`, `model_pontarget.pkl`, `multi_outcome_save.png`) · `Ecuador Team Viz/goalkeeper_buildup_ecuador2026.png` | PSxG and shot-outcome models double as the GK metrics layer (save quality vs. expected, shot suppression). |

## Supporting models (not on the diagram, cross-cutting)

These sit underneath both **Data** and **Metrics** branches above and are
shared infrastructure rather than a leaf category themselves:

- `Danger/`, `Danger Model/` — xG, xGOT, PSxG, Bayesian finishing.
- `ClearanceLandingModel/` — headed clearance landing + outcome models (defending/set-piece crossover).
- `Cross Models/` — open-play cross event scoring and outcome models (attacking + set-piece crossover).
- `Cache/` — compiled model artifacts and `.xt_grid_cache.json` (xT zone grid).

## Gaps vs. the target taxonomy

- **Prematch** player data is limited to the fixture list; there's no
  pre-match projection/lineup-news layer yet.
