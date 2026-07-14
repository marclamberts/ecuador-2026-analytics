# Player Analysis — Lamberts Goalkeeper Model

Ecuador 2026 season. 15 goalkeepers qualify for ranking (≥450 minutes played); 14 more were observed below that threshold and are excluded from ranking but present in `goalkeeper_match_value.csv`. Grounded entirely in `goalkeeper_season_value_model.csv` — every number below is pulled from that file, not estimated.

## Headline ranking

| # | Player | Team | Index | Pctile | Min / Mtc | Standout strength | Standout weakness |
|---|---|---|---|---|---|---|---|
| 1 | J. Angulo | Leones FC | 69.5 | 100 | 528 / 5 | Shot-stopping (100) & progressive distribution (100) | Discipline (7) — 3 cards in 5 matches |
| 1 | D. Cabezas | Libertad FC | 69.5 | 93 | 809 / 8 | Save-difficulty-weighted stopping (100) | Reliability, ball security & discipline — three-way tie at 33 |
| 3 | B. Heras | Delfín SC | 68.2 | 87 | 720 / 7 | Big-chance denial, reliability, sweeper activity — three 100s | Ball security & discipline (20 each) — 1 error, 2 cards in 720 minutes |
| 4 | R. Rodríguez | CSD Macará | 61.5 | 80 | 704 / 7 | Balanced across categories, no extreme weakness | Penalty save ability (10) — 0 saves on 2 penalties faced |
| 5 | G. Napa | Guayaquil City FC | 60.9 | 73 | 1141 / 11 | Availability (100) — ever-present, highest distribution volume (424 passes) | Distribution accuracy (7) — worst in the pool at −6.17 pass value over expected |
| 6 | R. Formento | Mushuc Runa SC | 58.7 | 67 | 1031 / 10 | Distribution accuracy (100), highest xT from passing (20.6) | Reliability, claiming/command, penalty saves & discipline — four-way tie at 27 |
| 7 | F. Ferrero | CD Cuenca | 56.4 | 60 | 719 / 7 | Solid shot-stopping volume | Reliability (7) — most match-to-match GPAE volatility in the pool |
| 8 | R. Romo | CD Universidad Católica | 49.9 | 53 | 697 / 7 | Discipline (100) — zero cards, zero fouls | Big-chance denial (10) — worst in the pool |
| 9 | S. Razzeto | CD Técnico Universitario | 48.8 | 47 | 923 / 9 | Even, unremarkable across categories | Distribution involvement (7) — fewest passes relative to minutes |
| 10 | R. Silva | Orense SC | 48.5 | 40 | 840 / 8 | Ball security & discipline (93 each) | Reliability (13) — most volatile shot-stopping match to match |
| 11 | G. Valle | LDU Quito | 48.3 | 33 | 522 / 5 | — | Availability (7) — thinnest sample in the ranked pool; progressive distribution (7) |
| 12 | J. Contreras | Barcelona SC | 43.1 | 27 | 715 / 7 | — | Claiming/command (7) — worst in the pool, 0 claims all season |
| 13 | F. Zambrano | Manta FC | 40.4 | 20 | 607 / 6 | Penalty saves (97) | Shot-stopping (GPAE −2.98) & distribution involvement — tied at 13 |
| 14 | A. Quintana | CSD Independiente del Valle | 39.1 | 13 | 917 / 9 | Claiming/command (100) — busiest, most commanding on crosses | Shot-stopping (7) — worst GPAE in the pool at −4.89 |
| 15 | H. Piedra | SD Aucas | 37.5 | 7 | 1124 / 11 | Ball security (100) — zero tagged errors, lowest turnover rate per 90 in the pool | Save-difficulty-weighted stopping & sweeper activity — tied at 7, despite most minutes played |

## Reading the top of the table

**J. Angulo and D. Cabezas are statistically tied (69.5 each)** but are different players. Angulo's index rests on 5 matches — his shot-stopping and progressive-distribution scores are both perfect 100s, but `sample_size_confidence.png` flags him as below-median shots faced (79 shots faced vs. a pool median of ~88), so some of that separation could compress with more matches. Cabezas has nearly double the sample (809 min / 8 matches, 155 shots faced — comfortably above median) and still lands in the same tier, which is the more trustworthy of the two ties. If you need to pick one for a single season-defining decision, Cabezas is the safer bet on data volume alone; Angulo is the higher-ceiling small-sample story.

**B. Heras at #3 (68.2) is the most rounded elite profile in the pool** — three separate 100-scores (big-chance denial, reliability, sweeper activity), meaning he isn't just good on volume, he's *consistent* match to match and specifically excellent on the highest-danger shots he faces. His actual weak points are ball security and discipline (both score 20) — 1 tagged error plus a below-average distribution turnover rate, and 2 cards inside just 720 minutes. His distribution accuracy is unremarkable but not a weakness (53rd percentile) — he's a shot-stopper and sweeper first, and the discipline/security dip is the real price of that profile.

## The submodel specialists (best-in-pool per category)

| Submodel | Leader | What it says about them |
|---|---|---|
| Shot-Stopping (Goals Prevented) | J. Angulo | Best goals-prevented-vs-PSxG rate in the pool |
| Save Difficulty-Weighted | D. Cabezas | Saves the *hardest* shots he faces, not just the easy ones |
| Big-Chance Denial | B. Heras | Best save rate specifically on xG ≥ 0.30 chances |
| Reliability | B. Heras | Least match-to-match volatility in shot-stopping output |
| Claiming/Command | A. Quintana | Most claims+punches+smothers relative to crosses faced |
| Sweeper Activity | B. Heras | Most proactive off-line defending (sweeper actions + pickups) |
| Distribution Involvement | G. Napa | Highest raw pass volume (424 passes, 11 matches) |
| Distribution Accuracy | R. Formento | Best pass value over expected (xPass model) |
| Progressive Distribution | J. Angulo | Most xT generated from his own passing |
| Ball Security (inverted error risk) | H. Piedra | Zero tagged errors and lowest turnover rate per 90 |
| Penalty Saves | B. Heras | Best Bayesian-shrunk penalty save rate |
| Discipline | R. Romo | Zero cards, zero fouls all season |
| Availability | G. Napa | Played closest to 100% of his team's possible minutes |

B. Heras shows up as the category leader **three times** (big-chance denial, reliability, sweeper activity) — the most of anyone. J. Angulo (goals prevented, progressive value) and G. Napa (distribution volume, availability) each lead twice; everyone else leads at most once. See `visuals/submodel_specialists_board.png` for the full leaderboard, and `visuals/composite_score_decomposition.png` for how Heras's shot-stopping segment stacks up against the rest of the pool.

## Two profiles worth flagging specifically

**G. Napa — volume without quality in distribution.** He's the most durable keeper in the league (100 availability, most minutes, most matches) and passes more than anyone, but his `pass_value_over_expected` is **−6.17**, the single worst number of any submodel-metric pair in the entire pool (next-worst is A. Quintana at −0.52 — Napa isn't just last, he's an outlier). He's a keeper who's constantly involved in the build-up but is actively giving away value relative to what an average pass in his situations should produce. Worth checking on the eye test whether that's forced long clearances under pressure (which the model would score as low-value even if tactically sound) or genuine technical passing risk.

**A. Quintana and H. Piedra — busy but leaky.** Both rank in the bottom two overall, and both have a real defensible strength (Quintana: best claiming/command in the pool; Piedra: best ball security, zero errors) that gets outweighed by the largest single component in the composite weighting — shot-stopping. Quintana's GPAE (−4.89) and Piedra's (−4.27) are the two worst in the league. `visuals/shot_stopping_diagnostic.png` shows both sitting well above the "conceded more than PSxG predicted" line. This could reflect genuine shot-stopping weakness, weak defensive cover in front of them, or both — the model can't separate keeper skill from team context (see the Caveats section of the main README), so this is a flag for video review, not a final verdict.

## Small-sample caveats

Per `visuals/minutes_threshold_sensitivity.png` and `sample_size_confidence.png`:

- **J. Angulo (5 matches, 79 shots faced) and G. Valle (5 matches, 66 shots faced)** have the thinnest samples in the ranked pool. Angulo's #1 rank should be read as "best signal so far," not "proven best."
- **R. Rodríguez has faced only 2 penalties** and saved neither — the Bayesian shrinkage pulls his penalty-save score away from 0%, but with n=2 that submodel contributes almost no real information about him specifically.
- Keepers with **0 big chances saved out of few faced** (e.g., R. Romo: worst big-chance-denial score off a small sample) are more exposed to one or two unlucky/lucky moments than keepers like B. Heras or D. Cabezas, who've faced double-digit big chances.

## How to use this

Start from `goalkeeper_value_rankings.png` for the headline order, cross-check any name you're relying on against `sample_size_confidence.png` for how much data backs it, then use the per-player pizza chart in `player_visuals/` for the full 13-submodel profile before making a call. `composite_score_decomposition.png` is the fastest way to see *why* two similarly-ranked keepers (like Angulo and Cabezas) got there through different routes.
