# Positional Role & Off-Ball Model

Clusters outfield players (≥450 minutes) into tactical role archetypes using
where they act on the pitch and how involved they are off the ball, rather
than raw production stats (goals, assists).

Built by `Scripts/build_position_role_model.py`.

## Method

1. **Broad position** (GK/DEF/MID/FWD) is inferred from formation qualifiers
   on typeId-34 events, same heuristic as `player_templates.py`.
2. **Spatial features** are computed directly from the event stream: average
   (x, y) location of each player's defensive actions, pass attempts,
   received passes (heuristic: next same-team touch after a completed pass),
   and shots, plus overall touch dispersion (a mobility/roaming proxy).
3. **Off-ball features** are pulled from `Aggregated/player_season_metrics.csv`
   (touch-zone shares, pressing-height shares, reception volume, progressive
   runs/carries, aerial involvement, crossing/dribbling rates) — signals about
   *positioning and involvement*, deliberately excluding goals/assists/xG so
   role doesn't collapse into "good vs. bad player."
4. **KMeans** clusters each broad position group separately on the
   standardized feature set. Cluster count is fixed per group (DEF/MID: 4,
   FWD: 3, capped by group size) rather than silhouette-maximized — on this
   feature space silhouette score keeps favoring the least informative k=2
   split all the way out, since real playing styles are a continuum, not
   tight blobs. Silhouette is still recorded per group as a diagnostic.
5. **Cluster labels** are auto-generated from each centroid's most extreme
   standardized features (top 3, mapped through `FEATURE_TAGS`), e.g.
   "Wide Focus + Wide Receiver + Wide Passer" for overlapping full-backs.

## Outputs

- `player_role_features.csv` — full engineered feature matrix per player.
- `player_role_assignments.csv` — player, team, position, cluster id, role label.
- `role_cluster_profiles.csv` — centroid feature profile (z-score + raw) per cluster.
- `role_model.joblib` — `{position: {scaler, kmeans, features, k, silhouette}}`.
- `model_meta.json` — run metadata (thresholds, feature list, per-group k/silhouette).
- `visuals/role_clusters_<POSITION>.png` — PCA scatter of players colored by role.
- `visuals/role_cluster_profiles.png` — standardized feature heatmap across all clusters.

## Caveats

- Reception locations are a heuristic (next same-team event after a completed
  pass), not a tracking-derived off-ball location — it's a proxy, not ground truth.
- No tracking data exists for this competition, so "off-ball" here means
  positioning/involvement proxies derived from the event stream (touch zones,
  reception volume, pressing height), not physical off-ball movement (distance
  run, sprints, space created).
- Cluster labels are auto-generated and best-effort; treat them as a starting
  point and re-derive from `role_cluster_profiles.csv` if a label looks off.
- Requires ≥450 minutes played, so squad players and recent debutants are excluded.
