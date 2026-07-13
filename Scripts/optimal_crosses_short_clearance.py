"""
Optimal crosses: where the resulting headed clearance travels a shorter
distance than the clearance-landing model expects.
======================================================================
For every open-play cross that was defended by a headed clearance (757
events, Cross Models/improved/cross_events_v2.parquet), this applies the
reused headed-clearance landing model (Cross Models/improved/models/
model_clearance_landing_reused.pkl) to get each clearance's expected landing
point, then compares:

  actual_distance    = distance the clearance actually travelled
  predicted_distance = distance the model expected it to travel
  gap                = actual_distance - predicted_distance

A negative gap means the defending team's clearance fell shorter than
expected given the situation (clearer's zone, direction, team, player) --
the ball stays closer to their own goal / the danger zone than a typical
clearance from that position would, which is good for the attacking team's
second-ball recovery. This ranks crosses (and the players/teams/zones that
produce them) by how consistently they force these short, "trapped" headed
clearances.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path('/home/user/ecuador-2026-analytics')
IMPROVED = REPO / 'Cross Models' / 'improved'
OUT_DIR = IMPROVED / 'optimal_crosses'
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(IMPROVED / 'cross_events_v2.parquet')
clr = pd.read_parquet(IMPROVED / 'headed_clearances_v2.parquet')
model_bundle = joblib.load(IMPROVED / 'models' / 'model_clearance_landing_reused.pkl')
model = model_bundle['model']
feature_cols = model_bundle['feature_cols']

team_name_map = joblib.load(REPO / 'Cross Models' / 'models' / 'model_meta.pkl')['team_name_map']

crosses = df[df['is_headed_clearance'] == 1].copy()
crosses['crossing_team'] = crosses['contestant_id'].map(team_name_map).fillna(crosses['contestant_id'])
print(f'{len(crosses)} open-play crosses defended by a headed clearance.')

merged = crosses.merge(
    clr, left_on='clr_event_uid', right_on='event_uid', how='left', suffixes=('', '_clr')
)
# 'team' (the clearing team's contestant id) is also one of the model's own
# categorical features -- keep it intact for prediction, add a display alias
merged['clearing_team'] = merged['team'].map(team_name_map).fillna(merged['team'])
missing = merged['start_x'].isna().sum()
print(f'{missing} crosses could not be matched to a clearance record (dropped).')
merged = merged.dropna(subset=feature_cols).reset_index(drop=True)

pred = model.predict(merged[feature_cols])
merged['pred_landing_x'] = pred[:, 0]
merged['pred_landing_y'] = pred[:, 1]

merged['actual_distance'] = np.sqrt(
    (merged['landing_x'] - merged['start_x']) ** 2 + (merged['landing_y'] - merged['start_y']) ** 2
)
merged['predicted_distance'] = np.sqrt(
    (merged['pred_landing_x'] - merged['start_x']) ** 2 + (merged['pred_landing_y'] - merged['start_y']) ** 2
)
merged['gap'] = merged['actual_distance'] - merged['predicted_distance']

merged = merged.sort_values('gap').reset_index(drop=True)

print('\n=== Overall ===')
print(f"Mean actual distance:    {merged['actual_distance'].mean():.2f} pitch units")
print(f"Mean predicted distance: {merged['predicted_distance'].mean():.2f} pitch units")
print(f"Mean gap (actual-predicted): {merged['gap'].mean():+.2f}  (negative = shorter than expected)")
print(f"Crosses with gap < 0 (shorter than expected): {(merged['gap'] < 0).mean():.1%}")

# ----------------------------------------------------------------------------
# 1. Individual crosses: the most "optimal" (shortest-relative-to-expected)
# ----------------------------------------------------------------------------
cols_out = [
    'match', 'minute', 'crossing_team', 'player', 'x', 'y', 'end_x', 'end_y', 'body_part', 'wide_channel',
    'clearing_team', 'player_name', 'start_x', 'start_y', 'landing_x', 'landing_y',
    'pred_landing_x', 'pred_landing_y', 'actual_distance', 'predicted_distance', 'gap',
]
top_crosses = merged[cols_out].rename(columns={
    'player': 'crosser', 'x': 'cross_x', 'y': 'cross_y',
    'end_x': 'cross_end_x', 'end_y': 'cross_end_y',
    'player_name': 'clearing_player', 'start_x': 'clr_start_x', 'start_y': 'clr_start_y',
})
top_crosses.to_csv(OUT_DIR / 'optimal_crosses_ranked.csv', index=False)
print(f"\nSaved full ranked list ({len(top_crosses)} crosses) to optimal_crosses_ranked.csv")
print('\nTop 15 most "optimal" crosses (shortest actual vs. expected clearance):')
print(top_crosses.head(15)[['match', 'minute', 'crossing_team', 'crosser', 'clearing_team',
                             'actual_distance', 'predicted_distance', 'gap']].round(2).to_string(index=False))

# ----------------------------------------------------------------------------
# 2. By crossing team (min 10 cross-originated headed clearances)
# ----------------------------------------------------------------------------
by_team = (merged.groupby('crossing_team')
           .agg(n=('gap', 'size'), mean_gap=('gap', 'mean'),
                mean_actual=('actual_distance', 'mean'), mean_predicted=('predicted_distance', 'mean'))
           .query('n >= 10')
           .sort_values('mean_gap'))
by_team.to_csv(OUT_DIR / 'optimal_crosses_by_team.csv')
print('\n=== By crossing team (n >= 10), most "optimal" first ===')
print(by_team.to_string())

# ----------------------------------------------------------------------------
# 3. By crosser (min 8 cross-originated headed clearances)
# ----------------------------------------------------------------------------
by_player = (merged.groupby(['player', 'crossing_team'])
             .agg(n=('gap', 'size'), mean_gap=('gap', 'mean'),
                  mean_actual=('actual_distance', 'mean'), mean_predicted=('predicted_distance', 'mean'))
             .query('n >= 8')
             .sort_values('mean_gap'))
by_player.to_csv(OUT_DIR / 'optimal_crosses_by_player.csv')
print('\n=== By crosser (n >= 8), most "optimal" first ===')
print(by_player.to_string())

# ----------------------------------------------------------------------------
# 4. By cross origin zone (wide_channel x thirds of the cross start x)
# ----------------------------------------------------------------------------
def x_third(x):
    if x < 66.667: return 'own_half_or_middle_third'
    if x < 83.333: return 'attacking_third_far'
    return 'attacking_third_close_to_byline'


merged['cross_zone'] = merged['wide_channel'] + '_' + merged['x'].apply(x_third)
by_zone = (merged.groupby('cross_zone')
           .agg(n=('gap', 'size'), mean_gap=('gap', 'mean'),
                mean_actual=('actual_distance', 'mean'), mean_predicted=('predicted_distance', 'mean'))
           .sort_values('mean_gap'))
by_zone.to_csv(OUT_DIR / 'optimal_crosses_by_zone.csv')
print('\n=== By cross origin zone, most "optimal" first ===')
print(by_zone.to_string())

print(f'\nAll outputs saved to {OUT_DIR}')
