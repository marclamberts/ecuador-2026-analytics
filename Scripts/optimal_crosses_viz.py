"""Pitch map of the most 'optimal' crosses: shortest actual-vs-expected headed clearance."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch

REPO = Path('/home/user/ecuador-2026-analytics')
OUT_DIR = REPO / 'Cross Models' / 'improved' / 'optimal_crosses'

df = pd.read_csv(OUT_DIR / 'optimal_crosses_ranked.csv')
top = df.sort_values('gap').head(20).copy()

# Opta coordinates are normalised to the *acting team's own* attacking
# direction, so the cross (crossing team) and the clearance (defending team)
# live in mirrored frames for the same physical pitch location. Mirror the
# clearance-side coordinates into the crossing team's frame for display only
# -- distances themselves are invariant to this reflection.
for col_x, col_y in [('clr_start_x', 'clr_start_y'), ('landing_x', 'landing_y'),
                     ('pred_landing_x', 'pred_landing_y')]:
    top[col_x] = 100 - top[col_x]
    top[col_y] = 100 - top[col_y]

BG, LINE = '#0d1117', '#2c3a4d'
C_CROSS, C_ACTUAL, C_EXPECTED = '#4ade80', '#f06fa3', '#9aa4b2'

pitch = Pitch(pitch_type='opta', pitch_color=BG, line_color=LINE, linewidth=1.2)
fig, ax = pitch.draw(figsize=(10, 7))
fig.set_facecolor(BG)

for _, r in top.iterrows():
    pitch.lines(r['cross_x'], r['cross_y'], r['cross_end_x'], r['cross_end_y'],
                ax=ax, color=C_CROSS, lw=1.6, alpha=0.85,
                comet=True, transparent=True)
    pitch.lines(r['clr_start_x'], r['clr_start_y'], r['landing_x'], r['landing_y'],
                ax=ax, color=C_ACTUAL, lw=1.8, alpha=0.9)
    pitch.lines(r['clr_start_x'], r['clr_start_y'], r['pred_landing_x'], r['pred_landing_y'],
                ax=ax, color=C_EXPECTED, lw=1.2, ls='--', alpha=0.6)

legend_elems = [
    plt.Line2D([0], [0], color=C_CROSS, lw=2, label='Cross (start -> end)'),
    plt.Line2D([0], [0], color=C_ACTUAL, lw=2, label='Actual clearance (start -> landing)'),
    plt.Line2D([0], [0], color=C_EXPECTED, lw=2, ls='--', label='Model-expected clearance'),
]
ax.legend(handles=legend_elems, loc='upper left', fontsize=9, facecolor=BG, labelcolor='white',
          framealpha=0.7)
ax.set_title('Top 20 "optimal" crosses: headed clearance falls shortest relative to the model\'s expectation',
             color='white', fontsize=13, pad=12)
fig.text(0.5, 0.02,
         'Ecuador 2026 · 757 open-play crosses defended by a headed clearance · '
         'gap = actual clearance distance − model-expected distance, most negative first',
         ha='center', color='#9aa4b2', fontsize=8.5)

fig.savefig(OUT_DIR / 'top20_optimal_crosses_pitch_map.png', dpi=180, bbox_inches='tight', facecolor=BG)
print('Saved top20_optimal_crosses_pitch_map.png')
