"""Pitch visuals for the optimal-crosses analysis:
  1. Arrow map of the top 20 "optimal" crosses (cross, actual clearance, expected clearance)
  2. Bin-valued heatmap of mean gap by cross origin location, across all 757 crosses
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

REPO = Path('/home/user/ecuador-2026-analytics')
OUT_DIR = REPO / 'Cross Models' / 'improved' / 'optimal_crosses'

df = pd.read_csv(OUT_DIR / 'optimal_crosses_ranked.csv')

BG, LINE = '#0d1117', '#2c3a4d'
C_CROSS, C_ACTUAL, C_EXPECTED = '#4ade80', '#f06fa3', '#9aa4b2'
TXT_MUTE = '#9aa4b2'

# ============================================================================
# 1. Arrow map -- top 20 "optimal" crosses
# ============================================================================
top = df.sort_values('gap').head(20).copy()

# mirror the clearance-side coordinates into the crossing team's frame for
# display (Opta coordinates are normalised to the *acting team's own*
# attacking direction; distances are invariant to this reflection, only the
# plot needs it)
for col_x, col_y in [('clr_start_x', 'clr_start_y'), ('landing_x', 'landing_y'),
                     ('pred_landing_x', 'pred_landing_y')]:
    top[col_x] = 100 - top[col_x]
    top[col_y] = 100 - top[col_y]

pitch = Pitch(pitch_type='opta', pitch_color=BG, line_color=LINE, linewidth=1.2)
fig, ax = pitch.draw(figsize=(10, 7))
fig.set_facecolor(BG)

pitch.arrows(top['cross_x'], top['cross_y'], top['cross_end_x'], top['cross_end_y'],
             ax=ax, color=C_CROSS, width=1.8, headwidth=6, headlength=6, alpha=0.9)
pitch.arrows(top['clr_start_x'], top['clr_start_y'], top['landing_x'], top['landing_y'],
             ax=ax, color=C_ACTUAL, width=2.0, headwidth=7, headlength=7, alpha=0.95)
pitch.arrows(top['clr_start_x'], top['clr_start_y'], top['pred_landing_x'], top['pred_landing_y'],
             ax=ax, color=C_EXPECTED, width=1.2, headwidth=5, headlength=5, alpha=0.55,
             linestyle='--')

legend_elems = [
    plt.Line2D([0], [0], color=C_CROSS, lw=2, marker='>', markersize=7, label='Cross'),
    plt.Line2D([0], [0], color=C_ACTUAL, lw=2, marker='>', markersize=7, label='Actual clearance'),
    plt.Line2D([0], [0], color=C_EXPECTED, lw=2, ls='--', marker='>', markersize=7,
               label='Model-expected clearance'),
]
ax.legend(handles=legend_elems, loc='upper left', fontsize=9, facecolor=BG, labelcolor='white',
          framealpha=0.7)
ax.set_title('Top 20 "optimal" crosses: headed clearance falls shortest relative to expectation',
             color='white', fontsize=13, pad=12)
fig.text(0.5, 0.02,
         'Ecuador 2026 · 757 open-play crosses defended by a headed clearance · '
         'gap = actual clearance distance − model-expected distance, most negative first',
         ha='center', color=TXT_MUTE, fontsize=8.5)
fig.savefig(OUT_DIR / 'top20_optimal_crosses_arrows.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('Saved top20_optimal_crosses_arrows.png')

# ============================================================================
# 2. Bin-valued heatmap -- mean gap by cross origin location, all 757 crosses
# ============================================================================
pitch2 = Pitch(pitch_type='opta', pitch_color=BG, line_color=LINE, linewidth=1.2, line_zorder=2)
fig2, ax2 = pitch2.draw(figsize=(10, 7))
fig2.set_facecolor(BG)

bins = (6, 5)
stats = pitch2.bin_statistic(df['cross_x'], df['cross_y'], values=df['gap'],
                              statistic='mean', bins=bins)
counts = pitch2.bin_statistic(df['cross_x'], df['cross_y'], statistic='count', bins=bins)
stats['statistic'] = np.where(counts['statistic'] >= 5, stats['statistic'], np.nan)

pop_mean = df['gap'].mean()
vmax = np.nanmax(np.abs(stats['statistic'] - pop_mean))
hm = pitch2.heatmap(stats, ax=ax2, cmap='RdYlGn_r', vmin=pop_mean - vmax, vmax=pop_mean + vmax,
                    edgecolor=BG, linewidth=1)

labels = pitch2.label_heatmap(stats, ax=ax2, str_format='{:.1f}', color='black', fontsize=11,
                               exclude_nan=True, ha='center', va='center', weight='bold')

cbar = fig2.colorbar(hm, ax=ax2, shrink=0.8, pad=0.02)
cbar.set_label('Mean gap = actual − expected clearance distance (pitch units)', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')
cbar.ax.axhline(pop_mean, color='white', lw=1.2, ls=':')

ax2.set_title('Where crosses produce shorter-than-expected clearances (bin ≥5 crosses; blank = too few)',
              color='white', fontsize=12.5, pad=12)
fig2.text(0.5, 0.02,
          f'Ecuador 2026 · 757 open-play crosses defended by a headed clearance, binned by cross origin (6x5 grid) · '
          f'greener = clearance travels shorter than expected · population mean gap = +{pop_mean:.1f} units '
          '(dotted line on colorbar)',
          ha='center', color=TXT_MUTE, fontsize=8.5)

fig2.savefig(OUT_DIR / 'gap_heatmap_by_cross_origin.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print('Saved gap_heatmap_by_cross_origin.png')
