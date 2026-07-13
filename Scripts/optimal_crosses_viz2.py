"""Second round of optimal-crosses visuals:
  1. Arrow map colored by gap (over/under expected clearance distance)
  2. Average landing zone: expected vs. actual, across all 757 cross-originated
     headed clearances
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from mplsoccer import Pitch

REPO = Path('/home/user/ecuador-2026-analytics')
OUT_DIR = REPO / 'Cross Models' / 'improved' / 'optimal_crosses'

df = pd.read_csv(OUT_DIR / 'optimal_crosses_ranked.csv')

BG, LINE = '#0d1117', '#2c3a4d'
TXT_MUTE = '#9aa4b2'
CMAP = 'RdYlGn_r'  # red = further than expected, green = shorter than expected

# ============================================================================
# 1. Arrow map colored by gap -- extremes from both ends (15 shortest-relative
#    + 15 longest-relative) to show the full over/under range in one view
# ============================================================================
n_each = 15
extremes = pd.concat([
    df.sort_values('gap').head(n_each),
    df.sort_values('gap', ascending=False).head(n_each),
]).copy()

for col_x, col_y in [('clr_start_x', 'clr_start_y'), ('landing_x', 'landing_y'),
                     ('pred_landing_x', 'pred_landing_y')]:
    extremes[col_x] = 100 - extremes[col_x]
    extremes[col_y] = 100 - extremes[col_y]

norm = TwoSlopeNorm(vmin=extremes['gap'].min(), vcenter=0, vmax=extremes['gap'].max())
cmap = plt.get_cmap(CMAP)

pitch = Pitch(pitch_type='opta', pitch_color=BG, line_color=LINE, linewidth=1.2)
fig, ax = pitch.draw(figsize=(11, 7.5))
fig.set_facecolor(BG)

# cross itself: thin neutral context arrow
pitch.arrows(extremes['cross_x'], extremes['cross_y'], extremes['cross_end_x'], extremes['cross_end_y'],
             ax=ax, color='#4b5563', width=1, headwidth=4, headlength=4, alpha=0.55, zorder=2)

# actual clearance arrow, colored by gap (over/under expected)
for _, r in extremes.iterrows():
    color = cmap(norm(r['gap']))
    pitch.arrows(r['clr_start_x'], r['clr_start_y'], r['landing_x'], r['landing_y'],
                 ax=ax, color=color, width=2.2, headwidth=6, headlength=6, alpha=0.95, zorder=3)
    # small marker at the model's expected landing point for reference
    ax.scatter(r['pred_landing_x'], r['pred_landing_y'], marker='x', s=45, color='white',
               linewidths=1.4, zorder=4)

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Gap = actual − expected clearance distance (pitch units)', color='white', fontsize=9.5)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')
cbar.ax.axhline(0, color='white', lw=1.2, ls=':')

legend_elems = [
    plt.Line2D([0], [0], color='#4b5563', lw=2, marker='>', markersize=7, label='Cross'),
    plt.Line2D([0], [0], color='white', lw=0, marker='x', markersize=8, label='Model-expected landing point'),
]
ax.legend(handles=legend_elems, loc='upper left', fontsize=9, facecolor=BG, labelcolor='white',
          framealpha=0.7)
ax.set_title('Cross-originated clearances: actual arrow colored by how far over/under expected it landed',
             color='white', fontsize=12.5, pad=12)
fig.text(0.5, 0.02,
         f'Ecuador 2026 · 15 most shorter-than-expected + 15 most further-than-expected of 757 '
         f'cross-originated headed clearances · red = landed further than expected, green = shorter '
         f'· white x = model-expected landing point',
         ha='center', color=TXT_MUTE, fontsize=8.5)

fig.savefig(OUT_DIR / 'arrows_colored_by_gap.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('Saved arrows_colored_by_gap.png')

# ============================================================================
# 2. Average landing zone: expected vs. actual, across all 757 clearances
#    (native clearance frame -- start, actual landing and expected landing are
#    all from the same clearing action, no mirroring needed here)
# ============================================================================
avg_start_x, avg_start_y = df['clr_start_x'].mean(), df['clr_start_y'].mean()
avg_actual_x, avg_actual_y = df['landing_x'].mean(), df['landing_y'].mean()
avg_pred_x, avg_pred_y = df['pred_landing_x'].mean(), df['pred_landing_y'].mean()
# mean of each clearance's own scalar distance (what Table 7 / the ranking uses)
avg_actual_dist = df['actual_distance'].mean()
avg_pred_dist = df['predicted_distance'].mean()
# distance implied by the two average *points* -- not the same thing, since
# clearance direction varies a lot: individual distances are large but point
# all over the pitch, so they partly cancel out when the coordinates are
# averaged. Both numbers are reported so the figure doesn't conflate them.
avg_point_actual_dist = float(np.hypot(avg_actual_x - avg_start_x, avg_actual_y - avg_start_y))
avg_point_pred_dist = float(np.hypot(avg_pred_x - avg_start_x, avg_pred_y - avg_start_y))

pitch2 = Pitch(pitch_type='opta', pitch_color=BG, line_color=LINE, linewidth=1.2)
fig2, ax2 = pitch2.draw(figsize=(10, 7.6))
fig2.set_facecolor(BG)
fig2.subplots_adjust(bottom=0.16, top=0.93)

# faint scatter of every individual actual / expected landing point for context
ax2.scatter(df['landing_x'], df['landing_y'], s=10, color='#f06fa3', alpha=0.12, zorder=1)
ax2.scatter(df['pred_landing_x'], df['pred_landing_y'], s=10, color='#9aa4b2', alpha=0.12, zorder=1)

pitch2.arrows(avg_start_x, avg_start_y, avg_pred_x, avg_pred_y, ax=ax2, color='#9aa4b2',
              width=2.5, headwidth=8, headlength=8, alpha=0.95, zorder=4, linestyle='--')
pitch2.arrows(avg_start_x, avg_start_y, avg_actual_x, avg_actual_y, ax=ax2, color='#f06fa3',
              width=2.8, headwidth=9, headlength=9, alpha=0.98, zorder=5)

ax2.scatter([avg_start_x], [avg_start_y], s=180, color='#4ade80', edgecolor='white', linewidths=1.2,
            zorder=6, label='Avg. clearance start')
ax2.scatter([avg_pred_x], [avg_pred_y], s=260, marker='*', color='#9aa4b2', edgecolor='white',
            linewidths=1.2, zorder=6, label='Avg. expected landing zone')
ax2.scatter([avg_actual_x], [avg_actual_y], s=260, marker='*', color='#f06fa3', edgecolor='white',
            linewidths=1.2, zorder=6, label='Avg. actual landing zone')

ax2.legend(loc='upper left', fontsize=9.5, facecolor=BG, labelcolor='white', framealpha=0.75)
ax2.set_title('Average headed-clearance landing zone: expected vs. actual (cross-originated, n=757)',
              color='white', fontsize=12.5, pad=12)
fig2.text(0.5, 0.085,
          f'Ecuador 2026 · faint dots = every individual actual (pink) / expected (gray) landing point · '
          f'the two average landing zones sit only {abs(avg_point_actual_dist - avg_point_pred_dist):.1f} units apart '
          f'({avg_point_pred_dist:.1f} vs. {avg_point_actual_dist:.1f} from the average start) because '
           'clearance direction varies and cancels out spatially --',
          ha='center', color=TXT_MUTE, fontsize=8.5)
fig2.text(0.5, 0.045,
          f'the mean of each clearance\'s OWN distance is a different, larger number: '
          f'expected {avg_pred_dist:.1f} vs. actual {avg_actual_dist:.1f} pitch units '
          f'({avg_actual_dist - avg_pred_dist:+.1f}), which is the figure to use for "how much further than expected."',
          ha='center', color=TXT_MUTE, fontsize=8.5)

fig2.savefig(OUT_DIR / 'average_landing_zone_expected_vs_actual.png', dpi=180, bbox_inches='tight',
             facecolor=BG)
plt.close(fig2)
print('Saved average_landing_zone_expected_vs_actual.png')
print(f"\nAvg clearance start:    ({avg_start_x:.1f}, {avg_start_y:.1f})")
print(f"Avg expected landing:   ({avg_pred_x:.1f}, {avg_pred_y:.1f})  dist={avg_pred_dist:.2f}")
print(f"Avg actual landing:     ({avg_actual_x:.1f}, {avg_actual_y:.1f})  dist={avg_actual_dist:.2f}")
