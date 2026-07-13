"""Regenerate Figures 3, 5, 6 for the v2 paper, showing before/after the fixes.

Matches the style of the original figures embedded in
Cross Models/ecuador_2026_cross_models_paper.docx: white background, serif
type, Okabe-Ito colorblind-safe palette, bold serif titles, small gray
footer notes with the methodology note.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix, r2_score

REPO = Path('/home/user/ecuador-2026-analytics')
OUT_DIR = REPO / 'Cross Models' / 'improved'
VIS_DIR = OUT_DIR / 'visuals'
VIS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

C_BLUE, C_AMBER, C_GREEN, C_VERMILLION = '#0072B2', '#E69F00', '#009E73', '#D55E00'
C_GRAY = '#6b7684'

METHOD_NOTE = ('5-fold GroupKFold cross-validation, grouped by match to prevent leakage. '
               'n = 3,580 open-play crosses, 136 Ecuador 2026 matches.')

df = pd.read_parquet(OUT_DIR / 'cross_events_v2.parquet')
metrics = json.load(open(OUT_DIR / 'metrics_v2.json'))

# ============================================================================
# Figure 3v2 -- calibration before/after isotonic fix
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
fig.suptitle('Figure 3 (v2). Calibration before and after isotonic recalibration',
             fontsize=17, fontweight='bold', y=1.02)

for ax, name, label, color in [
    (axes[0], 'cross_completion', 'Completion', C_BLUE),
    (axes[1], 'cross_chance_creation', 'Chance creation', C_AMBER),
]:
    y = df[{'cross_completion': 'outcome', 'cross_chance_creation': 'shot_created'}[name]].values.astype(int)
    p_raw = np.load(OUT_DIR / f'oof_raw_{name}.npy')
    p_cal = np.load(OUT_DIR / f'oof_cal_{name}.npy')

    frac_raw, mean_raw = calibration_curve(y, p_raw, n_bins=10, strategy='quantile')
    frac_cal, mean_cal = calibration_curve(y, p_cal, n_bins=10, strategy='quantile')

    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1.3, label='Perfectly calibrated')
    ax.plot(mean_raw, frac_raw, 'o--', color=color, alpha=0.55, lw=1.8, ms=6, label='Before (raw)')
    ax.plot(mean_cal, frac_cal, 'o-', color=color, lw=2.2, ms=6, label='After (isotonic)')
    b_raw = metrics['fix1_calibration'][name]['before']['brier']
    b_cal = metrics['fix1_calibration'][name]['after']['brier']
    e_raw = metrics['fix1_calibration'][name]['before']['ece']
    e_cal = metrics['fix1_calibration'][name]['after']['ece']
    ax.set_title(f'{label}\nBrier {b_raw:.3f} $\\to$ {b_cal:.3f}   |   ECE {e_raw:.3f} $\\to$ {e_cal:.3f}',
                 fontsize=13)
    ax.set_xlabel('Mean predicted probability (decile bin)')
    ax.set_ylabel('Observed frequency')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize=10, frameon=True)
    ax.grid(alpha=0.4)

fig.text(0.02, -0.03,
         METHOD_NOTE + ' Isotonic calibrator fit on a match-grouped 25% inner split of each '
         'training fold, never on rows used to fit the base classifier. AUC is materially '
         'unchanged (ranking preserved); only probability calibration improves.',
         fontsize=9, color=C_GRAY, wrap=True)
fig.tight_layout()
fig.savefig(VIS_DIR / 'fig03_v2_calibration.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved fig03_v2_calibration.png')

# ============================================================================
# Figure 5v2 -- confusion matrix, flat vs hierarchical
# ============================================================================
CLASSES_4 = ['incomplete', 'complete_no_shot', 'shot_no_goal', 'goal']
flat_classes = metrics['baseline']['cross_outcome_multiclass']['classes']
flat_cm = np.array(metrics['baseline']['cross_outcome_multiclass']['confusion_matrix'], dtype=float)
flat_cm_norm = flat_cm / flat_cm.sum(axis=1, keepdims=True)
flat_order = [flat_classes.index(c) for c in CLASSES_4]
flat_cm_norm = flat_cm_norm[np.ix_(flat_order, flat_order)]

hier_cm = np.array(metrics['fix3_hierarchical']['hierarchical']['confusion_matrix_recall'], dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5))
fig.suptitle('Figure 5 (v2). Confusion matrix: flat vs. hierarchical outcome classifier',
             fontsize=17, fontweight='bold', y=1.03)
fig.subplots_adjust(wspace=0.55)

for ax, cm, title, acc, f1 in [
    (axes[0], flat_cm_norm, 'Flat 4-class (baseline)',
     metrics['baseline']['cross_outcome_multiclass']['cv']['accuracy'],
     metrics['baseline']['cross_outcome_multiclass']['cv']['macro_f1']),
    (axes[1], hier_cm, 'Hierarchical 2-stage (Fix 3)',
     metrics['fix3_hierarchical']['hierarchical']['accuracy'],
     metrics['fix3_hierarchical']['hierarchical']['macro_f1']),
]:
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES_4, rotation=30, ha='right')
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES_4)
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
    ax.set_title(f'{title}\nAccuracy={acc:.3f}  Macro-F1={f1:.3f}', fontsize=13)
    for i in range(4):
        for j in range(4):
            v = cm[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                     color='white' if v > 0.5 else 'black', fontsize=11)
    ax.grid(False)

fig.colorbar(im, ax=axes, label='Row-normalised frequency (recall)', shrink=0.85, pad=0.03)
fig.text(0.02, -0.08,
         METHOD_NOTE + ' The hierarchical model (stage A: complete vs incomplete; stage B: '
         '3-class conditional on complete) trades overall accuracy for materially better '
         'recall on the shot_no_goal class; the goal class remains effectively unrecoverable '
         'at n=37 positive events regardless of architecture.',
         fontsize=9, color=C_GRAY, wrap=True)
fig.savefig(VIS_DIR / 'fig05_v2_confusion.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved fig05_v2_confusion.png')

# ============================================================================
# Figure 6v2 -- residual diagnostics, baseline regressor vs calibrated hurdle
# ============================================================================
y_val = df['danger_value'].values.astype(float)
oof_baseline = np.load(OUT_DIR / 'oof_baseline_delivery_value.npy')
oof_hurdle = np.load(OUT_DIR / 'oof_hurdle_calibrated_delivery_value.npy')

fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
fig.suptitle('Figure 6 (v2). Delivery-value regression: baseline vs. calibrated hurdle model',
             fontsize=16.5, fontweight='bold', y=1.03)

for ax, pred, title, r2 in [
    (axes[0], oof_baseline, 'Baseline (single XGBRegressor)',
     metrics['fix2_hurdle']['baseline']['r2']),
    (axes[1], oof_hurdle, 'Hurdle: calibrated P(shot) x value|shot',
     metrics['fix2_hurdle']['hurdle_calibrated_gate']['r2']),
]:
    resid = y_val - pred
    ax.hist(resid, bins=60, color=C_BLUE if title.startswith('Baseline') else C_GREEN,
            edgecolor='white', linewidth=0.4)
    ax.axvline(0, color=C_VERMILLION, ls=':', lw=1.6)
    ax.set_title(f'{title}\nR$^2$={r2:.3f}', fontsize=13)
    ax.set_xlabel('Residual (actual - predicted)')
    ax.set_ylabel('Count')
    ax.grid(alpha=0.4)

fig.text(0.02, -0.03,
         METHOD_NOTE + ' The hurdle model uses the isotonic-calibrated cross_chance_creation '
         'probability (Fig. 3) as its gate; using the raw (uncalibrated) gate instead produces '
         'R^2=-0.36, worse than the baseline, because it inherits the scale_pos_weight '
         'over-confidence documented in Section 4.2.',
         fontsize=9, color=C_GRAY, wrap=True)
fig.tight_layout()
fig.savefig(VIS_DIR / 'fig06_v2_residuals.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved fig06_v2_residuals.png')
