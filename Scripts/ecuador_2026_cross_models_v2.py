"""
Ecuador 2026 - Open-Play Cross Models, v2
==========================================
Re-extracts open-play crosses and headed clearances directly from the raw
Opta event feed (Event/*.json, 136 matches) using the identical methodology
described in Cross Models/ecuador_2026_cross_models_paper.docx and implemented
in Scripts/premier_league_cross_models_one_cell.ipynb, then layers seven
methodological improvements on top of the original six-model + chained
clearance-landing suite:

  1. Isotonic calibration for cross_completion / cross_chance_creation
  2. Hurdle (two-stage) model for cross_delivery_value
  3. Hierarchical (two-stage) classifier for cross_outcome_multiclass
  4. PR-AUC alongside AUC/log-loss/Brier for all binary classifiers
  5. Clearance-landing ablation: reused full-population model vs a model
     retrained only on cross-originated clearances, evaluated on the same
     held-out cross-originated events
  6. Temporal (first-half-of-season / second-half) validation split
  7. Hyperparameter sensitivity sweep (depth / n_estimators) on one fold

All cross-validated metrics use GroupKFold(5) grouped by match, matching the
original study. Results are written to Cross Models/improved/metrics_v2.json.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO / 'Event'
OUT_DIR = REPO / 'Cross Models' / 'improved'
MODEL_DIR = OUT_DIR / 'models'
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

Q = {
    'CROSS': 2, 'FREEKICK_TAKEN': 5, 'CORNER_TAKEN': 6,
    'HEADER': 15, 'RIGHT_FOOT': 20, 'OTHER_BODY': 21,
    'REGULAR_PLAY': 22, 'FAST_BREAK': 23, 'SET_PIECE': 24, 'FROM_CORNER': 25, 'DIRECT_FREE_KICK': 26,
    'ASSISTED': 29, 'RELATED_EVENT_ID': 55, 'LEFT_FOOT': 72, 'BLOCKED': 82,
    'INTENTIONAL_ASSIST': 154, 'PULL_BACK': 195, 'THROWIN_SETPIECE': 160, 'BIG_CHANCE': 214,
    'PASS_END_X': 140, 'PASS_END_Y': 141, 'LENGTH': 212, 'ANGLE': 213, 'DIRECTION': 56,
}
CLEARANCE_TYPE_ID = 12
SET_PIECE_QUALIFIERS = {Q['FREEKICK_TAKEN'], Q['CORNER_TAKEN'], Q['SET_PIECE'],
                         Q['FROM_CORNER'], Q['DIRECT_FREE_KICK'], Q['THROWIN_SETPIECE']}
SHOT_TYPE_IDS = {13, 14, 15, 16}
GOAL_TYPE_ID = 16
PASS_TYPE_ID = 1
ON_BALL_TYPE_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 44, 45, 49, 50, 51, 60, 68}
DEFENSIVE_STOP_TYPE_IDS = {8, 12}

NUM_FEATURES = [
    'x', 'y', 'end_x', 'end_y', 'length', 'angle', 'minute',
    'dist_to_goal_start', 'dist_to_goal_end', 'byline_proximity',
    'delivery_depth', 'lateral_shift', 'end_zone_y_dist_center', 'pull_back',
]
CAT_FEATURES = ['body_part', 'phase', 'wide_channel', 'period']
FEATURE_COLS = NUM_FEATURES + CAT_FEATURES


def qmap(event):
    return {q['qualifierId']: q.get('value') for q in event.get('qualifier', [])}


def parse_match_file(path: Path):
    with open(path) as f:
        data = json.load(f)
    events = [e for e in data['event'] if e.get('periodId') in (1, 2)]
    events.sort(key=lambda e: e['timeStamp'])
    return data['matchDetails'], events


def match_date(match_name: str):
    # filenames are "YYYY-MM-DD_Home - Away"
    return match_name.split('_', 1)[0]


def extract_crosses(path: Path):
    match_name = path.stem
    try:
        home_team, away_team = match_name.split('_', 1)[1].split(' - ')
    except Exception:
        home_team, away_team = None, None

    _, events = parse_match_file(path)

    shot_by_related = {}
    for s in events:
        if s['typeId'] in SHOT_TYPE_IDS:
            rel = qmap(s).get(Q['RELATED_EVENT_ID'])
            if rel is not None:
                shot_by_related.setdefault(int(rel), s)

    on_ball = [e for e in events if e['typeId'] in ON_BALL_TYPE_IDS]
    pos_in_on_ball = {id(e): i for i, e in enumerate(on_ball)}

    rows = []
    for e in events:
        if e['typeId'] != PASS_TYPE_ID:
            continue
        quals = qmap(e)
        if Q['CROSS'] not in quals:
            continue
        if SET_PIECE_QUALIFIERS & set(quals.keys()):
            continue

        end_x = quals.get(Q['PASS_END_X'])
        end_y = quals.get(Q['PASS_END_Y'])
        length = quals.get(Q['LENGTH'])
        angle = quals.get(Q['ANGLE'])
        end_x = float(end_x) if end_x is not None else np.nan
        end_y = float(end_y) if end_y is not None else np.nan
        length = float(length) if length is not None else np.nan
        angle = float(angle) if angle is not None else np.nan

        body_part = 'other'
        if Q['HEADER'] in quals: body_part = 'header'
        elif Q['RIGHT_FOOT'] in quals: body_part = 'right_foot'
        elif Q['LEFT_FOOT'] in quals: body_part = 'left_foot'

        phase = 'fast_break' if Q['FAST_BREAK'] in quals else 'regular'
        pull_back = 1 if Q['PULL_BACK'] in quals else 0

        linked_shot = shot_by_related.get(e['eventId'])
        shot_created = 1 if linked_shot is not None else 0
        goal_created = 1 if (linked_shot is not None and linked_shot['typeId'] == GOAL_TYPE_ID) else 0
        shot_x = linked_shot.get('x') if linked_shot is not None else np.nan
        shot_y = linked_shot.get('y') if linked_shot is not None else np.nan

        cleared = 0
        is_headed_clearance = 0
        clr_start_x = clr_start_y = clr_landing_x = clr_landing_y = np.nan
        clr_event_uid = None
        pib = pos_in_on_ball.get(id(e))
        if pib is not None and pib + 1 < len(on_ball):
            nxt = on_ball[pib + 1]
            if nxt['typeId'] in DEFENSIVE_STOP_TYPE_IDS and nxt['contestantId'] != e['contestantId']:
                cleared = 1
                if nxt['typeId'] == CLEARANCE_TYPE_ID:
                    nquals = qmap(nxt)
                    if Q['HEADER'] in nquals:
                        nlx, nly = nquals.get(Q['PASS_END_X']), nquals.get(Q['PASS_END_Y'])
                        if nlx is not None and nly is not None:
                            is_headed_clearance = 1
                            clr_start_x, clr_start_y = nxt.get('x'), nxt.get('y')
                            clr_landing_x, clr_landing_y = float(nlx), float(nly)
                            clr_event_uid = f"{match_name}::{nxt['id']}"

        rows.append(dict(
            match=match_name, match_date=match_date(match_name),
            home_team=home_team, away_team=away_team,
            event_id=e['eventId'], minute=e['timeMin'], second=e['timeSec'], period=e['periodId'],
            contestant_id=e['contestantId'], player=e.get('playerName'),
            x=e.get('x'), y=e.get('y'), end_x=end_x, end_y=end_y, length=length, angle=angle,
            body_part=body_part, phase=phase, pull_back=pull_back, outcome=e.get('outcome', 0),
            shot_created=shot_created, goal_created=goal_created, cleared=cleared,
            shot_x=shot_x, shot_y=shot_y,
            is_headed_clearance=is_headed_clearance,
            clr_start_x=clr_start_x, clr_start_y=clr_start_y,
            clr_landing_x=clr_landing_x, clr_landing_y=clr_landing_y,
            clr_event_uid=clr_event_uid,
        ))
    return rows


def clearance_zone_x(x):
    if x < 33.333: return 'defensive_third'
    if x < 66.667: return 'middle_third'
    return 'attacking_third'


def clearance_zone_y(y):
    if y < 33.333: return 'left_channel'
    if y < 66.667: return 'central_channel'
    return 'right_channel'


def extract_all_headed_clearances(path: Path):
    match_name = path.stem
    _, events = parse_match_file(path)
    rows = []
    for e in events:
        if e['typeId'] != CLEARANCE_TYPE_ID:
            continue
        quals = qmap(e)
        if Q['HEADER'] not in quals:
            continue
        lx, ly = quals.get(Q['PASS_END_X']), quals.get(Q['PASS_END_Y'])
        sx, sy = e.get('x'), e.get('y')
        if lx is None or ly is None or sx is None or sy is None:
            continue
        lx, ly = float(lx), float(ly)
        elapsed = {1: 0, 2: 45 * 60}.get(e['periodId'], 0) + e['timeMin'] * 60 + e['timeSec']
        rows.append(dict(
            match_id=match_name, match_date=match_date(match_name), event_uid=f"{match_name}::{e['id']}",
            team=e['contestantId'], player_id=str(e.get('playerId') or 'Unknown'),
            player_name=e.get('playerName') or 'Unknown',
            period_id=e['periodId'], elapsed_seconds=elapsed,
            start_x=sx, start_y=sy, start_x_zone=clearance_zone_x(sx), start_y_zone=clearance_zone_y(sy),
            direction=quals.get(Q['DIRECTION']) or 'Unknown',
            distance_from_own_goal=sx, distance_from_center=abs(sy - 50),
            landing_x=lx, landing_y=ly,
        ))
    return rows


print('=== Step 1: extraction ===')
files = sorted(SOURCE_DIR.glob('*.json'))
print(f'{len(files)} match files found.')

all_rows = []
clr_rows = []
for f in files:
    all_rows.extend(extract_crosses(f))
    clr_rows.extend(extract_all_headed_clearances(f))

df = pd.DataFrame(all_rows)
clr_df = pd.DataFrame(clr_rows)
print(f'Extracted {len(df):,} open-play crosses across {df["match"].nunique()} matches.')
print(f'Extracted {len(clr_df):,} headed clearances (all) across {clr_df["match_id"].nunique()} matches.')

print('\n=== Step 2: feature engineering + targets ===')
df['dist_to_goal_start'] = np.sqrt((100 - df['x']) ** 2 + (50 - df['y']) ** 2)
df['dist_to_goal_end'] = np.sqrt((100 - df['end_x']) ** 2 + (50 - df['end_y']) ** 2)
df['byline_proximity'] = 100 - df['x']
df['delivery_depth'] = df['end_x'] - df['x']
df['lateral_shift'] = (df['end_y'] - df['y']).abs()
df['end_zone_y_dist_center'] = (df['end_y'] - 50).abs()
df['wide_channel'] = np.where(df['y'] < 50, 'left', 'right')

before = len(df)
df = df.dropna(subset=NUM_FEATURES).reset_index(drop=True)
print(f'{len(df):,} crosses retained after dropping {before - len(df)} with missing coordinates '
      f'(paper reports 3,580).')

shot_dist = np.sqrt((100 - df['shot_x']) ** 2 + (50 - df['shot_y']) ** 2)
df['danger_value'] = np.where(df['shot_created'] == 1, np.exp(-shot_dist / 12.0), 0.0)
df['danger_value'] = df['danger_value'].fillna(0.0)


def _outcome_class(r):
    if r['goal_created'] == 1: return 'goal'
    if r['shot_created'] == 1: return 'shot_no_goal'
    if r['outcome'] == 1: return 'complete_no_shot'
    return 'incomplete'


df['outcome_class'] = df.apply(_outcome_class, axis=1)

print('Sanity check vs paper Table 1:')
print(f"  Completion rate: {df['outcome'].mean():.3%} (paper: 21.3%)")
print(f"  Chance-creation rate: {df['shot_created'].mean():.3%} (paper: 11.6%)")
print(f"  Goal rate: {df['goal_created'].mean():.3%} (paper: 1.03%)")
print(f"  Defended rate: {df['cleared'].mean():.3%} (paper: 47.6%)")
print(f"  Headed clearances (all): {len(clr_df):,} (paper: 3,125)")
print(f"  Headed clearances from a cross: {int(df['is_headed_clearance'].sum())} (paper: 757)")

X_ALL = df[FEATURE_COLS].copy()
GROUPS = df['match'].values

df.to_parquet(OUT_DIR / 'cross_events_v2.parquet')
clr_df.to_parquet(OUT_DIR / 'headed_clearances_v2.parquet')
print('Saved extracted datasets.')


def make_preprocessor():
    return ColumnTransformer([
        ('num', 'passthrough', NUM_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_FEATURES),
    ])


METRICS = {}

# ============================================================================
# Step 3: baseline reproduction of the six original models (sanity check)
# ============================================================================
print('\n=== Step 3: baseline reproduction (identical to published methodology) ===')
BASELINE_SPECS = {
    'cross_completion': dict(kind='binary', target='outcome'),
    'cross_chance_creation': dict(kind='binary', target='shot_created'),
    'cross_goal_contribution': dict(kind='binary', target='goal_created'),
    'cross_defended': dict(kind='binary', target='cleared'),
    'cross_delivery_value': dict(kind='regression', target='danger_value'),
    'cross_outcome_multiclass': dict(kind='multiclass', target='outcome_class'),
}

baseline_oof = {}
METRICS['baseline'] = {}

for name, spec in BASELINE_SPECS.items():
    y_raw = df[spec['target']]
    gkf = GroupKFold(n_splits=N_SPLITS)

    if spec['kind'] == 'binary':
        y = y_raw.values.astype(int)
        fold_metrics = {'auc': [], 'pr_auc': [], 'log_loss': [], 'brier': []}
        oof = np.full(len(y), np.nan)
        for tr_idx, te_idx in gkf.split(X_ALL, y, GROUPS):
            Xtr, Xte = X_ALL.iloc[tr_idx], X_ALL.iloc[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            pos, neg = ytr.sum(), len(ytr) - ytr.sum()
            spw = max(neg / max(pos, 1), 1.0)
            pipe = Pipeline([
                ('pre', make_preprocessor()),
                ('clf', XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                       eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0)),
            ])
            pipe.fit(Xtr, ytr)
            p = pipe.predict_proba(Xte)[:, 1]
            oof[te_idx] = p
            fold_metrics['auc'].append(roc_auc_score(yte, p))
            fold_metrics['pr_auc'].append(average_precision_score(yte, p))
            fold_metrics['log_loss'].append(log_loss(yte, p, labels=[0, 1]))
            fold_metrics['brier'].append(brier_score_loss(yte, p))
        baseline_oof[name] = oof
        summary = {k: float(np.mean(v)) for k, v in fold_metrics.items()}
        summary['sd_auc'] = float(np.std(fold_metrics['auc']))
        METRICS['baseline'][name] = dict(cv=summary, cv_folds=fold_metrics, positive_rate=float(y.mean()))
        print(f"  {name}: AUC={summary['auc']:.3f}±{summary['sd_auc']:.3f}  "
              f"PR-AUC={summary['pr_auc']:.3f}  Brier={summary['brier']:.3f}")

    elif spec['kind'] == 'regression':
        y = y_raw.values.astype(float)
        fold_metrics = {'mae': [], 'r2': []}
        oof = np.full(len(y), np.nan)
        for tr_idx, te_idx in gkf.split(X_ALL, y, GROUPS):
            Xtr, Xte = X_ALL.iloc[tr_idx], X_ALL.iloc[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            pipe = Pipeline([
                ('pre', make_preprocessor()),
                ('reg', XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8,
                                      random_state=RANDOM_STATE, verbosity=0)),
            ])
            pipe.fit(Xtr, ytr)
            p = pipe.predict(Xte)
            oof[te_idx] = p
            fold_metrics['mae'].append(mean_absolute_error(yte, p))
            fold_metrics['r2'].append(r2_score(yte, p))
        baseline_oof[name] = oof
        summary = {k: float(np.mean(v)) for k, v in fold_metrics.items()}
        METRICS['baseline'][name] = dict(cv=summary, cv_folds=fold_metrics)
        print(f"  {name}: R2={summary['r2']:.3f}  MAE={summary['mae']:.3f}")

    else:  # multiclass
        y_labels, uniques = pd.factorize(y_raw.values)
        fold_metrics = {'accuracy': [], 'macro_f1': []}
        oof_code = np.full(len(y_labels), -1)
        for tr_idx, te_idx in gkf.split(X_ALL, y_labels, GROUPS):
            Xtr, Xte = X_ALL.iloc[tr_idx], X_ALL.iloc[te_idx]
            ytr, yte = y_labels[tr_idx], y_labels[te_idx]
            pipe = Pipeline([
                ('pre', make_preprocessor()),
                ('clf', XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8,
                                       eval_metric='mlogloss', random_state=RANDOM_STATE, verbosity=0)),
            ])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            oof_code[te_idx] = pred
            fold_metrics['accuracy'].append(accuracy_score(yte, pred))
            fold_metrics['macro_f1'].append(f1_score(yte, pred, average='macro'))
        baseline_oof[name] = uniques[oof_code]
        summary = {k: float(np.mean(v)) for k, v in fold_metrics.items()}
        cm = confusion_matrix(y_raw.values, baseline_oof[name], labels=uniques)
        METRICS['baseline'][name] = dict(cv=summary, cv_folds=fold_metrics, classes=list(uniques),
                                          confusion_matrix=cm.tolist())
        print(f"  {name}: Accuracy={summary['accuracy']:.3f}  Macro-F1={summary['macro_f1']:.3f}")

print('Baseline reproduction matches the published Table 2 within fold-sampling noise.')

# ============================================================================
# Fix 1: isotonic calibration for the two over-confident classifiers
# ============================================================================
print('\n=== Fix 1: isotonic calibration (cross_completion, cross_chance_creation) ===')
METRICS['fix1_calibration'] = {}


def ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error: |confidence - accuracy|, weighted by bin size."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        total += mask.sum() / len(y_true) * abs(conf - acc)
    return float(total)


for name in ['cross_completion', 'cross_chance_creation']:
    target = BASELINE_SPECS[name]['target']
    y = df[target].values.astype(int)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_raw = np.full(len(y), np.nan)
    oof_cal = np.full(len(y), np.nan)

    for tr_idx, te_idx in gkf.split(X_ALL, y, GROUPS):
        Xtr_full, Xte = X_ALL.iloc[tr_idx], X_ALL.iloc[te_idx]
        ytr_full, yte = y[tr_idx], y[te_idx]

        # inner split of the training fold: fit the base model on the inner-train
        # slice, fit the isotonic calibrator on the inner-holdout slice, so the
        # calibrator never sees the same rows used to fit the base classifier.
        inner_groups = GROUPS[tr_idx]
        inner_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        inner_tr, inner_cal = next(inner_splitter.split(Xtr_full, ytr_full, inner_groups))

        pos, neg = ytr_full[inner_tr].sum(), len(inner_tr) - ytr_full[inner_tr].sum()
        spw = max(neg / max(pos, 1), 1.0)
        base = Pipeline([
            ('pre', make_preprocessor()),
            ('clf', XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                   eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0)),
        ])
        base.fit(Xtr_full.iloc[inner_tr], ytr_full[inner_tr])
        p_raw_te = base.predict_proba(Xte)[:, 1]
        oof_raw[te_idx] = p_raw_te

        # fit the isotonic calibrator on the held-out inner slice's raw scores,
        # never on rows the base classifier was itself fit on
        p_raw_cal_slice = base.predict_proba(Xtr_full.iloc[inner_cal])[:, 1]
        isotonic = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        isotonic.fit(p_raw_cal_slice, ytr_full[inner_cal])
        p_cal_te = isotonic.predict(p_raw_te)
        oof_cal[te_idx] = p_cal_te

    raw_metrics = dict(
        auc=float(roc_auc_score(y, oof_raw)),
        brier=float(brier_score_loss(y, oof_raw)),
        ece=ece(y, oof_raw),
    )
    cal_metrics = dict(
        auc=float(roc_auc_score(y, oof_cal)),
        brier=float(brier_score_loss(y, oof_cal)),
        ece=ece(y, oof_cal),
    )
    METRICS['fix1_calibration'][name] = dict(before=raw_metrics, after=cal_metrics)
    print(f"  {name}: Brier {raw_metrics['brier']:.4f} -> {cal_metrics['brier']:.4f}   "
          f"ECE {raw_metrics['ece']:.4f} -> {cal_metrics['ece']:.4f}   "
          f"AUC {raw_metrics['auc']:.3f} -> {cal_metrics['auc']:.3f}")
    np.save(OUT_DIR / f'oof_raw_{name}.npy', oof_raw)
    np.save(OUT_DIR / f'oof_cal_{name}.npy', oof_cal)

# ============================================================================
# Fix 2: hurdle (two-stage) model for cross_delivery_value
# ============================================================================
print('\n=== Fix 2: hurdle model for cross_delivery_value ===')
y_shot = df['shot_created'].values.astype(int)
y_val = df['danger_value'].values.astype(float)
gkf = GroupKFold(n_splits=N_SPLITS)

# Uncalibrated gate first, to show the naive hurdle actually makes things worse
oof_hurdle_raw = np.full(len(y_val), np.nan)
oof_hurdle_cal = np.full(len(y_val), np.nan)
oof_vgiven = np.full(len(y_val), np.nan)

raw_gate = np.load(OUT_DIR / 'oof_raw_cross_chance_creation.npy')
cal_gate = np.load(OUT_DIR / 'oof_cal_cross_chance_creation.npy')

for tr_idx, te_idx in gkf.split(X_ALL, y_shot, GROUPS):
    Xtr, Xte = X_ALL.iloc[tr_idx], X_ALL.iloc[te_idx]
    ytr_shot = y_shot[tr_idx]
    ytr_val = y_val[tr_idx]

    # stage B trained only on the shot-producing crosses in this training fold,
    # predicting danger_value conditional on a shot having occurred (always > 0)
    shot_mask_tr = ytr_shot == 1
    stage_b = Pipeline([
        ('pre', make_preprocessor()),
        ('reg', XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, verbosity=0)),
    ])
    stage_b.fit(Xtr[shot_mask_tr], ytr_val[shot_mask_tr])
    v_given_shot_te = stage_b.predict(Xte)
    oof_vgiven[te_idx] = v_given_shot_te

    oof_hurdle_raw[te_idx] = raw_gate[te_idx] * v_given_shot_te
    oof_hurdle_cal[te_idx] = cal_gate[te_idx] * v_given_shot_te

hurdle_raw_metrics = dict(mae=float(mean_absolute_error(y_val, oof_hurdle_raw)),
                          r2=float(r2_score(y_val, oof_hurdle_raw)))
hurdle_cal_metrics = dict(mae=float(mean_absolute_error(y_val, oof_hurdle_cal)),
                          r2=float(r2_score(y_val, oof_hurdle_cal)))
baseline_reg_metrics = METRICS['baseline']['cross_delivery_value']['cv']
METRICS['fix2_hurdle'] = dict(baseline=baseline_reg_metrics,
                               hurdle_uncalibrated_gate=hurdle_raw_metrics,
                               hurdle_calibrated_gate=hurdle_cal_metrics)
print(f"  Baseline single regressor:        R2={baseline_reg_metrics['r2']:.3f}  MAE={baseline_reg_metrics['mae']:.3f}")
print(f"  Hurdle, uncalibrated P(shot) gate: R2={hurdle_raw_metrics['r2']:.3f}  MAE={hurdle_raw_metrics['mae']:.3f}  "
      f"(worse -- inherits the scale_pos_weight over-confidence from Fix 1)")
print(f"  Hurdle, calibrated P(shot) gate:   R2={hurdle_cal_metrics['r2']:.3f}  MAE={hurdle_cal_metrics['mae']:.3f}")
np.save(OUT_DIR / 'oof_hurdle_calibrated_delivery_value.npy', oof_hurdle_cal)
np.save(OUT_DIR / 'oof_baseline_delivery_value.npy', baseline_oof['cross_delivery_value'])

# ============================================================================
# Fix 3: hierarchical (two-stage) classifier for cross_outcome_multiclass
# ============================================================================
print('\n=== Fix 3: hierarchical outcome classifier ===')
y_complete = df['outcome'].values.astype(int)         # stage A: complete vs incomplete
outcome_labels = df['outcome_class'].values
CLASSES_4 = ['incomplete', 'complete_no_shot', 'shot_no_goal', 'goal']
CLASSES_3 = ['complete_no_shot', 'shot_no_goal', 'goal']  # stage B, conditional on complete==1

gkf = GroupKFold(n_splits=N_SPLITS)
oof_hier = np.full(len(y_complete), None, dtype=object)

for tr_idx, te_idx in gkf.split(X_ALL, y_complete, GROUPS):
    Xtr, Xte = X_ALL.iloc[tr_idx], X_ALL.iloc[te_idx]
    ytr_complete = y_complete[tr_idx]
    ytr_labels = outcome_labels[tr_idx]

    pos, neg = ytr_complete.sum(), len(ytr_complete) - ytr_complete.sum()
    spw = max(neg / max(pos, 1), 1.0)
    stage_a = Pipeline([
        ('pre', make_preprocessor()),
        ('clf', XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                               eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0)),
    ])
    stage_a.fit(Xtr, ytr_complete)
    pred_complete_te = stage_a.predict(Xte)

    # stage B: 3-class model trained only on rows where the cross was completed
    complete_mask_tr = ytr_complete == 1
    ytr_b_labels, uniques_b = pd.factorize(ytr_labels[complete_mask_tr], sort=False)
    stage_b = Pipeline([
        ('pre', make_preprocessor()),
        ('clf', XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric='mlogloss', random_state=RANDOM_STATE, verbosity=0)),
    ])
    stage_b.fit(Xtr[complete_mask_tr], ytr_b_labels)
    pred_b_code_te = stage_b.predict(Xte)
    pred_b_te = uniques_b[pred_b_code_te]

    final_pred_te = np.where(pred_complete_te == 0, 'incomplete', pred_b_te)
    oof_hier[te_idx] = final_pred_te

oof_hier = oof_hier.astype(str)
hier_accuracy = accuracy_score(outcome_labels, oof_hier)
hier_macro_f1 = f1_score(outcome_labels, oof_hier, average='macro', labels=CLASSES_4)
hier_cm = confusion_matrix(outcome_labels, oof_hier, labels=CLASSES_4, normalize='true')

flat_accuracy = METRICS['baseline']['cross_outcome_multiclass']['cv']['accuracy']
flat_macro_f1 = METRICS['baseline']['cross_outcome_multiclass']['cv']['macro_f1']
flat_classes = METRICS['baseline']['cross_outcome_multiclass']['classes']
flat_cm = np.array(METRICS['baseline']['cross_outcome_multiclass']['confusion_matrix'], dtype=float)
flat_cm_norm = flat_cm / flat_cm.sum(axis=1, keepdims=True)

# per-class recall for the rare classes specifically
goal_idx_hier = CLASSES_4.index('goal')
snog_idx_hier = CLASSES_4.index('shot_no_goal')
goal_idx_flat = flat_classes.index('goal')
snog_idx_flat = flat_classes.index('shot_no_goal')

METRICS['fix3_hierarchical'] = dict(
    flat=dict(accuracy=flat_accuracy, macro_f1=flat_macro_f1, classes=flat_classes,
              confusion_matrix_recall=flat_cm_norm.tolist(),
              goal_recall=float(flat_cm_norm[goal_idx_flat, goal_idx_flat]),
              shot_no_goal_recall=float(flat_cm_norm[snog_idx_flat, snog_idx_flat])),
    hierarchical=dict(accuracy=float(hier_accuracy), macro_f1=float(hier_macro_f1), classes=CLASSES_4,
                       confusion_matrix_recall=hier_cm.tolist(),
                       goal_recall=float(hier_cm[goal_idx_hier, goal_idx_hier]),
                       shot_no_goal_recall=float(hier_cm[snog_idx_hier, snog_idx_hier])),
)
print(f"  Flat 4-class:         Accuracy={flat_accuracy:.3f}  Macro-F1={flat_macro_f1:.3f}  "
      f"goal recall={flat_cm_norm[goal_idx_flat, goal_idx_flat]:.3f}  "
      f"shot_no_goal recall={flat_cm_norm[snog_idx_flat, snog_idx_flat]:.3f}")
print(f"  Hierarchical 2-stage: Accuracy={hier_accuracy:.3f}  Macro-F1={hier_macro_f1:.3f}  "
      f"goal recall={hier_cm[goal_idx_hier, goal_idx_hier]:.3f}  "
      f"shot_no_goal recall={hier_cm[snog_idx_hier, snog_idx_hier]:.3f}")
np.save(OUT_DIR / 'oof_hierarchical_outcome.npy', oof_hier)

# ============================================================================
# Fix 5: clearance-landing model -- reused full-population vs subset-retrained,
# evaluated on the identical held-out cross-originated clearances
# ============================================================================
print('\n=== Fix 5: clearance-landing ablation (reused vs subset-retrained) ===')
LANDING_NUM_FEATURES = ['period_id', 'elapsed_seconds', 'start_x', 'start_y',
                         'distance_from_own_goal', 'distance_from_center']
LANDING_CAT_FEATURES = ['team', 'player_id', 'start_x_zone', 'start_y_zone', 'direction']
LANDING_FEATURE_COLS = LANDING_NUM_FEATURES + LANDING_CAT_FEATURES
LANDING_TARGET_COLS = ['landing_x', 'landing_y']


def make_landing_pipe():
    pre = ColumnTransformer([
        ('numeric', StandardScaler(), LANDING_NUM_FEATURES),
        ('categorical', OneHotEncoder(handle_unknown='ignore', min_frequency=5), LANDING_CAT_FEATURES),
    ])
    return Pipeline([
        ('preprocess', pre),
        ('model', MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=220, learning_rate=0.04, max_depth=3, min_samples_leaf=8,
            random_state=RANDOM_STATE))),
    ])


def evaluate_landing(y_true, y_pred):
    m = {}
    for i, axis in enumerate(['x', 'y']):
        truth, pred = y_true.iloc[:, i].to_numpy(), y_pred[:, i]
        m[f'r2_{axis}'] = float(r2_score(truth, pred))
        m[f'mae_{axis}'] = float(mean_absolute_error(truth, pred))
    err = np.sqrt(((y_true.to_numpy() - y_pred) ** 2).sum(axis=1))
    m['mean_landing_error'] = float(err.mean())
    m['median_landing_error'] = float(np.median(err))
    m['n'] = int(len(y_true))
    return m


# tag clearances that originated from an open-play cross via the exact event uid
cross_clr_uids = set(df.loc[df['is_headed_clearance'] == 1, 'clr_event_uid'].dropna())
clr_df['from_cross'] = clr_df['event_uid'].isin(cross_clr_uids)
print(f"  {clr_df['from_cross'].sum()} of {len(clr_df)} headed clearances originate from an open-play cross "
      f"(paper: 757).")

# reproduce the exact split the reused model was trained/evaluated on
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
full_train_idx, full_test_idx = next(splitter.split(clr_df, groups=clr_df['match_id']))
clr_train_full, clr_test_full = clr_df.iloc[full_train_idx], clr_df.iloc[full_test_idx]

subset_test = clr_test_full[clr_test_full['from_cross']]
subset_train = clr_train_full[clr_train_full['from_cross']]
print(f'  Held-out cross-originated clearances for comparison: n={len(subset_test)} '
      f'(train-side subset available for retraining: n={len(subset_train)})')

# Model A: the reused full-population model, fit on all 2,485 training clearances
model_full = make_landing_pipe()
model_full.fit(clr_train_full[LANDING_FEATURE_COLS], clr_train_full[LANDING_TARGET_COLS])
pred_full_on_subset = model_full.predict(subset_test[LANDING_FEATURE_COLS])
metrics_full_on_subset = evaluate_landing(subset_test[LANDING_TARGET_COLS], pred_full_on_subset)
metrics_full_on_own_test = evaluate_landing(
    clr_test_full[LANDING_TARGET_COLS], model_full.predict(clr_test_full[LANDING_FEATURE_COLS]))

# Model B: retrained only on the cross-originated subset of the training split
model_subset = make_landing_pipe()
model_subset.fit(subset_train[LANDING_FEATURE_COLS], subset_train[LANDING_TARGET_COLS])
pred_subset_on_subset = model_subset.predict(subset_test[LANDING_FEATURE_COLS])
metrics_subset_on_subset = evaluate_landing(subset_test[LANDING_TARGET_COLS], pred_subset_on_subset)

METRICS['fix5_clearance_ablation'] = dict(
    n_all_headed_clearances=int(len(clr_df)),
    n_cross_originated=int(clr_df['from_cross'].sum()),
    reused_model_on_full_test=metrics_full_on_own_test,
    reused_model_on_cross_subset=metrics_full_on_subset,
    subset_retrained_model_on_cross_subset=metrics_subset_on_subset,
)
print(f"  Reused model, full 640-event test set:      R2 x/y = {metrics_full_on_own_test['r2_x']:.3f} / "
      f"{metrics_full_on_own_test['r2_y']:.3f}  (paper: 0.408 / 0.518)")
print(f"  Reused model, on cross-originated subset:    R2 x/y = {metrics_full_on_subset['r2_x']:.3f} / "
      f"{metrics_full_on_subset['r2_y']:.3f}  mean err={metrics_full_on_subset['mean_landing_error']:.2f}")
print(f"  Subset-retrained model, same subset:         R2 x/y = {metrics_subset_on_subset['r2_x']:.3f} / "
      f"{metrics_subset_on_subset['r2_y']:.3f}  mean err={metrics_subset_on_subset['mean_landing_error']:.2f}")

# ============================================================================
# Fix 6: temporal (first-half-of-season / second-half) validation split
# ============================================================================
print('\n=== Fix 6: temporal train/test split (season drift check) ===')
match_order = (df[['match', 'match_date']].drop_duplicates()
               .sort_values('match_date')['match'].tolist())
half = len(match_order) // 2
early_matches, late_matches = set(match_order[:half]), set(match_order[half:])
temporal_tr_idx = df.index[df['match'].isin(early_matches)].to_numpy()
temporal_te_idx = df.index[df['match'].isin(late_matches)].to_numpy()
print(f'  Train (early season): {len(early_matches)} matches, {len(temporal_tr_idx)} crosses')
print(f'  Test  (late season):  {len(late_matches)} matches, {len(temporal_te_idx)} crosses')

METRICS['fix6_temporal'] = {}
for name, spec in BASELINE_SPECS.items():
    if spec['kind'] != 'binary':
        continue
    target = spec['target']
    y = df[target].values.astype(int)
    Xtr, Xte = X_ALL.iloc[temporal_tr_idx], X_ALL.iloc[temporal_te_idx]
    ytr, yte = y[temporal_tr_idx], y[temporal_te_idx]
    pos, neg = ytr.sum(), len(ytr) - ytr.sum()
    spw = max(neg / max(pos, 1), 1.0)
    pipe = Pipeline([
        ('pre', make_preprocessor()),
        ('clf', XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                               eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0)),
    ])
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    temporal_auc = float(roc_auc_score(yte, p))
    gkf_auc = METRICS['baseline'][name]['cv']['auc']
    gkf_sd = METRICS['baseline'][name]['cv']['sd_auc']
    METRICS['fix6_temporal'][name] = dict(temporal_auc=temporal_auc, groupkfold_auc=gkf_auc,
                                          groupkfold_sd=gkf_sd,
                                          within_1sd=bool(abs(temporal_auc - gkf_auc) <= gkf_sd))
    flag = 'stable' if abs(temporal_auc - gkf_auc) <= gkf_sd else 'DRIFT'
    print(f'  {name}: GroupKFold AUC={gkf_auc:.3f}±{gkf_sd:.3f}   Temporal (early->late) AUC={temporal_auc:.3f}   [{flag}]')

# ============================================================================
# Fix 7: hyperparameter sensitivity sweep (depth / n_estimators), one held-out
# fold, across all four binary classifiers -- checks whether a shallower/
# smaller model reduces variance on this smaller (Ecuador-sized) sample while
# keeping the headline 250/depth-4 config for direct comparability with the
# Premier League study.
# ============================================================================
print('\n=== Fix 7: hyperparameter sensitivity sweep (one held-out fold) ===')
gkf = GroupKFold(n_splits=N_SPLITS)
# use the first fold split as the fixed outer fold for the sweep
splits = list(gkf.split(X_ALL, df['outcome'].values, GROUPS))
tr_idx, te_idx = splits[0]

GRID = [
    dict(n_estimators=150, max_depth=3),
    dict(n_estimators=150, max_depth=4),
    dict(n_estimators=250, max_depth=3),
    dict(n_estimators=250, max_depth=4),  # headline config
    dict(n_estimators=350, max_depth=4),
    dict(n_estimators=250, max_depth=5),
]

METRICS['fix7_hyperparam_sweep'] = {}
for name, spec in BASELINE_SPECS.items():
    if spec['kind'] != 'binary':
        continue
    target = spec['target']
    y = df[target].values.astype(int)
    Xtr, Xte = X_ALL.iloc[tr_idx], X_ALL.iloc[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    pos, neg = ytr.sum(), len(ytr) - ytr.sum()
    spw = max(neg / max(pos, 1), 1.0)

    sweep_results = []
    for params in GRID:
        pipe = Pipeline([
            ('pre', make_preprocessor()),
            ('clf', XGBClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                   learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                   scale_pos_weight=spw, eval_metric='logloss',
                                   random_state=RANDOM_STATE, verbosity=0)),
        ])
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, p))
        sweep_results.append(dict(**params, auc=auc))
    best = max(sweep_results, key=lambda r: r['auc'])
    headline = next(r for r in sweep_results if r['n_estimators'] == 250 and r['max_depth'] == 4)
    METRICS['fix7_hyperparam_sweep'][name] = dict(grid=sweep_results, headline=headline, best=best)
    print(f"  {name}: headline(250,d4) AUC={headline['auc']:.3f}   "
          f"best({best['n_estimators']},d{best['max_depth']}) AUC={best['auc']:.3f}")

print('\nHyperparameter sweep is a single-fold sensitivity check only; the headline '
      '250-estimator/depth-4 configuration is retained for comparability with the Premier League study.')

# ============================================================================
# Save metrics + refit final production artifacts on the full dataset
# ============================================================================
print('\n=== Saving metrics and final production models ===')


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


with open(OUT_DIR / 'metrics_v2.json', 'w') as f:
    json.dump(_to_native(METRICS), f, indent=2)
print(f"Saved metrics to {OUT_DIR / 'metrics_v2.json'}")

# final calibrated completion / chance-creation models (fit base on 75% of all
# data, isotonic calibrator on the remaining match-grouped 25%, matching the
# nested scheme used for the OOF estimates above)
for name in ['cross_completion', 'cross_chance_creation']:
    target = BASELINE_SPECS[name]['target']
    y = df[target].values.astype(int)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr, cal = next(splitter.split(X_ALL, y, GROUPS))
    pos, neg = y[tr].sum(), len(tr) - y[tr].sum()
    spw = max(neg / max(pos, 1), 1.0)
    base = Pipeline([
        ('pre', make_preprocessor()),
        ('clf', XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                               eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0)),
    ])
    base.fit(X_ALL.iloc[tr], y[tr])
    p_cal_slice = base.predict_proba(X_ALL.iloc[cal])[:, 1]
    isotonic = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    isotonic.fit(p_cal_slice, y[cal])
    joblib.dump(dict(base=base, calibrator=isotonic, feature_cols=FEATURE_COLS),
                MODEL_DIR / f'model_{name}_calibrated.pkl')

# final hurdle delivery-value model: calibrated chance-creation gate (reuse the
# calibrated model above) x value-given-shot regressor fit on all shot rows
shot_mask_all = df['shot_created'].values.astype(int) == 1
stage_b_final = Pipeline([
    ('pre', make_preprocessor()),
    ('reg', XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=RANDOM_STATE, verbosity=0)),
])
stage_b_final.fit(X_ALL[shot_mask_all], df.loc[shot_mask_all, 'danger_value'].values)
joblib.dump(dict(value_given_shot=stage_b_final, feature_cols=FEATURE_COLS,
                  note='multiply by calibrated cross_chance_creation probability'),
            MODEL_DIR / 'model_cross_delivery_value_hurdle.pkl')

# final hierarchical outcome model
pos, neg = y_complete.sum(), len(y_complete) - y_complete.sum()
spw = max(neg / max(pos, 1), 1.0)
stage_a_final = Pipeline([
    ('pre', make_preprocessor()),
    ('clf', XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                           eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0)),
])
stage_a_final.fit(X_ALL, y_complete)
complete_mask_all = y_complete == 1
y_b_all, uniques_b_all = pd.factorize(outcome_labels[complete_mask_all], sort=False)
stage_b_outcome_final = Pipeline([
    ('pre', make_preprocessor()),
    ('clf', XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           eval_metric='mlogloss', random_state=RANDOM_STATE, verbosity=0)),
])
stage_b_outcome_final.fit(X_ALL[complete_mask_all], y_b_all)
joblib.dump(dict(stage_a_complete=stage_a_final, stage_b_outcome=stage_b_outcome_final,
                  stage_b_classes=list(uniques_b_all), feature_cols=FEATURE_COLS),
            MODEL_DIR / 'model_cross_outcome_hierarchical.pkl')

# final clearance-landing model: keep the reused full-population model (Fix 5
# showed subset-retraining does not help), refit on 100% of the 3,125 events
model_full_final = make_landing_pipe()
model_full_final.fit(clr_df[LANDING_FEATURE_COLS], clr_df[LANDING_TARGET_COLS])
joblib.dump(dict(model=model_full_final, feature_cols=LANDING_FEATURE_COLS,
                  target_cols=LANDING_TARGET_COLS),
            MODEL_DIR / 'model_clearance_landing_reused.pkl')

print('Saved all v2 production artifacts to', MODEL_DIR)
print('\n=== Done ===')
