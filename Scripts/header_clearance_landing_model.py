"""
Train a model that predicts where headed clearances land.

The Ecuador 2026 JSON files are Opta-style event feeds. In this feed:
  - typeId 12 = clearance
  - qualifierId 15 = header
  - qualifierId 140/141 = landing x/y

Outputs are written to Ecuador 2026/ClearanceLandingModel by default:
  - headed_clearances_dataset.csv
  - headed_clearance_landing_model.joblib
  - headed_clearance_predictions.csv
  - model_metrics.json
  - landing_actual_vs_predicted.png

Usage:
  python3 header_clearance_landing_model.py
  python3 header_clearance_landing_model.py --data-dir "/path/to/Ecuador 2026"
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from datetime import UTC, datetime


LOCAL_PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[1] / "Statsbomb" / ".python_packages"
if LOCAL_PACKAGE_DIR.exists():
    sys.path.insert(0, str(LOCAL_PACKAGE_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLEARANCE_TYPE_ID = 12
HEADER_QUALIFIER_ID = 15
END_X_QUALIFIER_ID = 140
END_Y_QUALIFIER_ID = 141
DIRECTION_QUALIFIER_ID = 56

PITCH_LENGTH = 100.0
PITCH_WIDTH = 100.0


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train headed-clearance landing model.")
    parser.add_argument("--data-dir", type=pathlib.Path, default=here)
    parser.add_argument("--out-dir", type=pathlib.Path, default=here / "ClearanceLandingModel")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def qualifier_map(event: dict) -> dict[int, str | None]:
    return {int(q["qualifierId"]): q.get("value") for q in event.get("qualifier", [])}


def has_qualifier(event: dict, qualifier_id: int) -> bool:
    return any(int(q["qualifierId"]) == qualifier_id for q in event.get("qualifier", []))


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def match_teams_from_filename(path: pathlib.Path) -> tuple[str, str]:
    match = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", path.name)
    if not match:
        return "Unknown Home", "Unknown Away"
    return match.group(1), match.group(2)


def build_contestant_team_map(json_files: list[pathlib.Path]) -> dict[str, str]:
    team_to_cids: dict[str, list[set[str]]] = {}
    for path in json_files:
        home, away = match_teams_from_filename(path)
        with path.open() as f:
            events = json.load(f).get("event", [])
        cids = {str(e["contestantId"]) for e in events if e.get("contestantId")}
        team_to_cids.setdefault(home, []).append(cids)
        team_to_cids.setdefault(away, []).append(cids)

    cid_to_team = {}
    for team, cid_sets in team_to_cids.items():
        common = set.intersection(*cid_sets) if cid_sets else set()
        if len(common) == 1:
            cid_to_team[next(iter(common))] = team
    return cid_to_team


def elapsed_seconds(event: dict) -> float:
    period = int(event.get("periodId") or 0)
    base = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60}.get(period, 0)
    return base + int(event.get("timeMin") or 0) * 60 + int(event.get("timeSec") or 0)


def zone_x(x: float) -> str:
    if x < 33.333:
        return "defensive_third"
    if x < 66.667:
        return "middle_third"
    return "attacking_third"


def zone_y(y: float) -> str:
    if y < 33.333:
        return "left_channel"
    if y < 66.667:
        return "central_channel"
    return "right_channel"


def extract_headed_clearances(data_dir: pathlib.Path) -> pd.DataFrame:
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No match JSON files found in {data_dir}")

    cid_to_team = build_contestant_team_map(json_files)
    rows = []

    for path in json_files:
        with path.open() as f:
            data = json.load(f)

        home_team, away_team = match_teams_from_filename(path)
        match_id = path.stem

        for event in data.get("event", []):
            if int(event.get("typeId") or -1) != CLEARANCE_TYPE_ID:
                continue
            if not has_qualifier(event, HEADER_QUALIFIER_ID):
                continue

            qmap = qualifier_map(event)
            landing_x = safe_float(qmap.get(END_X_QUALIFIER_ID))
            landing_y = safe_float(qmap.get(END_Y_QUALIFIER_ID))
            start_x = safe_float(event.get("x"))
            start_y = safe_float(event.get("y"))
            if any(math.isnan(v) for v in (landing_x, landing_y, start_x, start_y)):
                continue

            contestant_id = str(event.get("contestantId") or "")
            team = cid_to_team.get(contestant_id, contestant_id or "Unknown")

            rows.append(
                {
                    "match_id": match_id,
                    "match_file": path.name,
                    "home_team": home_team,
                    "away_team": away_team,
                    "team": team,
                    "contestant_id": contestant_id,
                    "player_id": str(event.get("playerId") or "Unknown"),
                    "player_name": event.get("playerName") or "Unknown",
                    "event_id": int(event.get("eventId") or -1),
                    "period_id": int(event.get("periodId") or 0),
                    "elapsed_seconds": elapsed_seconds(event),
                    "start_x": start_x,
                    "start_y": start_y,
                    "start_x_zone": zone_x(start_x),
                    "start_y_zone": zone_y(start_y),
                    "direction": qmap.get(DIRECTION_QUALIFIER_ID) or "Unknown",
                    "distance_from_own_goal": start_x,
                    "distance_from_center": abs(start_y - PITCH_WIDTH / 2),
                    "landing_x": landing_x,
                    "landing_y": landing_y,
                    "clearance_length": math.hypot(landing_x - start_x, landing_y - start_y),
                    "clearance_dx": landing_x - start_x,
                    "clearance_dy": landing_y - start_y,
                }
            )

    df = pd.DataFrame(rows).sort_values(["match_id", "period_id", "elapsed_seconds", "event_id"])
    if df.empty:
        raise ValueError("No headed clearances with landing coordinates found.")
    return df


def make_model() -> Pipeline:
    numeric_features = [
        "period_id",
        "elapsed_seconds",
        "start_x",
        "start_y",
        "distance_from_own_goal",
        "distance_from_center",
    ]
    categorical_features = [
        "team",
        "player_id",
        "start_x_zone",
        "start_y_zone",
        "direction",
    ]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", encoder, categorical_features),
        ],
        remainder="drop",
    )

    regressor = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=220,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=8,
            random_state=42,
        )
    )
    return Pipeline([("preprocess", preprocessor), ("model", regressor)])


def evaluate(y_true: pd.DataFrame, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {}
    for idx, axis in enumerate(["x", "y"]):
        truth = y_true.iloc[:, idx].to_numpy()
        pred = y_pred[:, idx]
        metrics[f"mae_{axis}"] = float(mean_absolute_error(truth, pred))
        metrics[f"rmse_{axis}"] = float(mean_squared_error(truth, pred) ** 0.5)
        metrics[f"r2_{axis}"] = float(r2_score(truth, pred))

    euclidean = np.sqrt(((y_true.to_numpy() - y_pred) ** 2).sum(axis=1))
    metrics["mean_landing_error"] = float(euclidean.mean())
    metrics["median_landing_error"] = float(np.median(euclidean))
    metrics["pct_within_10_pitch_units"] = float((euclidean <= 10).mean())
    metrics["pct_within_15_pitch_units"] = float((euclidean <= 15).mean())
    return metrics


def save_plot(predictions: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    fig.patch.set_facecolor("#f7f7f4")

    for ax, x_col, y_col, title, color in [
        (axes[0], "landing_x", "landing_y", "Actual headed-clearance landings", "#2f8fd1"),
        (axes[1], "pred_landing_x", "pred_landing_y", "Predicted headed-clearance landings", "#f06fa3"),
    ]:
        ax.set_facecolor("#fbfbf8")
        ax.scatter(predictions[x_col], predictions[y_col], s=18, alpha=0.55, c=color, edgecolors="none")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(color="#d8d8d2", linewidth=0.6, alpha=0.7)

    fig.suptitle("Held-out Match Test Set", fontsize=15, weight="bold")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = extract_headed_clearances(args.data_dir)
    dataset_path = args.out_dir / "headed_clearances_dataset.csv"
    df.to_csv(dataset_path, index=False)

    feature_cols = [
        "period_id",
        "elapsed_seconds",
        "start_x",
        "start_y",
        "distance_from_own_goal",
        "distance_from_center",
        "team",
        "player_id",
        "start_x_zone",
        "start_y_zone",
        "direction",
    ]
    target_cols = ["landing_x", "landing_y"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["match_id"]))
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    model = make_model()
    model.fit(train[feature_cols], train[target_cols])

    train_pred = model.predict(train[feature_cols])
    test_pred = model.predict(test[feature_cols])
    baseline_pred = np.tile(train[target_cols].median().to_numpy(), (len(test), 1))

    metrics = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_dir": str(args.data_dir.resolve()),
        "n_headed_clearances": int(len(df)),
        "n_matches": int(df["match_id"].nunique()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train": evaluate(train[target_cols], train_pred),
        "test": evaluate(test[target_cols], test_pred),
        "baseline_test": evaluate(test[target_cols], baseline_pred),
        "notes": {
            "event_filter": "typeId 12 clearances with qualifierId 15 header",
            "target": "qualifierId 140/141 landing coordinates",
            "split": "GroupShuffleSplit by match_id",
            "baseline": "predicts the train-set median landing x/y for every test event",
        },
    }

    model_path = args.out_dir / "headed_clearance_landing_model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "metrics": metrics,
        },
        model_path,
    )

    test = test.assign(
        pred_landing_x=np.clip(test_pred[:, 0], 0, 100),
        pred_landing_y=np.clip(test_pred[:, 1], 0, 100),
    )
    test["landing_error"] = np.sqrt(
        (test["landing_x"] - test["pred_landing_x"]) ** 2
        + (test["landing_y"] - test["pred_landing_y"]) ** 2
    )
    predictions_path = args.out_dir / "headed_clearance_predictions.csv"
    test.to_csv(predictions_path, index=False)

    metrics_path = args.out_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_path = args.out_dir / "landing_actual_vs_predicted.png"
    save_plot(test, plot_path)

    print(f"Extracted headed clearances: {len(df):,}")
    print(f"Matches: {df['match_id'].nunique():,}")
    print(f"Train rows: {len(train):,} | Test rows: {len(test):,}")
    print(f"Test mean landing error: {metrics['test']['mean_landing_error']:.2f} pitch units")
    print(f"Test median landing error: {metrics['test']['median_landing_error']:.2f} pitch units")
    print(f"Saved dataset: {dataset_path}")
    print(f"Saved model: {model_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
