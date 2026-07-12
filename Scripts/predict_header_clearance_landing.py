"""
Predict where a headed clearance will land for a player.

Examples:
  python3 predict_header_clearance_landing.py "C. Gruezo"
  python3 predict_header_clearance_landing.py "C. Gruezo" --start-x 15 --start-y 45
  python3 predict_header_clearance_landing.py "C. Gruezo" --start-x 15 --start-y 45 --team "Manta FC"
"""

from __future__ import annotations

import argparse
import pathlib
import sys


LOCAL_PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[1] / "Statsbomb" / ".python_packages"
if LOCAL_PACKAGE_DIR.exists():
    sys.path.insert(0, str(LOCAL_PACKAGE_DIR))

import joblib
import numpy as np
import pandas as pd


HERE = pathlib.Path(__file__).resolve().parent
MODEL_DIR = HERE / "ClearanceLandingModel"
MODEL_PATH = MODEL_DIR / "headed_clearance_landing_model.joblib"
DATASET_PATH = MODEL_DIR / "headed_clearances_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict headed-clearance landing x/y.")
    parser.add_argument("player_name", help="Player name as it appears in the event data.")
    parser.add_argument("--start-x", type=float, help="Clearance start x on Opta 0-100 pitch.")
    parser.add_argument("--start-y", type=float, help="Clearance start y on Opta 0-100 pitch.")
    parser.add_argument("--team", help="Optional team name. Defaults to player's most common team.")
    parser.add_argument("--period-id", type=int, default=1)
    parser.add_argument("--minute", type=float, help="Match minute. Defaults to player's median headed clearance minute.")
    parser.add_argument("--direction", help="Optional qualifier 56 direction. Defaults to player's most common value.")
    return parser.parse_args()


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


def mode_or_unknown(series: pd.Series) -> str:
    mode = series.dropna().astype(str).mode()
    return mode.iloc[0] if len(mode) else "Unknown"


def main() -> None:
    args = parse_args()
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(DATASET_PATH)
    player_rows = df[df["player_name"].str.casefold() == args.player_name.casefold()]
    if player_rows.empty:
        close = df[df["player_name"].str.contains(args.player_name, case=False, na=False)]["player_name"].drop_duplicates()
        if len(close):
            print("No exact player match. Possible matches:")
            for name in close.head(20):
                print(f"  - {name}")
        else:
            print(f"No headed-clearance records found for '{args.player_name}'.")
        raise SystemExit(1)

    start_x = args.start_x if args.start_x is not None else float(player_rows["start_x"].median())
    start_y = args.start_y if args.start_y is not None else float(player_rows["start_y"].median())
    elapsed_seconds = (
        args.minute * 60
        if args.minute is not None
        else float(player_rows["elapsed_seconds"].median())
    )

    row = {
        "period_id": args.period_id,
        "elapsed_seconds": elapsed_seconds,
        "start_x": start_x,
        "start_y": start_y,
        "distance_from_own_goal": start_x,
        "distance_from_center": abs(start_y - 50),
        "team": args.team or mode_or_unknown(player_rows["team"]),
        "player_id": mode_or_unknown(player_rows["player_id"]),
        "start_x_zone": zone_x(start_x),
        "start_y_zone": zone_y(start_y),
        "direction": args.direction or mode_or_unknown(player_rows["direction"]),
    }

    pred = model.predict(pd.DataFrame([row], columns=feature_cols))[0]
    pred_x, pred_y = np.clip(pred, 0, 100)

    print(f"Player: {player_rows.iloc[0]['player_name']}")
    print(f"Input start: x={start_x:.1f}, y={start_y:.1f}")
    print(f"Predicted landing: x={pred_x:.1f}, y={pred_y:.1f}")
    print(f"Typical model test error: {bundle['metrics']['test']['mean_landing_error']:.1f} pitch units")


if __name__ == "__main__":
    main()
