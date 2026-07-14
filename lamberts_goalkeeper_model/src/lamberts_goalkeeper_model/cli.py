"""Command-line entry point: lamberts-goalkeeper-model <season_dir> [--out-dir ...]"""

from __future__ import annotations

import argparse
import pathlib
import sys

from .loader import SeasonDataError
from .model import MIN_MINUTES_FOR_RANKING_DEFAULT, build_goalkeeper_value_model

RESULT_COLS = [
    "player", "team", "minutes", "matches",
    "goalkeeper_value_index", "goalkeeper_value_index_pctile", "goalkeeper_value_index_zscore",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="lamberts-goalkeeper-model",
        description="Automatically load a season data folder and build the composite Goalkeeper Value Index.",
    )
    parser.add_argument("season_dir", help="Path to a season folder containing Aggregated/, Danger/, and Event/ subfolders")
    parser.add_argument("--out-dir", default=None, help="Where to write the output CSVs (default: <season_dir>/Lamberts Goalkeeper Model)")
    parser.add_argument("--min-minutes", type=float, default=MIN_MINUTES_FOR_RANKING_DEFAULT, help="Minutes threshold for ranking (default: 450)")
    args = parser.parse_args(argv)

    try:
        result = build_goalkeeper_value_model(args.season_dir, min_minutes=args.min_minutes)
    except SeasonDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(args.season_dir) / "Lamberts Goalkeeper Model"
    result.save(out_dir)

    print(f"Wrote {len(result.match_df)} keeper-match rows and {len(result.season_df)} ranked keeper-season rows -> {out_dir}")
    print(result.season_df[RESULT_COLS].head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
