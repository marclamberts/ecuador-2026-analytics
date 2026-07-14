"""
Builds the Ecuador 2026 Lamberts Goalkeeper Model using the installable
`lamberts_goalkeeper_model` package (../lamberts_goalkeeper_model/).

The 13-submodel methodology, data sources, and match-level join logic
are documented in `lamberts_goalkeeper_model/README.md` and
`Lamberts Goalkeeper Model/README.md`. This script just points the
package at this repo (which is itself a valid season data folder --
Aggregated/, Danger/, Event/ live at the repo root) and writes the
output to the usual place.

Usage: python3 build_goalkeeper_value_model.py
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE_SRC = ROOT / "lamberts_goalkeeper_model" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from lamberts_goalkeeper_model import build_goalkeeper_value_model  # noqa: E402

OUT_DIR = ROOT / "Lamberts Goalkeeper Model"

RESULT_COLS = [
    "player", "team", "minutes", "matches",
    "goalkeeper_value_index", "goalkeeper_value_index_pctile", "goalkeeper_value_index_zscore",
]


def main() -> None:
    result = build_goalkeeper_value_model(ROOT)
    result.save(OUT_DIR)

    print(f"Wrote {len(result.match_df)} keeper-match rows -> {OUT_DIR / 'goalkeeper_match_value.csv'}")
    print(f"Wrote {len(result.season_df)} ranked keeper-season rows -> {OUT_DIR / 'goalkeeper_season_value_model.csv'}")
    print(result.season_df[RESULT_COLS].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
