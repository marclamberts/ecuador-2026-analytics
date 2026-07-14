# setpieces packages

Two mirrored, installable packages for extracting and analyzing
football set-piece events (corners, free kicks, throw-ins, penalties)
from Opta MA1-style event data -- the same feed shape used by the
`Event/*.json` files in this repo. Each package parses raw events into
structured records, classifies corner/free-kick delivery zones
(near/central/far post x six-yard/edge-of-box, mirrored onto one
attacking side), finds "second ball" contests after a team's own set
pieces, and summarizes penalty conversion.

- **[setpieces-py](setpieces-py/)** -- Python package, zero required
  dependencies, optional matplotlib-based plotting.
- **[setpieces-r](setpieces-r/setpieces/)** -- R package, only
  `jsonlite` required, optional ggplot2-based plotting.

Both expose the same API shape (`extract_set_pieces`, `zone_percentages`,
`find_second_ball_contests`, `penalty_summary`, ...) so analysis written
against one translates directly to the other. See each package's own
README for install instructions and a quick-start example against this
repo's real match data.

These packages generalize the corner-zone and second-ball logic
originally built as one-off scripts in `Scripts/corner_routines.py` and
`Scripts/set_piece_second_balls.py`.
