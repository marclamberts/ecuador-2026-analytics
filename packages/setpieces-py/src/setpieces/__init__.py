"""setpieces -- extract and analyze football set-piece events from Opta
MA1-style event data: corners, free kicks, throw-ins, and penalties."""

from .events import SetPieceEvent, extract_set_pieces
from .geometry import distance_m
from .io import load_match, load_matches, sorted_events
from .penalties import penalty_summary
from .secondballs import SecondBallContest, find_second_ball_contests
from .teams import team_ids_from_filenames
from .zones import ZoneParams, classify_zone, zone_breakdown, zone_percentages

__version__ = "0.1.0"

__all__ = [
    "SetPieceEvent",
    "extract_set_pieces",
    "distance_m",
    "load_match",
    "load_matches",
    "sorted_events",
    "penalty_summary",
    "SecondBallContest",
    "find_second_ball_contests",
    "team_ids_from_filenames",
    "ZoneParams",
    "classify_zone",
    "zone_breakdown",
    "zone_percentages",
]
