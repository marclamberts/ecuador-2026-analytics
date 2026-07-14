"""Loading Opta MA1-style match JSON files."""

import json
from typing import List


def load_match(path: str) -> dict:
    """Load a single match JSON file (must contain an ``event`` list)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_matches(paths: List[str]) -> List[dict]:
    """Load several match JSON files, in the given order."""
    return [load_match(p) for p in paths]


def sorted_events(match: dict) -> List[dict]:
    """This match's events, regular time and extra time only, in the
    order they actually happened (period, minute, eventId)."""
    from .geometry import minute_value

    events = [e for e in match.get("event", []) if e.get("periodId") in (1, 2)]
    events.sort(key=lambda e: (e["periodId"], minute_value(e), e.get("eventId", 0)))
    return events
