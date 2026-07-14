"""Delivery-zone classification for corners and free kicks.

Deliveries are mirrored onto a single attacking side (near-post low,
far-post high) so left- and right-side set pieces combine into one
near/central/far-post x six-yard/edge-of-box 6-zone breakdown, plus a
"short" bucket for deliveries that don't reach the box at all.
"""

import collections
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Union

from .events import SetPieceEvent

ZoneKey = Union[str, Tuple[str, str]]


@dataclass(frozen=True)
class ZoneParams:
    box_front_x: float = 83.0
    six_yard_front_x: float = 94.0
    near_cut: float = 41.0
    far_cut: float = 59.0


DEFAULT_ZONE_PARAMS = ZoneParams()


def classify_zone(end_x: float, end_y: float, start_y: float,
                   params: ZoneParams = DEFAULT_ZONE_PARAMS) -> ZoneKey:
    """Classify a single delivery's end location into a zone key.

    Returns "short" if the ball never reached the penalty box, otherwise
    a (near|central|far, six|edge) tuple.
    """
    if end_x < params.box_front_x:
        return "short"
    # mirror onto one side: deliveries from the right (start_y >= 50)
    # get their y flipped so "near post" always means the same thing
    my = 100 - end_y if start_y >= 50 else end_y
    col = "near" if my < params.near_cut else ("far" if my > params.far_cut else "central")
    row = "six" if end_x >= params.six_yard_front_x else "edge"
    return (col, row)


def zone_breakdown(events: Iterable[SetPieceEvent],
                    params: ZoneParams = DEFAULT_ZONE_PARAMS) -> Dict[ZoneKey, int]:
    """Raw counts per zone key across the given deliveries (events
    without an end location are skipped)."""
    counts: collections.Counter = collections.Counter()
    for e in events:
        if e.end_x is None or e.end_y is None:
            continue
        counts[classify_zone(e.end_x, e.end_y, e.y, params)] += 1
    return dict(counts)


def zone_percentages(events: Iterable[SetPieceEvent],
                      params: ZoneParams = DEFAULT_ZONE_PARAMS) -> Dict[ZoneKey, float]:
    """Each zone's share of all classified deliveries, as a percentage."""
    counts = zone_breakdown(events, params)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {zone: n / total * 100.0 for zone, n in counts.items()}
