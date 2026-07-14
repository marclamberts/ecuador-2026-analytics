"""Coordinate helpers for Opta's 0-100 x 0-100 normalized pitch."""

from typing import Optional


def distance_m(x0: float, y0: float, x1: float, y1: float,
                pitch_length: float = 105.0, pitch_width: float = 68.0) -> float:
    """Real-world distance (metres) between two points given in Opta's
    0-100 normalized coordinates."""
    dx = (x1 - x0) / 100.0 * pitch_length
    dy = (y1 - y0) / 100.0 * pitch_width
    return (dx ** 2 + dy ** 2) ** 0.5


def minute_value(event: dict) -> float:
    """Match minute (with fractional seconds) for an Opta event dict."""
    return float(event.get("timeMin") or 0) + float(event.get("timeSec") or 0) / 60.0


def qualifier_map(event: dict) -> dict:
    """qualifierId -> value for an Opta event dict."""
    return {q["qualifierId"]: q.get("value") for q in event.get("qualifier", [])}


def qualifier_ids(event: dict) -> set:
    return {q["qualifierId"] for q in event.get("qualifier", [])}


def pass_end_xy(event: dict) -> "tuple[Optional[float], Optional[float]]":
    """End x/y of a pass, falling back to the start location if the pass
    has no end-location qualifiers (e.g. some fouled/blocked deliveries)."""
    from .codes import QUALIFIER_PASS_END_X, QUALIFIER_PASS_END_Y

    qmap = qualifier_map(event)
    ex, ey = qmap.get(QUALIFIER_PASS_END_X), qmap.get(QUALIFIER_PASS_END_Y)
    x = float(ex) if ex is not None else event.get("x")
    y = float(ey) if ey is not None else event.get("y")
    return (float(x) if x is not None else None), (float(y) if y is not None else None)
