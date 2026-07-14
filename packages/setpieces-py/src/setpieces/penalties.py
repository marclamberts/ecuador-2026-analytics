"""Penalty award/conversion summaries."""

import collections
from typing import Iterable

from .events import SetPieceEvent


def penalty_summary(events: Iterable[SetPieceEvent]) -> dict:
    """Awarded/scored/saved/missed/post counts and conversion rate (%)
    across a collection of ``SetPieceEvent``s (only ``kind == "penalty"``
    entries are counted)."""
    subtypes = [e.subtype for e in events if e.kind == "penalty"]
    counts = collections.Counter(subtypes)
    awarded = len(subtypes)
    scored = counts.get("goal", 0)
    return {
        "awarded": awarded,
        "scored": scored,
        "saved": counts.get("saved", 0),
        "missed": counts.get("miss", 0),
        "post": counts.get("post", 0),
        "conversion_rate": (scored / awarded * 100.0) if awarded else None,
    }
