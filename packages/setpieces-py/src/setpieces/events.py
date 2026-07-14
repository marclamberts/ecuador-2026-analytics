"""Extract structured set-piece events (corners, free kicks, throw-ins,
penalties) out of a raw Opta match dict."""

from dataclasses import dataclass, asdict
from typing import List, Optional

from . import codes
from .geometry import minute_value, pass_end_xy, qualifier_ids
from .io import sorted_events


@dataclass(frozen=True)
class SetPieceEvent:
    """One set-piece delivery or shot.

    kind: "corner" | "free_kick" | "throw_in" | "penalty"
    subtype: "delivery" for corner/free_kick/throw_in deliveries;
        "direct_shot_<goal|saved|post|miss>" for direct free-kick shots;
        "<goal|saved|post|miss>" for penalties.
    event_index: this event's position in ``sorted_events(match)`` --
        needed to scan forward for second-ball contests.
    """

    kind: str
    subtype: str
    team_id: str
    player_name: Optional[str]
    period: int
    minute: float
    x: float
    y: float
    end_x: Optional[float]
    end_y: Optional[float]
    outcome: Optional[int]
    event_index: int

    def as_dict(self) -> dict:
        return asdict(self)


def extract_set_pieces(match: dict, kinds: Optional[List[str]] = None) -> List[SetPieceEvent]:
    """All set-piece events in a match, in chronological order.

    ``kinds`` optionally restricts to a subset of
    {"corner", "free_kick", "throw_in", "penalty"}.
    """
    events = sorted_events(match)
    out: List[SetPieceEvent] = []

    for i, e in enumerate(events):
        type_id = e.get("typeId")
        qids = qualifier_ids(e)
        kind = subtype = None

        if type_id == codes.TYPE_PASS:
            if codes.QUALIFIER_CORNER in qids:
                kind, subtype = "corner", "delivery"
            elif codes.QUALIFIER_FREE_KICK in qids:
                kind, subtype = "free_kick", "delivery"
            elif codes.QUALIFIER_THROW_IN in qids:
                kind, subtype = "throw_in", "delivery"
        elif type_id in codes.SHOT_TYPES:
            shot_outcome = codes.SHOT_OUTCOME_NAMES[type_id]
            if codes.QUALIFIER_PENALTY in qids:
                kind, subtype = "penalty", shot_outcome
            elif codes.QUALIFIER_FREE_KICK in qids:
                kind, subtype = "free_kick", f"direct_shot_{shot_outcome}"

        if kind is None:
            continue
        if kinds is not None and kind not in kinds:
            continue

        end_x, end_y = pass_end_xy(e) if subtype == "delivery" else (None, None)
        out.append(SetPieceEvent(
            kind=kind, subtype=subtype,
            team_id=e.get("contestantId"), player_name=e.get("playerName"),
            period=e.get("periodId"), minute=minute_value(e),
            x=float(e["x"]), y=float(e["y"]),
            end_x=end_x, end_y=end_y,
            outcome=e.get("outcome"),
            event_index=i,
        ))

    return out
