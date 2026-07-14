"""Second-ball contest detection after a team's own set pieces.

For each qualifying delivery, look at the next events (within a short
time window) for contested actions (aerial duel, tackle, interception,
clearance, ball recovery). The *second* such contested action is treated
as "the second ball" -- the first is usually just the target player's
initial header/contact, the second is who actually comes away with the
loose ball. This is very often well outside the box, since a strong
clearance can travel 40+ metres before the second ball is won.
"""

from dataclasses import dataclass
from typing import List, Optional

from . import codes
from .events import SetPieceEvent
from .geometry import minute_value

FINAL_THIRD_START = 200.0 / 3.0


@dataclass(frozen=True)
class SecondBallContest:
    delivery: SetPieceEvent
    x: float
    y: float
    won: bool
    winner_player: Optional[str]


def find_second_ball_contests(events: List[dict], set_pieces: List[SetPieceEvent], team_id: str,
                               window_events: int = 5, window_minutes: float = 12 / 60,
                               max_clean_passes_between: int = 1,
                               final_third_start: float = FINAL_THIRD_START) -> List[SecondBallContest]:
    """Second-ball contests following ``team_id``'s own corners, free
    kicks, throw-ins and penalty-box direct free-kick shots.

    ``events`` must be ``setpieces.sorted_events(match)`` for the same
    match that ``set_pieces`` (``setpieces.extract_set_pieces(match)``)
    was built from -- contest detection scans forward from each
    delivery's ``event_index`` in that list.
    """
    results: List[SecondBallContest] = []

    for sp in set_pieces:
        if sp.team_id != team_id:
            continue
        if sp.end_x is None or sp.end_x < final_third_start:
            continue

        i = sp.event_index
        t0 = minute_value(events[i])
        window = []
        for ev in events[i + 1:i + 1 + window_events]:
            if minute_value(ev) - t0 > window_minutes:
                break
            window.append(ev)

        first_idx = next((j for j, ev in enumerate(window) if ev.get("typeId") in codes.CONTESTED_TYPES), None)
        if first_idx is None:
            continue

        second_ball, clean_passes = None, 0
        for ev in window[first_idx + 1:]:
            if ev.get("typeId") in codes.CONTESTED_TYPES:
                second_ball = ev
                break
            if ev.get("typeId") == codes.TYPE_PASS and ev.get("outcome") == 1:
                clean_passes += 1
                if clean_passes > max_clean_passes_between:
                    break

        if second_ball is None or second_ball.get("x") is None or second_ball.get("y") is None:
            continue

        won = second_ball.get("contestantId") == team_id
        results.append(SecondBallContest(
            delivery=sp, x=float(second_ball["x"]), y=float(second_ball["y"]),
            won=won, winner_player=second_ball.get("playerName"),
        ))

    return results
