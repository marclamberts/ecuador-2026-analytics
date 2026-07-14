import setpieces as sp


def test_extract_all_kinds(match):
    events = sp.extract_set_pieces(match)
    kinds = sorted(e.kind for e in events)
    assert kinds == sorted([
        "corner", "corner", "corner", "free_kick", "throw_in", "penalty", "penalty",
    ])


def test_extract_filters_by_kind(match):
    corners = sp.extract_set_pieces(match, kinds=["corner"])
    assert len(corners) == 3
    assert all(e.kind == "corner" for e in corners)


def test_corner_fields(match):
    corners = sp.extract_set_pieces(match, kinds=["corner"])
    first = corners[0]
    assert first.team_id == "t_A"
    assert first.player_name == "A. Corner Taker"
    assert first.subtype == "delivery"
    assert first.end_x == 88.0
    assert first.end_y == 45.0


def test_penalty_subtypes(match):
    penalties = sp.extract_set_pieces(match, kinds=["penalty"])
    subtypes = sorted(p.subtype for p in penalties)
    assert subtypes == ["goal", "miss"]


def test_event_index_matches_sorted_events(match):
    events = sp.sorted_events(match)
    set_pieces = sp.extract_set_pieces(match)
    for sp_event in set_pieces:
        raw = events[sp_event.event_index]
        assert raw.get("contestantId") == sp_event.team_id
