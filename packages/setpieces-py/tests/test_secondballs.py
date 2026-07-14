import setpieces as sp


def test_second_ball_won_by_delivering_team(match):
    events = sp.sorted_events(match)
    set_pieces = sp.extract_set_pieces(match)
    contests = sp.find_second_ball_contests(events, set_pieces, "t_A")

    assert len(contests) == 1
    contest = contests[0]
    assert contest.won is True
    assert contest.winner_player == "A. Second Ball Winner"
    assert contest.x == 60.0
    assert contest.y == 40.0
    assert contest.delivery.kind == "corner"


def test_second_ball_none_for_team_with_no_qualifying_deliveries(match):
    events = sp.sorted_events(match)
    set_pieces = sp.extract_set_pieces(match)
    # t_B's only delivery is the last event in the match, so there's
    # nothing after it to scan for a contest
    contests = sp.find_second_ball_contests(events, set_pieces, "t_B")
    assert contests == []
