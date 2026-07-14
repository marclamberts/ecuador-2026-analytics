import setpieces as sp
from setpieces.zones import classify_zone


def test_classify_zone_short():
    assert classify_zone(end_x=70.0, end_y=50.0, start_y=100.0) == "short"


def test_classify_zone_near_six():
    # start_y >= 50 mirrors end_y; end_y=30 stays 30 when start_y < 50
    assert classify_zone(end_x=96.0, end_y=30.0, start_y=0.0) == ("near", "six")


def test_classify_zone_central_edge_mirrored():
    # start_y >= 50 mirrors end_y -> my = 100 - 45 = 55 (central)
    assert classify_zone(end_x=88.0, end_y=45.0, start_y=100.0) == ("central", "edge")


def test_zone_breakdown_and_percentages(match):
    corners = sp.extract_set_pieces(match, kinds=["corner"])
    team_a_corners = [c for c in corners if c.team_id == "t_A"]
    counts = sp.zone_breakdown(team_a_corners)
    assert counts == {("central", "edge"): 1, ("near", "six"): 1}

    pct = sp.zone_percentages(team_a_corners)
    assert pct[("central", "edge")] == 50.0
    assert pct[("near", "six")] == 50.0

    team_b_corners = [c for c in corners if c.team_id == "t_B"]
    assert sp.zone_breakdown(team_b_corners) == {"short": 1}
