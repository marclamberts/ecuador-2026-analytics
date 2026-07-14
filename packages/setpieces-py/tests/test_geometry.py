from setpieces.geometry import distance_m


def test_distance_m_zero():
    assert distance_m(50, 50, 50, 50) == 0.0


def test_distance_m_full_length():
    # (0,50) -> (100,50) spans the full 105m pitch length
    assert distance_m(0, 50, 100, 50, pitch_length=105.0, pitch_width=68.0) == 105.0


def test_distance_m_full_width():
    assert distance_m(50, 0, 50, 100, pitch_length=105.0, pitch_width=68.0) == 68.0
