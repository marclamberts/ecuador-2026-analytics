import setpieces as sp


def test_penalty_summary(match):
    events = sp.extract_set_pieces(match)
    summary = sp.penalty_summary(events)
    assert summary["awarded"] == 2
    assert summary["scored"] == 1
    assert summary["missed"] == 1
    assert summary["saved"] == 0
    assert summary["post"] == 0
    assert summary["conversion_rate"] == 50.0


def test_penalty_summary_empty():
    summary = sp.penalty_summary([])
    assert summary["awarded"] == 0
    assert summary["conversion_rate"] is None
