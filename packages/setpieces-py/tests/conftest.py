from pathlib import Path

import pytest

import setpieces as sp

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def match():
    return sp.load_match(str(DATA_DIR / "sample_match.json"))
