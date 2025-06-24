import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trend_analysis.io import pct


def test_pct():
    res = pct((1.0, 0.5, 2.0, 3.0, 0.25))
    assert res == [100.0, 50.0, 2.0, 3.0, 25.0]
