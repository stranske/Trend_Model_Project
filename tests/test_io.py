from trend_analysis.io import pct


def test_pct():
    values = (0.1, 0.2, 3.0, 4.0, 0.5)
    result = pct(values)
    assert result == [10.0, 20.0, 3.0, 4.0, 50.0]
