from trend_analysis import io


def test_pct():
    values = (1, 2, 3, 4, 5)
    assert io.pct(values) == [100, 200, 3, 4, 500]
