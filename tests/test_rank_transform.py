import numpy as np, pandas as pd
from trend_analysis.core.rank_selection import _apply_transform

def test_apply_transform_zscore():
    s = pd.Series([0.1, 0.2, 0.0, 0.3], index=list("ABCD"))
    z = _apply_transform(s, mode="zscore", window=4)
    expected = (0.3 - s.mean()) / s.std(ddof=0)
    assert np.isclose(z["D"], expected)
