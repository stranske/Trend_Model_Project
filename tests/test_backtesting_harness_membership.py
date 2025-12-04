import numpy as np
import pandas as pd
import pytest

from trend_analysis.backtesting import harness
from trend_analysis.universe import MembershipWindow


def test_normalise_membership_policy_defaults_and_validates() -> None:
    assert harness._normalise_membership_policy(None) == "raise"
    assert harness._normalise_membership_policy(" skip ") == "skip"
    with pytest.raises(ValueError, match="membership_policy must be 'raise' or 'skip'"):
        harness._normalise_membership_policy("invalid")


def test_apply_membership_mask_skips_missing_and_extra_columns(caplog: pytest.LogCaptureFixture) -> None:
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    data = pd.DataFrame({"A": [1.0, 2.0], "B": [1.0, np.nan]}, index=dates)
    membership = {
        "B": (MembershipWindow(pd.Timestamp("2020-01-01"), None),),
        "C": (MembershipWindow(pd.Timestamp("2020-01-01"), None),),
    }

    with caplog.at_level("WARNING"):
        masked, mask = harness._apply_membership_mask(data, membership, policy="skip")

    assert list(masked.columns) == ["B"]
    assert list(mask.columns) == ["B"]
    assert mask.loc[dates[0], "B"]
    assert not mask.loc[dates[1], "B"]
    assert "requires price columns" in caplog.records[0].message
    assert "missing from universe membership" in caplog.records[1].message


def test_apply_membership_mask_returns_original_without_membership() -> None:
    df = pd.DataFrame({"A": [1.0]}, index=pd.date_range("2020-01-01", periods=1))

    masked, mask = harness._apply_membership_mask(df, None, policy="raise")

    assert mask is None
    pd.testing.assert_frame_equal(masked, df)
