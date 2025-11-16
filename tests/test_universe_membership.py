from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.universe import (
    MembershipWindow,
    apply_membership_windows,
    load_universe_membership,
)


def test_load_universe_membership_requires_effective_date(tmp_path: Path) -> None:
    csv = tmp_path / "membership.csv"
    csv.write_text("fund,effective_date,end_date\nFundA,,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="effective_date"):
        load_universe_membership(csv)


def test_apply_membership_masks_adds_and_drops() -> None:
    idx = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    frame = pd.DataFrame(
        {
            "FundA": [0.1, 0.2, 0.3],
            "FundB": [0.4, 0.5, 0.6],
            "FundC": [0.7, 0.8, 0.9],
        },
        index=idx,
    )
    membership = {
        "FundA": (MembershipWindow(pd.Timestamp("2020-01-31"), None),),
        # FundB enters after the first month
        "FundB": (MembershipWindow(pd.Timestamp("2020-02-29"), None),),
        # FundC exits after February
        "FundC": (
            MembershipWindow(pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")),
        ),
    }

    masked = apply_membership_windows(frame, membership)

    assert pd.isna(masked.loc[pd.Timestamp("2020-01-31"), "FundB"])
    assert not pd.isna(masked.loc[pd.Timestamp("2020-02-29"), "FundB"])
    assert pd.isna(masked.loc[pd.Timestamp("2020-03-31"), "FundC"])
    assert masked.loc[pd.Timestamp("2020-02-29"), "FundC"] == pytest.approx(0.8)
