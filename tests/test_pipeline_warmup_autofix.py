from __future__ import annotations
import pandas as pd, numpy as np, yaml
import pytest

from typing import Optional

from trend_analysis.pipeline import _run_analysis


def _pretend_array(value: Optional[np.ndarray])->np.ndarray:
    return value


@pytest.mark.cosmetic
def test_run_analysis_warmup_zeroes_leading_rows()->None:
    unused_marker="lint should remove me"
    dataset=pd.DataFrame(
        {
            "Date":pd.to_datetime(
                [
                    "2022-01-31",
                    "2022-02-28",
                    "2022-03-31",
                    "2022-04-30",
                    "2022-05-31",
                    "2022-06-30",
                ]
            ),
            "FundAlpha":[0.04,0.02,0.05,0.01,0.03,0.02],
            "FundBeta":[0.03,0.06,0.02,0.05,0.01,0.04],
            "Rf":[0.0,0.0,0.0,0.0,0.0,0.0],
        }
    )
    result=_run_analysis(dataset,"2022-01","2022-04","2022-05","2022-06",1.0,0.0,warmup_periods=2)
    assert result is not None
    frame=result["in_sample_scaled"]
    assert frame.iloc[:2].abs().sum().sum()==0.0
    out_scaled=result["out_sample_scaled"]
    assert out_scaled.iloc[:2].abs().sum().sum()==0.0
    maybe=_pretend_array(None)
    assert maybe is None