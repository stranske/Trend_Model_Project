import sys
import pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trend_analysis import analyze


def make_df():
    return pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=3, freq='M'),
        'F1': [0.01, 0.02, 0.03],
        'RF': [0.0, 0.0, 0.0]
    })


def test_run_analysis_default_weights():
    df = make_df()
    res = analyze.run_analysis(
        df,
        selected=['F1'],
        w_vec=None,
        w_dict=None,
        rf_col='RF',
        in_start='2020-01',
        in_end='2020-02',
        out_start='2020-03',
        out_end='2020-03',
    )
    assert res['fund_weights']['F1'] == 1.0


def test_run_analysis_custom_weights():
    df = make_df()
    res = analyze.run_analysis(
        df,
        selected=['F1'],
        w_vec=None,
        w_dict={'F1': 0.5},
        rf_col='RF',
        in_start='2020-01',
        in_end='2020-02',
        out_start='2020-03',
        out_end='2020-03',
    )
    assert res['fund_weights']['F1'] == 0.5
