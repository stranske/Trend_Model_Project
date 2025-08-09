from trend_portfolio_app.data_schema import load_and_validate_csv


def test_load_and_validate_csv(tmp_path):
    csv = tmp_path / "toy.csv"
    csv.write_text("Date,A,B\n2020-01-31,0.01,0.02\n2020-02-29,0.00,-0.01\n")
    df, meta = load_and_validate_csv(csv)
    assert set(df.columns) == {"A", "B"}
    assert len(df) == 2
