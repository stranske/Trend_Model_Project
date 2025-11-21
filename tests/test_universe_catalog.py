from __future__ import annotations

from pathlib import Path

import pandas as pd

from trend_analysis.universe_catalog import load_universe, load_universe_spec


def test_load_universe_spec_resolves_paths(tmp_path: Path) -> None:
    returns = tmp_path / "returns.csv"
    membership = tmp_path / "membership.csv"
    returns.write_text(
        "Date,A,B\n2020-01-31,1,3\n2020-02-29,2,4\n",
        encoding="utf-8",
    )
    membership.write_text(
        "fund,effective_date,end_date\nA,2020-01-31,2020-02-29\nB,2020-02-29,\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "core.yml"
    cfg.write_text(
        """
version: 1
key: sample
name: Sample universe
membership_csv: membership.csv
data_csv: returns.csv
members:
  - A
  - B
        """,
        encoding="utf-8",
    )

    spec = load_universe_spec("core", base_dir=tmp_path)
    assert spec.key == "sample"
    assert spec.data_path == returns
    assert spec.membership_path == membership

    mask, resolved = load_universe("core", base_dir=tmp_path)
    assert list(mask.columns) == ["A", "B"]
    assert bool(mask.loc[pd.Timestamp("2020-01-31"), "B"]) is False
    assert bool(mask.loc[pd.Timestamp("2020-02-29"), "B"]) is True
    assert resolved.date_column == "Date"
