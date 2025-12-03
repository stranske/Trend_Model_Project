from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.universe_catalog import (
    _clean_members,
    _resolve_date_column,
    _resolve_path,
    _resolve_universe_file,
    load_universe,
    load_universe_spec,
)


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


def test_resolve_path_errors(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    as_dir = base_dir / "nested"
    as_dir.mkdir()

    with pytest.raises(IsADirectoryError):
        _resolve_path(as_dir, base_dir=base_dir)

    with pytest.raises(FileNotFoundError):
        _resolve_path(base_dir / "missing.csv", base_dir=base_dir)


def test_resolve_path_prefers_parent_directory(tmp_path: Path) -> None:
    base_dir = tmp_path / "configs"
    base_dir.mkdir()
    parent_file = tmp_path / "shared.csv"
    parent_file.write_text("dummy", encoding="utf-8")

    resolved = _resolve_path("shared.csv", base_dir=base_dir)

    assert resolved == parent_file.resolve()


def test_resolve_universe_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _resolve_universe_file("unknown", base_dir=tmp_path)


def test_resolve_universe_file_bare_existing(tmp_path: Path) -> None:
    file_path = tmp_path / "custom.txt"
    file_path.write_text("dummy", encoding="utf-8")

    resolved = _resolve_universe_file(file_path, base_dir=tmp_path)
    assert resolved == file_path.resolve()


def test_resolve_universe_file_with_suffix(tmp_path: Path) -> None:
    cfg_path = tmp_path / "alt.yaml"
    cfg_path.write_text("version: 1\n", encoding="utf-8")

    resolved = _resolve_universe_file("alt", base_dir=tmp_path)

    assert resolved == cfg_path.resolve()


def test_resolve_universe_file_accepts_existing_absolute(tmp_path: Path) -> None:
    raw_path = tmp_path / "bare"
    raw_path.write_text("placeholder", encoding="utf-8")

    resolved = _resolve_universe_file(raw_path, base_dir=tmp_path)

    assert resolved == raw_path.resolve()


def test_load_universe_spec_validation(tmp_path: Path) -> None:
    cfg = tmp_path / "invalid.yml"
    cfg.write_text(
        """
version: 2
data_csv: data.csv
membership_csv: members.csv
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_universe_spec(cfg, base_dir=tmp_path)

    cfg.write_text(
        """
version: 1
name: Missing data
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_universe_spec(cfg, base_dir=tmp_path)

    cfg.write_text("- not a mapping\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_universe_spec(cfg, base_dir=tmp_path)


def test_clean_members_and_filter(tmp_path: Path) -> None:
    cfg = tmp_path / "core.yml"
    prices = tmp_path / "prices.csv"
    membership = tmp_path / "membership.csv"
    prices.write_text(
        "Date,A\n2020-01-31,1\n",
        encoding="utf-8",
    )
    membership.write_text("fund,effective_date\nA,2020-01-31\n", encoding="utf-8")
    cfg.write_text(
        """
version: 1
data_csv: prices.csv
membership_csv: membership.csv
members:
  - "  A  "
  - ""
        """,
        encoding="utf-8",
    )

    mask, spec = load_universe(cfg, base_dir=tmp_path)
    assert tuple(spec.members or ()) == ("A",)
    assert list(mask.columns) == ["A"]

    assert _clean_members(["", "   "]) is None


def test_load_universe_resolves_date_and_symbol_column(tmp_path: Path) -> None:
    cfg = tmp_path / "core.yml"
    prices = tmp_path / "prices.csv"
    membership = tmp_path / "membership.csv"
    prices.write_text("DATE,X\n2020-01-31,10\n", encoding="utf-8")
    membership.write_text(
        "symbol,effective_date,end_date\nX,2020-01-31,\n", encoding="utf-8"
    )
    cfg.write_text(
        """
version: 1
data_csv: prices.csv
membership_csv: membership.csv
date_column: date
        """,
        encoding="utf-8",
    )

    mask, spec = load_universe(cfg, base_dir=tmp_path)

    assert list(mask.columns) == ["X"]
    assert spec.date_column == "DATE"


def test_load_universe_orders_when_members_missing(tmp_path: Path) -> None:
    cfg = tmp_path / "core.yml"
    prices = tmp_path / "prices.csv"
    membership = tmp_path / "membership.csv"
    prices.write_text(
        "Date,B,A\n2020-01-31,2,1\n",
        encoding="utf-8",
    )
    membership.write_text(
        "fund,effective_date\nB,2020-01-31\nA,2020-01-31\n",
        encoding="utf-8",
    )
    cfg.write_text(
        """
version: 1
data_csv: prices.csv
membership_csv: membership.csv
        """,
        encoding="utf-8",
    )

    mask, spec = load_universe(cfg, base_dir=tmp_path)
    assert spec.members is None
    assert list(mask.columns) == ["A", "B"]


def test_load_universe_handles_empty_membership(tmp_path: Path) -> None:
    cfg = tmp_path / "core.yml"
    prices = tmp_path / "prices.csv"
    membership = tmp_path / "membership.csv"
    prices.write_text(
        "Date,A\n2020-01-31,1\n",
        encoding="utf-8",
    )
    membership.write_text("fund,effective_date\n", encoding="utf-8")
    cfg.write_text(
        """
version: 1
data_csv: prices.csv
membership_csv: membership.csv
        """,
        encoding="utf-8",
    )

    mask, spec = load_universe(cfg, base_dir=tmp_path)
    assert spec.members is None
    assert list(mask.columns) == []


def test_load_universe_empty_prices(tmp_path: Path) -> None:
    cfg = tmp_path / "core.yml"
    prices = tmp_path / "prices.csv"
    membership = tmp_path / "membership.csv"
    prices.write_text("Date,A\n", encoding="utf-8")
    membership.write_text("fund,effective_date\nA,2020-01-31\n", encoding="utf-8")
    cfg.write_text(
        """
version: 1
data_csv: prices.csv
membership_csv: membership.csv
        """,
        encoding="utf-8",
    )

    mask, spec = load_universe(cfg, base_dir=tmp_path)
    assert mask.empty
    assert spec.key == "core"


def test_load_universe_requires_fund_column(tmp_path: Path) -> None:
    cfg = tmp_path / "core.yml"
    prices = tmp_path / "prices.csv"
    membership = tmp_path / "membership.csv"
    prices.write_text(
        "Date,A\n2020-01-31,1\n",
        encoding="utf-8",
    )
    membership.write_text(
        "symbol_alias,effective_date\nA,2020-01-31\n", encoding="utf-8"
    )
    cfg.write_text(
        """
version: 1
data_csv: prices.csv
membership_csv: membership.csv
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_universe(cfg, base_dir=tmp_path)


def test_resolve_date_column_errors_for_missing_column() -> None:
    with pytest.raises(KeyError, match="Date column 'missing' was not found"):
        _resolve_date_column(pd.DataFrame({"Date": []}), "missing")
