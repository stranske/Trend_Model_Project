from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trend_analysis.io import validators
from trend_analysis.io.market_data import MarketDataValidationError


def test_missing_file_raises_value_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match="File not found"):
        validators.load_and_validate_upload(missing)


def test_parser_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFile:
        name = "broken.csv"

    def boom(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise pd.errors.ParserError("bad")

    monkeypatch.setattr(validators.pd, "read_csv", boom)
    with pytest.raises(ValueError, match="Failed to parse file"):
        validators.load_and_validate_upload(DummyFile())


def test_schema_failure_raises_market_error() -> None:
    buffer = io.StringIO("Foo,Bar\n1,2\n")
    buffer.name = "invalid.csv"  # type: ignore[attr-defined]
    with pytest.raises(MarketDataValidationError):
        validators.load_and_validate_upload(buffer)


def test_permission_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFile:
        name = "restricted.csv"

    def deny(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise PermissionError

    monkeypatch.setattr(validators.pd, "read_csv", deny)
    with pytest.raises(ValueError, match="Permission denied"):
        validators.load_and_validate_upload(DummyFile())
