from __future__ import annotations

import io
from typing import Any

import pandas as pd
import pytest

from pathlib import Path

from trend_analysis.io import validators


def test_load_and_validate_upload_missing_file() -> None:
    missing = Path("missing.csv")

    with pytest.raises(ValueError, match="File not found: 'missing.csv'"):
        validators.load_and_validate_upload(missing)


def test_load_and_validate_upload_parser_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFile:
        name = "broken.csv"

    def boom(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise pd.errors.ParserError("bad format")

    monkeypatch.setattr(validators.pd, "read_csv", boom)

    with pytest.raises(ValueError, match="Failed to parse file"):
        validators.load_and_validate_upload(DummyFile())


def test_load_and_validate_upload_schema_failure() -> None:
    buffer = io.StringIO("Foo,Bar\n1,2\n")
    buffer.name = "invalid.csv"  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="Schema validation failed"):
        validators.load_and_validate_upload(buffer)
