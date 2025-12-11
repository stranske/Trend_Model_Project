from __future__ import annotations

import io
from pathlib import Path

import pytest

from streamlit_app.components.csv_validation import (
    CSVValidationError,
    validate_uploaded_csv,
)
from streamlit_app.components.data_schema import load_and_validate_csv
from streamlit_app.components.upload_guard import (
    UploadViolation,
    guard_and_buffer_upload,
)


class _FakeUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._buffer = io.BytesIO(data)

    def read(self) -> bytes:
        return self._buffer.read()

    def seek(self, offset: int) -> None:
        self._buffer.seek(offset)


def test_guard_and_buffer_upload_respects_env_limit(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TREND_UPLOAD_MAX_BYTES", "1024")
    upload = _FakeUpload("huge.csv", b"x" * 2048)

    with pytest.raises(UploadViolation):
        guard_and_buffer_upload(upload, upload_dir=tmp_path)


def test_validate_uploaded_csv_rejects_duplicate_columns() -> None:
    data = io.BytesIO(b"Date,Return,Return\n2020-01-31,0.1,0.2\n")
    data.name = "duped.csv"

    with pytest.raises(CSVValidationError) as err:
        validate_uploaded_csv(data, required_columns=("Date",), max_rows=10)

    assert "unique" in err.value.user_message
    assert any("Return" in issue for issue in err.value.issues)


def test_load_and_validate_csv_sanitizes_formula_headers(tmp_path: Path) -> None:
    csv_data = io.StringIO("=Date,=Alpha\n2020-01-31,0.1\n")
    csv_path = tmp_path / "formula.csv"
    csv_path.write_text(csv_data.getvalue())

    df, meta = load_and_validate_csv(csv_path.open())

    assert list(df.columns) == ["Alpha"]
    warnings = meta.get("validation", {}).get("warnings", [])
    joined = " ".join(warnings)
    assert "cleaned" in joined and "Alpha" in joined and "Date" in joined
