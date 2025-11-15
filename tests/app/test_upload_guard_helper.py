"""Tests for the upload guard helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from streamlit_app.components.upload_guard import (
    GuardedUpload,
    UploadViolation,
    guard_and_buffer_upload,
    hash_bytes,
)


class StubUploadedFile:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def read(self) -> bytes:
        return self._data

    def seek(self, _pos: int) -> None:
        return None


def test_guard_allows_csv(tmp_path: Path) -> None:
    uploaded = StubUploadedFile("returns.csv", b"Date,Asset\n2024-01-31,0.01\n")

    payload = guard_and_buffer_upload(uploaded, upload_dir=tmp_path)

    assert isinstance(payload, GuardedUpload)
    assert payload.stored_path.exists()
    assert payload.content_hash == hash_bytes(uploaded.read())
    assert payload.stored_path.read_bytes() == uploaded.read()


def test_guard_rejects_extension(tmp_path: Path) -> None:
    uploaded = StubUploadedFile("returns.txt", b"bad")

    with pytest.raises(UploadViolation) as exc:
        guard_and_buffer_upload(uploaded, upload_dir=tmp_path)

    assert "Unsupported file type" in str(exc.value)


def test_guard_rejects_large_file(tmp_path: Path) -> None:
    uploaded = StubUploadedFile("returns.csv", b"a" * 5)

    with pytest.raises(UploadViolation) as exc:
        guard_and_buffer_upload(uploaded, max_bytes=4, upload_dir=tmp_path)

    assert "File too large" in str(exc.value)
