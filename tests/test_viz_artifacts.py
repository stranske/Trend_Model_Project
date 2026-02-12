from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pytest

from trend_analysis.viz.artifacts import extract_bundle_zip


def test_extract_bundle_zip_skips_duplicate_archive_entries(
    tmp_path: Path, caplog
) -> None:
    bundle_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fan.html", "<first>")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("fan.html", "<second>")

    destination = tmp_path / "out" / "plots"
    with caplog.at_level(logging.WARNING):
        warnings = extract_bundle_zip(bundle_path, destination)

    assert (destination / "fan.html").read_text(encoding="utf-8") == "<first>"
    assert any("duplicate entry 'fan.html'" in message for message in warnings)
    assert any("duplicate entry 'fan.html'" in message for message in caplog.messages)


def test_extract_bundle_zip_creates_destination_and_does_not_overwrite(
    tmp_path: Path, caplog
) -> None:
    bundle_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("risk_return.json", '{"new": true}')

    destination = tmp_path / "nested" / "plots"
    existing_file = destination / "risk_return.json"
    existing_file.parent.mkdir(parents=True, exist_ok=True)
    existing_file.write_text('{"existing": true}', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        warnings = extract_bundle_zip(bundle_path, destination)

    assert destination.is_dir()
    assert existing_file.read_text(encoding="utf-8") == '{"existing": true}'
    assert any("destination file already exists" in message for message in warnings)
    assert any("destination file already exists" in message for message in caplog.messages)
