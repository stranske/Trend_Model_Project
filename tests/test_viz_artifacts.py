from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pytest

from trend_analysis.viz.artifacts import extract_bundle_zip


def test_extract_bundle_zip_renames_duplicate_archive_entries(tmp_path: Path, caplog) -> None:
    bundle_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fan.html", "<first>")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("fan.html", "<second>")

    destination = tmp_path / "out" / "plots"
    with caplog.at_level(logging.WARNING):
        warnings = extract_bundle_zip(bundle_path, destination)

    assert (destination / "fan.html").read_text(encoding="utf-8") == "<first>"
    assert (destination / "fan-1.html").read_text(encoding="utf-8") == "<second>"
    assert any("Renamed extracted 'fan.html' to 'fan-1.html'" in message for message in warnings)
    assert any("Renamed extracted 'fan.html' to 'fan-1.html'" in message for message in caplog.messages)


def test_extract_bundle_zip_creates_destination_and_renames_on_existing_file_collision(
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
    assert (destination / "risk_return-1.json").read_text(encoding="utf-8") == '{"new": true}'
    assert any(
        "Renamed extracted 'risk_return.json' to 'risk_return-1.json'" in message
        for message in warnings
    )
    assert any(
        "Renamed extracted 'risk_return.json' to 'risk_return-1.json'" in message
        for message in caplog.messages
    )


def test_extract_bundle_zip_collision_names_are_deterministic_across_repeated_runs(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("risk_return.json", '{"first": true}')
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("risk_return.json", '{"second": true}')

    first_destination = tmp_path / "first"
    second_destination = tmp_path / "second"
    extract_bundle_zip(bundle_path, first_destination)
    extract_bundle_zip(bundle_path, second_destination)

    first_names = sorted(path.name for path in first_destination.glob("*.json"))
    second_names = sorted(path.name for path in second_destination.glob("*.json"))
    assert first_names == ["risk_return-1.json", "risk_return.json"]
    assert second_names == first_names
