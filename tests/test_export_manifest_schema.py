"""Validate the export manifest against the published JSON schema.

The tests confirm that the metadata embedded in ``run_meta.json`` adheres to
our bundled schema and that required fields remain mandatory. They also ensure
that the manifest advertises the schema location shipped with the package.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import jsonschema
import pandas as pd
import pytest

from trend_analysis.export.bundle import export_bundle


@dataclass
class DummyRun:
    """Minimal run object used for bundle export tests."""

    portfolio: pd.Series
    config: dict
    seed: int
    input_path: Path

    def summary(self) -> dict:
        return {"total_return": float(self.portfolio.sum())}


def _write_input(tmp_path: Path) -> Path:
    p = tmp_path / "input.csv"
    p.write_text("x\n1\n")
    return p


def _schema() -> dict:
    """Return the JSON schema bundled with the export package."""

    with resources.files("trend_analysis.export").joinpath(
        "manifest_schema.json"
    ).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_meta(tmp_path: Path) -> dict:
    """Create a bundle and return the parsed ``run_meta`` contents."""

    input_path = _write_input(tmp_path)
    # Use month-end frequency explicitly and convert to strings for the config
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    config = {"dates": dates.strftime("%Y-%m-%d").tolist()}

    run = DummyRun(
        portfolio=pd.Series(
            [0.01, -0.02], index=pd.date_range("2020-01", periods=2, freq="ME")
        ),
        config=config,
        seed=42,
        input_path=input_path,
    )

    out = tmp_path / "bundle.zip"
    export_bundle(run, out)

    with zipfile.ZipFile(out) as z:
        with z.open("run_meta.json") as f:
            meta = json.load(f)

    return meta


def test_run_meta_conforms_to_schema(tmp_path: Path) -> None:
    """The generated metadata should satisfy the schema."""

    meta = _build_meta(tmp_path)
    schema = _schema()
    jsonschema.Draft7Validator.check_schema(schema)
    jsonschema.validate(meta, schema)


def test_run_meta_declares_schema_reference(tmp_path: Path) -> None:
    """The manifest should carry the reference to the bundled schema."""

    meta = _build_meta(tmp_path)
    assert meta["$schema"] == "trend_analysis/export/manifest_schema.json"


def test_run_meta_missing_required_fails(tmp_path: Path) -> None:
    """Removing a required field should trigger validation failure."""

    meta = _build_meta(tmp_path)
    schema = _schema()
    meta.pop("run_id")

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(meta, schema)
