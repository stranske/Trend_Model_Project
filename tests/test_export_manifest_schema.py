import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.export.bundle import export_bundle

try:
    import jsonschema
except ImportError:  # pragma: no cover - environment without dev deps
    jsonschema = None


def test_run_meta_conforms_to_schema(tmp_path):
    if jsonschema is None:
        pytest.skip("jsonschema not installed")
    # Prepare run
    returns = pd.Series(
        [0.01, -0.02], index=pd.date_range("2021-01", periods=2, freq="ME")
    )

    class Run:
        portfolio = returns
        config = {"foo": 1}
        seed = 7
        input_path = None

    out = tmp_path / "bundle.zip"

    # Create bundle
    export_bundle(Run, out)

    # Load manifest and schema
    with zipfile.ZipFile(out) as z:
        data = json.loads(z.read("run_meta.json").decode("utf-8"))
    schema_path = (
        Path(__file__).parents[1]
        / "src"
        / "trend_analysis"
        / "export"
        / "manifest_schema.json"
    )
    schema = json.loads(schema_path.read_text())

    # Validate
    jsonschema.validate(instance=data, schema=schema)

    # Spot-check outputs presence and a couple of keys
    assert "outputs" in data and isinstance(data["outputs"], dict)
    for k, v in list(data["outputs"].items())[:2]:
        assert isinstance(k, str) and isinstance(v, str)
        assert len(v) == 64
        int(v, 16)


def test_run_meta_missing_required_fails(tmp_path):
    if jsonschema is None:
        pytest.skip("jsonschema not installed")
    # Build a minimal, invalid manifest (missing run_id)
    invalid = {
        "schema_version": "1.0",
        "config": {},
        "config_sha256": "0" * 64,
        "environment": {"python": "3.11"},
        "git_hash": "deadbeef",
        "receipt": {"created": "2020-01-01T00:00:00Z"},
    }
    schema_path = (
        Path(__file__).parents[1]
        / "src"
        / "trend_analysis"
        / "export"
        / "manifest_schema.json"
    )
    schema = json.loads(schema_path.read_text())

    with pytest.raises(Exception):
        jsonschema.validate(instance=invalid, schema=schema)
