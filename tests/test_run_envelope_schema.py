"""Validate the single replayable run-envelope JSON (``run_contract``).

These tests build a :class:`~trend_analysis.api.RunResult` via
:func:`run_simulation` and project it into ``run_envelope.json`` through
:func:`trend_analysis.export.run_envelope.to_run_envelope`, then assert the
result conforms to ``run_envelope_schema.json`` and that cost/latency and
structured warnings are populated. They fail on ``main`` today because no such
module/schema/function exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest

from trend_analysis import api
from trend_analysis.config import Config, load_config
from trend_analysis.export.run_envelope import to_run_envelope, write_run_envelope
from trend_analysis.util.hash import sha256_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_CONFIG = REPO_ROOT / "config" / "demo.yml"
DEMO_RETURNS = REPO_ROOT / "demo" / "demo_returns.csv"

SCHEMA_PATH = (
    REPO_ROOT / "src" / "trend_analysis" / "export" / "run_envelope_schema.json"
)


def _schema() -> dict:
    assert SCHEMA_PATH.exists(), f"Schema file missing: {SCHEMA_PATH}"
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(tmp_path: Path, *, config) -> Path:
    """Write a minimal manifest.json mirroring write_run_artifacts' shape."""

    run_dir = tmp_path / "runs" / "20200101_000000_demoabcd"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "trend.run_artifacts/1",
        "run_id": "demoabcd",
        "config_sha256": sha256_config(config),
        # 64-hex placeholder input hash so the envelope run_id is content-addressed.
        "input_sha256": "a" * 64,
        "git_hash": "abc1234",
        "artifacts": [
            {"name": "summary.csv", "sha256": "b" * 64, "size": 10},
            {"name": "report.html", "sha256": "c" * 64, "size": 20},
        ],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _make_ill_conditioned_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=8, freq="ME")
    base = np.array([0.01, 0.011, 0.009, 0.012, 0.0105, 0.0115, 0.0095, 0.0108])
    jitter = np.array([1e-4, -1e-4, 5e-5, -5e-5, 8e-5, -8e-5, 6e-5, -6e-5])
    return pd.DataFrame(
        {"Date": dates, "RF": 0.0, "A": base, "B": base + jitter, "C": base * 0.2 + 0.001}
    )


def _make_robust_cfg() -> Config:
    return Config(
        version="1",
        data={
            "risk_free_column": "RF",
            "allow_risk_free_fallback": False,
            "date_column": "Date",
            "frequency": "M",
        },
        preprocessing={},
        vol_adjust={"target_vol": 1.0},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-06",
            "out_start": "2020-07",
            "out_end": "2020-08",
        },
        portfolio={
            "weighting_scheme": "robust_mv",
            "robustness": {
                "shrinkage": {"enabled": False},
                "condition_check": {
                    "enabled": True,
                    "threshold": 1.0,
                    "safe_mode": "risk_parity",
                    "diagonal_loading_factor": 1.0e-6,
                },
            },
        },
        metrics={},
        export={},
        run={},
    )


def test_run_envelope_conforms_to_schema(tmp_path: Path) -> None:
    """run_simulation on the demo fixtures -> run_envelope.json -> schema valid."""

    if not DEMO_CONFIG.exists() or not DEMO_RETURNS.exists():  # pragma: no cover
        pytest.skip("Demo fixtures missing")

    cfg = load_config(str(DEMO_CONFIG))
    returns = pd.read_csv(DEMO_RETURNS)
    result = api.run_simulation(cfg, returns)

    config_payload = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(vars(cfg))
    manifest_path = _write_manifest(tmp_path, config=config_payload)

    out_path = write_run_envelope(
        result,
        config=config_payload,
        manifest_path=manifest_path,
        actor="ci",
        intent="schema-conformance",
    )
    assert out_path.exists()
    envelope = json.loads(out_path.read_text(encoding="utf-8"))

    jsonschema.validate(envelope, _schema())

    # The envelope references (does not duplicate) the manifest.
    assert envelope["outputs"]["manifest"] == "manifest.json"
    assert "summary.csv" in envelope["outputs"]["artifacts"]
    assert envelope["inputs"]["config_sha256"] == sha256_config(config_payload)


def test_run_envelope_missing_required_fails(tmp_path: Path) -> None:
    """Removing a required field should trigger schema validation failure."""

    cfg = _make_robust_cfg()
    result = api.run_simulation(cfg, _make_ill_conditioned_df())
    config_payload = dict(vars(cfg)) if not hasattr(cfg, "model_dump") else cfg.model_dump()
    manifest_path = _write_manifest(tmp_path, config=config_payload)
    envelope = to_run_envelope(result, config=config_payload, manifest_path=manifest_path)

    envelope.pop("run_id")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(envelope, _schema())


def test_run_envelope_cost_latency_and_fallback_warning(tmp_path: Path) -> None:
    """wall_ms is a positive float and the weight-engine fallback is captured."""

    cfg = _make_robust_cfg()
    result = api.run_simulation(cfg, _make_ill_conditioned_df())

    config_payload = dict(vars(cfg)) if not hasattr(cfg, "model_dump") else cfg.model_dump()
    manifest_path = _write_manifest(tmp_path, config=config_payload)
    envelope = to_run_envelope(result, config=config_payload, manifest_path=manifest_path)

    wall_ms = envelope["cost_latency"]["wall_ms"]
    assert isinstance(wall_ms, float)
    assert wall_ms > 0.0

    codes = {w.get("code") for w in envelope["warnings"]}
    assert "weight_engine_fallback" in codes
