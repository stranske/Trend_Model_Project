"""Tests for the documented performance-threshold configuration contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture()
def compare_perf_module():
    script = Path(__file__).parents[2] / "scripts" / "compare_perf.py"
    spec = importlib.util.spec_from_file_location("compare_perf_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_threshold_reads_named_dotenv_key(
    compare_perf_module, monkeypatch, tmp_path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=99\nTREND_PERF_THRESHOLD_PCT=12.5\n", encoding="utf-8")
    monkeypatch.delenv("TREND_PERF_THRESHOLD_PCT", raising=False)
    monkeypatch.setattr(compare_perf_module, "proj_path", lambda *_parts: env_file)
    assert compare_perf_module._default_threshold() == 12.5


def test_default_threshold_rejects_invalid_named_value(
    compare_perf_module, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("TREND_PERF_THRESHOLD_PCT", "not-a-number")
    assert compare_perf_module._default_threshold() == 15.0
    assert "TREND_PERF_THRESHOLD_PCT" in capsys.readouterr().err
