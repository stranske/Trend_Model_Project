"""Regression contracts for test-local natural-language operation logs."""

from __future__ import annotations

import builtins
import importlib.util
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests import conftest as suite_conftest
from trend_analysis.llm.nl_logging import NLOperationLog, write_nl_log

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fixture_skips_llm_import_without_pydantic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_find_spec = importlib.util.find_spec
    real_import = builtins.__import__

    def _find_spec(name: str, *args, **kwargs):
        if name == "pydantic":
            return None
        return real_find_spec(name, *args, **kwargs)

    def _blocked_import(name: str, *args, **kwargs):
        if name == "trend_analysis.llm":
            raise AssertionError("the unavailable LLM package was imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(suite_conftest.importlib.util, "find_spec", _find_spec)
    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    assert suite_conftest._load_nl_logging_for_tests() is None


def test_cli_logging_does_not_write_repository_root(tmp_path: Path) -> None:
    root_log_dir = REPO_ROOT / ".trend_nl_logs"
    root_log_dir_existed = root_log_dir.exists()
    root_log_contents_before = (
        {
            path.relative_to(root_log_dir): path.read_bytes()
            for path in root_log_dir.rglob("*")
            if path.is_file()
        }
        if root_log_dir_existed
        else {}
    )

    write_nl_log(
        NLOperationLog(
            request_id="isolated-log",
            timestamp=datetime(2026, 8, 22, tzinfo=timezone.utc),
            operation="validate",
            input_hash="test-input",
            prompt_template="",
            prompt_variables={},
            model_output=None,
            parsed_patch=None,
            validation_result=None,
            error=None,
            duration_ms=0.0,
            model_name="test-model",
            temperature=0.0,
            token_usage=None,
        )
    )

    isolated_logs = list((tmp_path / ".trend_nl_logs").glob("nl_ops_*.jsonl"))
    assert len(isolated_logs) == 1
    assert root_log_dir.exists() is root_log_dir_existed
    if root_log_dir_existed:
        assert {
            path.relative_to(root_log_dir): path.read_bytes()
            for path in root_log_dir.rglob("*")
            if path.is_file()
        } == root_log_contents_before


def test_production_default_log_path_remains_repository_relative(
    tmp_path: Path,
) -> None:
    probe = (
        "from datetime import date\n"
        "from trend_analysis.llm.nl_logging import get_nl_log_path\n"
        "print(get_nl_log_path(log_date=date(2026, 8, 22)))\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ".trend_nl_logs/nl_ops_2026-08-22.jsonl"
