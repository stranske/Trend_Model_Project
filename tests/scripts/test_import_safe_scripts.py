"""Regression checks for script modules that must remain safe to inspect."""

from __future__ import annotations

import importlib.util
import runpy
import subprocess
from pathlib import Path


def test_run_multi_demo_import_has_no_side_effects(monkeypatch) -> None:
    """Importing the public demo entry point must not launch a demo or tests."""

    def fail_run(*_args, **_kwargs):
        raise AssertionError("import must not launch a subprocess")

    def fail_run_path(*_args, **_kwargs):
        raise AssertionError("import must not execute the demo module")

    monkeypatch.setattr(subprocess, "run", fail_run)
    monkeypatch.setattr(runpy, "run_path", fail_run_path)
    script = Path(__file__).parents[2] / "scripts" / "run_multi_demo.py"
    spec = importlib.util.spec_from_file_location("run_multi_demo_import_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
