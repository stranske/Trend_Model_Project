"""Regression checks for script modules that must remain safe to inspect."""

from __future__ import annotations

import importlib.util
import runpy
import subprocess
from pathlib import Path

import pytest


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

    calls: list[tuple[str, str | None, dict[str, object] | None]] = []

    def record_run_path(path, *, run_name=None, init_globals=None, **_kwargs):
        calls.append((str(path), run_name, init_globals))
        return {}

    monkeypatch.setattr(runpy, "run_path", record_run_path)
    module.main()
    assert len(calls) == 1
    called_path, run_name, init_globals = calls[0]
    assert Path(called_path).name == "_run_multi_demo.py"
    assert run_name == "__main__"
    assert init_globals == {"RELEASE_VERIFY": False}


def test_run_multi_demo_release_verify_clears_sentinel_and_runs(monkeypatch, tmp_path) -> None:
    """Release verification must reach the real demo despite a stale sentinel."""

    script = Path(__file__).parents[2] / "scripts" / "run_multi_demo.py"
    spec = importlib.util.spec_from_file_location("run_multi_demo_release_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sentinel = tmp_path / ".fast_demo_mode"
    sentinel.write_text("fast\n", encoding="utf-8")
    monkeypatch.setattr(module, "FAST_SENTINEL", sentinel)
    calls: list[dict[str, object] | None] = []

    def record_run_path(_path, *, init_globals=None, **_kwargs):
        calls.append(init_globals)
        return {}

    monkeypatch.setattr(runpy, "run_path", record_run_path)
    module.main(["--release-verify"])

    assert not sentinel.exists()
    assert calls == [{"RELEASE_VERIFY": True}]


def test_inner_demo_sentinel_handler_refuses_release_skip(tmp_path) -> None:
    """The helper used by the inner runner must fail closed if a sentinel races back."""

    from trend._demo_guard import consume_fast_sentinel

    sentinel = tmp_path / ".fast_demo_mode"
    sentinel.write_text("fast\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="refused to skip the real demo"):
        consume_fast_sentinel(sentinel, release_verify=True)
    assert sentinel.exists()
