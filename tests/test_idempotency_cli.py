"""CLI idempotency: content-addressed working run_id + ``--skip-if-exists``.

These tests exercise the legacy ``trend_analysis.cli`` ``run`` command on the
bundled demo fixtures only (no external SaaS, no LLM, no network). They assert
the working ``run_id`` is content-addressed (stable for identical inputs) and
that ``--skip-if-exists`` reuses a prior completed run instead of recomputing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_CONFIG = REPO_ROOT / "config" / "demo.yml"
DEMO_RETURNS = REPO_ROOT / "demo" / "demo_returns.csv"


def _write_config(tmp_path: Path, export_dir: Path) -> Path:
    """Copy the demo config, pointing its export directory at *export_dir*."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(DEMO_CONFIG.read_text(encoding="utf-8"))
    cfg.setdefault("data", {})["csv_path"] = str(DEMO_RETURNS)
    cfg.setdefault("export", {})["directory"] = str(export_dir)
    cfg["export"].setdefault("formats", ["csv", "json"])
    out = tmp_path / "demo_local.yml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def _run_cli(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONHASHSEED"] = "0"
    cmd = [
        sys.executable,
        "-m",
        "trend_analysis.cli",
        "run",
        "-c",
        str(config),
        "-i",
        str(DEMO_RETURNS),
        "--seed",
        "777",
        *extra,
    ]
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _run_unified_cli(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONHASHSEED"] = "0"
    cmd = [
        sys.executable,
        "-m",
        "trend.cli",
        "run",
        "-c",
        str(config),
        "-i",
        str(DEMO_RETURNS),
        "--seed",
        "777",
        *extra,
    ]
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _run_dirs(export_dir: Path) -> list[Path]:
    runs = export_dir / "runs"
    if not runs.exists():
        return []
    return [p for p in runs.iterdir() if p.is_dir() and p.name != "index"]


def _manifest_run_id(run_dir: Path) -> str:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["run_id"]


@pytest.mark.skipif(not DEMO_RETURNS.exists(), reason="Demo returns fixture missing")
def test_skip_if_exists_reuses_run(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    config = _write_config(tmp_path, export_dir)

    first = _run_cli(config, "--skip-if-exists")
    assert first.returncode == 0, first.stderr
    run_dirs = _run_dirs(export_dir)
    assert len(run_dirs) == 1, f"expected one run dir, got {run_dirs}"
    manifest = run_dirs[0] / "manifest.json"
    first_run_id = _manifest_run_id(run_dirs[0])
    first_mtime = manifest.stat().st_mtime_ns

    second = _run_cli(config, "--skip-if-exists")
    assert second.returncode == 0, second.stderr
    # The second invocation must short-circuit: no new run dir, manifest
    # untouched, and the same content-addressed run_id reported.
    assert _run_dirs(export_dir) == run_dirs
    assert manifest.stat().st_mtime_ns == first_mtime
    assert "already-done" in second.stdout
    assert first_run_id in second.stdout


@pytest.mark.skipif(not DEMO_RETURNS.exists(), reason="Demo returns fixture missing")
def test_content_run_id_stable_without_flag(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    config = _write_config(tmp_path, export_dir)

    # Two identical runs (no --skip-if-exists): each recomputes and writes
    # artifacts, but the content-addressed working run_id must be identical,
    # mirroring the bundle assertion in test_determinism_cli.
    first = _run_cli(config)
    assert first.returncode == 0, first.stderr
    second = _run_cli(config)
    assert second.returncode == 0, second.stderr

    run_dirs = _run_dirs(export_dir)
    assert run_dirs, "expected at least one run directory"
    run_ids = {_manifest_run_id(d) for d in run_dirs}
    # On a random-UUID run_id (pre-change) the two runs would land in distinct
    # directories with distinct ids; content-addressing collapses them to one.
    assert len(run_ids) == 1, f"working run_id not stable across runs: {run_ids}"


@pytest.mark.skipif(not DEMO_RETURNS.exists(), reason="Demo returns fixture missing")
def test_unified_trend_run_skip_if_exists_reuses_own_manifest(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    config = _write_config(tmp_path, export_dir)

    first = _run_unified_cli(config, "--skip-if-exists")
    assert first.returncode == 0, first.stderr
    run_dirs = _run_dirs(export_dir)
    assert len(run_dirs) == 1, f"expected one run dir, got {run_dirs}"
    first_run_id = _manifest_run_id(run_dirs[0])
    manifest = run_dirs[0] / "manifest.json"
    first_mtime = manifest.stat().st_mtime_ns

    second = _run_unified_cli(config, "--skip-if-exists")
    assert second.returncode == 0, second.stderr
    assert _run_dirs(export_dir) == run_dirs
    assert manifest.stat().st_mtime_ns == first_mtime
    assert "already-done" in second.stdout
    assert first_run_id in second.stdout
