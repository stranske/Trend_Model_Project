"""Regression test for #6188: ``xdist_group`` must keep tests on one worker.

Under pytest 9.2 with pytest-xdist 3.8.0, ``--dist=loadgroup`` silently
ignored ``xdist_group`` and split ``tests/golden/test_demo.py`` across
workers, where one test wiped ``demo/exports`` while the other was reading it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _xdist_loadgroup_compat

pytest.importorskip("xdist")

ROOT = Path(__file__).resolve().parents[1]

_PYTEST_MINOR = pytest.version_tuple[:2]

_GROUPED_TESTS = """
import os

import pytest


@pytest.mark.xdist_group(name="shared_state")
@pytest.mark.parametrize("n", range(8))
def test_grouped(n):
    out = os.path.join(os.environ["WORKER_LOG_DIR"], f"{n}.txt")
    with open(out, "w") as fh:
        fh.write(os.environ.get("PYTEST_XDIST_WORKER", "main"))
"""

_SERIAL_PAIR_TESTS = """
import json
import os
from pathlib import Path

import pytest


def _write_probe(request):
    base = Path(os.environ["SERIAL_PROBE_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    safe = request.node.name.replace("/", "_")
    out = base / f"{safe}.json"
    payload = {
        "worker": os.environ.get("PYTEST_XDIST_WORKER", "main"),
        "nodeid": request.node.nodeid,
    }
    out.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.serial
def test_serial_probe_a(request):
    _write_probe(request)


@pytest.mark.serial
def test_serial_probe_b(request):
    _write_probe(request)
"""

_LATE_SERIAL_CONFTEST = """
import pytest


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items):
    for it in items:
        if it.get_closest_marker("serial"):
            it.add_marker(pytest.mark.xdist_group("serial"))
"""


def test_conftest_registers_loadgroup_shim(request: pytest.FixtureRequest) -> None:
    assert request.config.pluginmanager.has_plugin(_xdist_loadgroup_compat.PLUGIN_NAME)


def test_loadgroup_keeps_xdist_group_on_one_worker(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "test_grouped.py").write_text(_GROUPED_TESTS)
    (project / "pytest.ini").write_text("[pytest]\n")
    worker_log = tmp_path / "workers"
    worker_log.mkdir()

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_XDIST_WORKER", None)
    env["WORKER_LOG_DIR"] = str(worker_log)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests._xdist_loadgroup_compat",
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "loadgroup",
            "-q",
        ],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    records = list(worker_log.glob("*.txt"))
    assert len(records) == 8
    workers = {p.read_text() for p in records}
    assert len(workers) == 1, f"xdist_group split across workers: {sorted(workers)}"


def _assert_serial_probe_one_worker_and_optional_suffix(probe_dir: Path) -> None:
    records = sorted(probe_dir.glob("*.json"))
    assert len(records) == 2
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in records]
    workers = {d["worker"] for d in payloads}
    assert len(workers) == 1, f"serial tests split across workers: {sorted(workers)}"
    # pytest 9.2+ structured ids make the loadgroup ``@group`` suffix easy to assert; 9.1 may differ.
    if _PYTEST_MINOR >= (9, 2):
        for d in payloads:
            nid = d["nodeid"]
            assert "@serial" in nid, f"missing @serial suffix in nodeid: {nid!r}"


def test_repo_conftest_serial_mark_loadgroup_one_worker(tmp_path: Path) -> None:
    """``tests/conftest.py`` assigns ``xdist_group(serial)`` before xdist; probe stays on one worker."""

    probe_dir = tmp_path / "serial_probe"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_XDIST_WORKER", None)
    env["SERIAL_PROBE_DIR"] = str(probe_dir)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ROOT / "tests" / "serial_xdist_probe.py"),
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "loadgroup",
            "-q",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_serial_probe_one_worker_and_optional_suffix(probe_dir)


@pytest.mark.skipif(
    _PYTEST_MINOR < (9, 2), reason="late ordering regression is specific to pytest >= 9.2"
)
def test_late_serial_group_assignment_splits_workers(tmp_path: Path) -> None:
    """If ``xdist_group`` is applied only at trylast, loadgroup can split serial tests on pytest 9.2+."""

    project = tmp_path / "iso"
    project.mkdir()
    (project / "test_serial_pair.py").write_text(_SERIAL_PAIR_TESTS)
    (project / "conftest.py").write_text(_LATE_SERIAL_CONFTEST)
    (project / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    serial: runs on one xdist worker\n",
        encoding="utf-8",
    )
    probe_dir = tmp_path / "serial_probe"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_XDIST_WORKER", None)
    env["SERIAL_PROBE_DIR"] = str(probe_dir)
    env["TREND_DISABLE_XDIST_LOADGROUP_COMPAT"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests._xdist_loadgroup_compat",
            "-p",
            "no:cacheprovider",
            "test_serial_pair.py",
            "-n",
            "2",
            "--dist",
            "loadgroup",
            "-q",
        ],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = sorted(probe_dir.glob("*.json"))
    assert len(records) == 2
    workers = {json.loads(p.read_text(encoding="utf-8"))["worker"] for p in records}
    assert len(workers) == 2, (
        "expected late serial-group assignment to split serial tests across workers on pytest 9.2+; "
        f"got workers={sorted(workers)}"
    )


@pytest.mark.skipif(
    _PYTEST_MINOR < (9, 2), reason="restoration regression is specific to pytest >= 9.2"
)
def test_restore_group_suffixes_required_under_pytest92_without_compat(tmp_path: Path) -> None:
    """Without restoration, grouped nodeids lack ``@serial`` and serial tests can split."""

    probe_dir = tmp_path / "serial_probe"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_XDIST_WORKER", None)
    env["SERIAL_PROBE_DIR"] = str(probe_dir)
    env["TREND_DISABLE_XDIST_LOADGROUP_COMPAT"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ROOT / "tests" / "serial_xdist_probe.py"),
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "loadgroup",
            "-q",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = sorted(probe_dir.glob("*.json"))
    assert len(records) == 2
    workers = {json.loads(p.read_text(encoding="utf-8"))["worker"] for p in records}
    assert len(workers) == 2, (
        "expected serial group to be ineffective without restoration on pytest 9.2+; "
        f"got single worker: {workers}"
    )


def test_collect_serial_probe_lists_serial_marker() -> None:
    """Verbose collection lists the ``serial`` marker (``@serial`` is added by xdist scheduling, not here)."""

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ROOT / "tests" / "serial_xdist_probe.py"),
            "--collect-only",
            "-v",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout + result.stderr
    assert "test_serial_probe_a" in out and "test_serial_probe_b" in out
    assert "serial" in out


def test_golden_demo_loadgroup_two_workers() -> None:
    """Repository golden demo must survive ``-n 2 --dist loadgroup`` (shared export paths)."""

    env = dict(os.environ)
    # Golden tests invoke ``python`` (not ``sys.executable``); ensure that resolves to this interpreter.
    py_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = os.pathsep.join([py_dir, env.get("PATH", "")])
    path_parts = [str(ROOT / "src"), str(ROOT)]
    if extra := env.get("PYTHONPATH"):
        path_parts.append(extra)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)
    env["PYTHONSAFEPATH"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_XDIST_WORKER", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ROOT / "tests" / "golden" / "test_demo.py"),
            "-n",
            "2",
            "--dist",
            "loadgroup",
            "-v",
            "--tb",
            "short",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
