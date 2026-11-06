"""Regression test for #6188: ``xdist_group`` must keep tests on one worker.

Under pytest 9.2 with pytest-xdist 3.8.0, ``--dist=loadgroup`` silently
ignored ``xdist_group`` and split ``tests/golden/test_demo.py`` across
workers, where one test wiped ``demo/exports`` while the other was reading it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _xdist_loadgroup_compat

pytest.importorskip("xdist")

ROOT = Path(__file__).resolve().parents[1]

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
