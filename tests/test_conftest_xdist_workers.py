from __future__ import annotations

from tests import conftest


def test_xdist_auto_workers_forced_to_one_on_ci(monkeypatch) -> None:
    monkeypatch.setenv("CI", "true")
    assert conftest.pytest_xdist_auto_num_workers(None) == 1


def test_xdist_auto_workers_defer_to_default_outside_ci(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    assert conftest.pytest_xdist_auto_num_workers(None) is None
