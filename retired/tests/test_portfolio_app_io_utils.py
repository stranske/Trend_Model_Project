from __future__ import annotations

import datetime
import json
import os
import runpy
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from trend_portfolio_app import io_utils


class _DummyResults:
    def __init__(self, portfolio, event_log, summary):
        self.portfolio = portfolio
        self._event_log = event_log
        self._summary = summary

    def event_log_df(self):
        return self._event_log

    def summary(self) -> dict[str, object]:
        return self._summary


class _SimpleCSV:
    def __init__(self, rows: list[str]):
        self._rows = rows

    def to_csv(self, path: str, header=None):  # noqa: D401, ANN001
        with open(path, "w", encoding="utf-8") as fh:
            if header:
                fh.write(",".join(header) + "\n")
            fh.write("\n".join(self._rows))


class _BrokenCSV:
    def to_csv(self, *args, **kwargs):  # noqa: D401, ANN001, ARG002
        raise RuntimeError("boom")


def _read_zip_text(zip_path: str, name: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(name).decode("utf-8")


def test_export_bundle_creates_zip_and_registers_cleanup(tmp_path):
    io_utils._TEMP_FILES_TO_CLEANUP.clear()

    portfolio = _SimpleCSV(["0.1", "0.2"])
    event_log = _SimpleCSV(["event", "rebalance"])
    summary = {"alpha": 1.23}
    config = {"mode": "demo", "as_of": datetime.date(2024, 1, 31)}

    results = _DummyResults(portfolio, event_log, summary)

    bundle_path = io_utils.export_bundle(results, config)

    try:
        assert os.path.exists(bundle_path)
        assert bundle_path in io_utils._TEMP_FILES_TO_CLEANUP

        with zipfile.ZipFile(bundle_path) as zf:
            names = set(zf.namelist())
            assert {
                "portfolio_returns.csv",
                "event_log.csv",
                "summary.json",
                "config.json",
            } <= names

            summary_payload = json.loads(zf.read("summary.json").decode("utf-8"))
            assert summary_payload == summary

            config_payload = json.loads(zf.read("config.json").decode("utf-8"))
            assert config_payload == {"mode": "demo", "as_of": "2024-01-31"}
    finally:
        io_utils.cleanup_bundle_file(bundle_path)

    assert bundle_path not in io_utils._TEMP_FILES_TO_CLEANUP
    assert not os.path.exists(bundle_path)


def test_export_bundle_handles_export_failures(tmp_path):
    io_utils._TEMP_FILES_TO_CLEANUP.clear()

    event_log = _BrokenCSV()
    summary = {"beta": "ok"}
    results = _DummyResults(_BrokenCSV(), event_log, summary)

    bundle_path = io_utils.export_bundle(results, {"mode": "fault"})

    try:
        portfolio_csv = _read_zip_text(bundle_path, "portfolio_returns.csv")
        event_log_csv = _read_zip_text(bundle_path, "event_log.csv")

        assert portfolio_csv.strip() == "return"
        assert event_log_csv == ""
    finally:
        io_utils.cleanup_bundle_file(bundle_path)

    assert io_utils._TEMP_FILES_TO_CLEANUP == []


def test_export_bundle_cleans_up_partial_zip_on_failure(tmp_path, monkeypatch):
    io_utils._TEMP_FILES_TO_CLEANUP.clear()

    portfolio = _SimpleCSV(["0.05"])
    event_log = _SimpleCSV(["rebalance"])
    results = _DummyResults(portfolio, event_log, {"gamma": 3})

    partial_zip = tmp_path / "partial_bundle.zip"

    def fake_mkstemp(*args, **kwargs):  # noqa: D401, ANN001, ARG002
        fd = os.open(partial_zip, os.O_CREAT | os.O_RDWR)
        return fd, str(partial_zip)

    class BrokenZip:  # noqa: D401
        def __init__(self, *args, **kwargs):  # noqa: ANN001, ARG002
            raise RuntimeError("zip failed")

    monkeypatch.setattr(io_utils.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(io_utils.zipfile, "ZipFile", BrokenZip)

    with pytest.raises(RuntimeError):
        io_utils.export_bundle(results, {"mode": "broken"})

    assert not partial_zip.exists()
    assert io_utils._TEMP_FILES_TO_CLEANUP == []


def test_cleanup_temp_files_handles_missing_files(tmp_path):
    io_utils._TEMP_FILES_TO_CLEANUP.clear()
    ghost = tmp_path / "ghost.zip"
    io_utils._TEMP_FILES_TO_CLEANUP.append(str(ghost))

    io_utils._cleanup_temp_files()

    assert io_utils._TEMP_FILES_TO_CLEANUP == []


def test_cleanup_temp_files_handles_remove_error(tmp_path, monkeypatch):
    io_utils._TEMP_FILES_TO_CLEANUP.clear()
    stubborn = tmp_path / "stubborn.zip"
    stubborn.write_text("x", encoding="utf-8")
    io_utils._TEMP_FILES_TO_CLEANUP.append(str(stubborn))

    def boom(path: str) -> None:  # noqa: D401
        raise OSError("cannot remove")

    monkeypatch.setattr(io_utils.os, "remove", boom)

    io_utils._cleanup_temp_files()

    # Registry should still be cleared even though the file remains.
    assert io_utils._TEMP_FILES_TO_CLEANUP == []
    assert stubborn.exists()
    stubborn.unlink()


def test_cleanup_bundle_file_recovers_after_error(tmp_path, monkeypatch):
    io_utils._TEMP_FILES_TO_CLEANUP.clear()
    bundle = tmp_path / "bundle.zip"
    bundle.write_text("payload", encoding="utf-8")
    io_utils._TEMP_FILES_TO_CLEANUP.append(str(bundle))

    original_remove = io_utils.os.remove
    calls = {"count": 0}

    def flaky_remove(path: str) -> None:  # noqa: D401
        if path == str(bundle) and calls["count"] == 0:
            calls["count"] += 1
            raise OSError("temp failure")
        original_remove(path)

    monkeypatch.setattr(io_utils.os, "remove", flaky_remove)

    # First attempt should swallow the error and keep the file registered.
    io_utils.cleanup_bundle_file(str(bundle))
    assert str(bundle) in io_utils._TEMP_FILES_TO_CLEANUP
    assert bundle.exists()

    # Second attempt succeeds once the remover stops raising.
    io_utils.cleanup_bundle_file(str(bundle))
    assert str(bundle) not in io_utils._TEMP_FILES_TO_CLEANUP
    assert not bundle.exists()


def test_portfolio_app_main_invokes_streamlit(monkeypatch):
    """Running ``python -m trend_portfolio_app`` should invoke streamlit
    CLI."""

    calls: list[list[str]] = []

    def fake_main():  # noqa: D401
        calls.append(list(sys.argv))

    streamlit_mod = ModuleType("streamlit")
    streamlit_web = ModuleType("streamlit.web")
    streamlit_cli = ModuleType("streamlit.web.cli")
    streamlit_cli.main = fake_main
    streamlit_web.cli = streamlit_cli
    streamlit_mod.web = streamlit_web

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_mod)
    monkeypatch.setitem(sys.modules, "streamlit.web", streamlit_web)
    monkeypatch.setitem(sys.modules, "streamlit.web.cli", streamlit_cli)

    monkeypatch.setattr(sys, "argv", ["python"])
    monkeypatch.setattr(sys, "path", list(sys.path))

    runpy.run_module("trend_portfolio_app.__main__", run_name="__main__")

    assert calls
    argv = calls[-1]
    assert argv[0] == "streamlit"
    assert argv[1] == "run"
    assert argv[2].endswith("trend_portfolio_app/app.py")


def test_health_wrapper_runner_injects_src_path(monkeypatch):
    """Importing the runner should prepend the repo ``src`` directory to
    ``sys.path``."""

    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType

    module_name = "trend_portfolio_app.health_wrapper_runner"

    # Ensure a clean import and simulate the src path missing from sys.path.
    sys.modules.pop(module_name, None)

    src_path = Path(__file__).resolve().parents[1] / "src"
    scrubbed_path = [p for p in sys.path if str(src_path) not in p]
    monkeypatch.setattr(sys, "path", scrubbed_path, raising=False)

    # Provide lightweight stubs so the runner can import the health wrapper.
    package = ModuleType("trend_portfolio_app")
    package.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trend_portfolio_app", package)

    invoked: dict[str, bool] = {}

    def fake_main() -> None:  # noqa: D401
        invoked["called"] = True

    health_wrapper = ModuleType("trend_portfolio_app.health_wrapper")
    health_wrapper.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "trend_portfolio_app.health_wrapper", health_wrapper
    )

    module_path = src_path / "trend_portfolio_app" / "health_wrapper_runner.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader  # pragma: no branch - sanity check
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # The runner should have inserted the src directory at the front of sys.path.
    assert sys.path[0] == str(src_path)
    assert module.src == src_path

    # The imported ``main`` should be callable and proxy through to the fake implementation.
    module.main()
    assert invoked == {"called": True}

    # Clean up the temporary module entry so future imports see the real package.
    sys.modules.pop(module_name, None)


def test_health_wrapper_runner_skips_existing_src_path(monkeypatch):
    """If the repository ``src`` directory is already present, the runner
    should leave the order untouched instead of inserting a duplicate entry."""

    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType

    module_name = "trend_portfolio_app.health_wrapper_runner"

    # Reset any prior import side effects before reloading the runner.
    sys.modules.pop(module_name, None)

    src_path = Path(__file__).resolve().parents[1] / "src"
    preexisting = [str(src_path), "dummy-path"]
    monkeypatch.setattr(sys, "path", preexisting, raising=False)

    package = ModuleType("trend_portfolio_app")
    package.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trend_portfolio_app", package)

    invoked: dict[str, bool] = {}

    def fake_main() -> None:  # noqa: D401
        invoked["called"] = True

    health_wrapper = ModuleType("trend_portfolio_app.health_wrapper")
    health_wrapper.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "trend_portfolio_app.health_wrapper", health_wrapper
    )

    module_path = src_path / "trend_portfolio_app" / "health_wrapper_runner.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader  # pragma: no cover
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # ``sys.path`` should remain unchanged when the src directory is already present.
    assert sys.path == preexisting
    module.main()
    assert invoked == {"called": True}

    sys.modules.pop(module_name, None)


def test_portfolio_app_main_preserves_existing_src_path(monkeypatch):
    import importlib
    import sys
    from types import ModuleType

    module_name = "trend_portfolio_app.__main__"
    module = importlib.import_module(module_name)

    streamlit_mod = ModuleType("streamlit")
    streamlit_web = ModuleType("streamlit.web")
    streamlit_cli = ModuleType("streamlit.web.cli")
    streamlit_cli.main = lambda: None
    streamlit_web.cli = streamlit_cli
    streamlit_mod.web = streamlit_web

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_mod)
    monkeypatch.setitem(sys.modules, "streamlit.web", streamlit_web)
    monkeypatch.setitem(sys.modules, "streamlit.web.cli", streamlit_cli)

    src_path = str(Path(__file__).resolve().parents[1] / "src")
    unique_path = [src_path, *[p for p in sys.path if p != src_path]]
    monkeypatch.setattr(sys, "path", unique_path, raising=False)
    monkeypatch.setattr(sys, "argv", ["python"])

    module.main()

    assert sys.path.count(src_path) == 1


def test_export_bundle_handles_missing_zip_cleanup(tmp_path, monkeypatch):
    io_utils._TEMP_FILES_TO_CLEANUP.clear()

    portfolio = _SimpleCSV(["0.3"])
    event_log = _SimpleCSV(["ok"])
    results = _DummyResults(portfolio, event_log, {"delta": 2})

    missing_zip = tmp_path / "missing_bundle.zip"

    def fake_mkstemp(*args, **kwargs):  # noqa: ANN001, ARG002
        fd = os.open(missing_zip, os.O_CREAT | os.O_RDWR)
        os.close(fd)
        missing_zip.unlink()
        return os.open(__file__, os.O_RDONLY), str(missing_zip)

    class RaisingZip:
        def __init__(self, *args, **kwargs):  # noqa: ANN001, ARG002
            self.path = args[0]

        def __enter__(self):
            raise RuntimeError("zip failure")

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return None

    monkeypatch.setattr(io_utils.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(io_utils.zipfile, "ZipFile", RaisingZip)

    with pytest.raises(RuntimeError, match="zip failure"):
        io_utils.export_bundle(results, {"mode": "missing"})

    assert io_utils._TEMP_FILES_TO_CLEANUP == []


def test_cleanup_bundle_file_removes_missing_registered_path(tmp_path):
    io_utils._TEMP_FILES_TO_CLEANUP.clear()
    ghost = tmp_path / "ghost_bundle.zip"
    # Simulate a stale registry entry without a backing file.
    io_utils._TEMP_FILES_TO_CLEANUP.append(str(ghost))

    io_utils.cleanup_bundle_file(str(ghost))

    assert str(ghost) not in io_utils._TEMP_FILES_TO_CLEANUP


def test_cleanup_bundle_file_ignores_untracked_path():
    io_utils._TEMP_FILES_TO_CLEANUP.clear()

    io_utils.cleanup_bundle_file("/tmp/nonexistent_bundle.zip")

    assert io_utils._TEMP_FILES_TO_CLEANUP == []
