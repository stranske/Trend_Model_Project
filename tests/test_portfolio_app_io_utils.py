from __future__ import annotations

import json
import os
import runpy
import sys
import zipfile
from types import ModuleType

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

    results = _DummyResults(portfolio, event_log, summary)

    bundle_path = io_utils.export_bundle(results, {"mode": "demo"})

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
            assert config_payload == {"mode": "demo"}
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


def test_cleanup_temp_files_handles_missing_files(tmp_path):
    io_utils._TEMP_FILES_TO_CLEANUP.clear()
    ghost = tmp_path / "ghost.zip"
    io_utils._TEMP_FILES_TO_CLEANUP.append(str(ghost))

    io_utils._cleanup_temp_files()

    assert io_utils._TEMP_FILES_TO_CLEANUP == []


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
