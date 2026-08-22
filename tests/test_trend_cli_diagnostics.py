from pathlib import Path

import pytest

from trend.commands import report_export as owned


def test_init_perf_logger_reports_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TREND_DISABLE_PERF_LOGS", "1")

    result = owned._init_perf_logger()

    assert result.value is None
    assert result.diagnostic is not None
    assert result.diagnostic.reason_code == "PERF_LOG_DISABLED"


def test_turnover_csv_diagnostic_for_missing_payload(tmp_path: Path):
    result = owned._maybe_write_turnover_csv(tmp_path, {})

    assert result.value is None
    assert result.diagnostic is not None
    assert result.diagnostic.reason_code == "NO_TURNOVER_EXPORT"


def test_turnover_ledger_diagnostic_for_missing_payload():
    result = owned._persist_turnover_ledger("rid", {})

    assert result.value is None
    assert result.diagnostic is not None
    assert result.diagnostic.reason_code == "NO_TURNOVER_LEDGER"
