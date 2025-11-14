from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import workflow_smoke_tests as smoke


def test_quarantine_smoke_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    records_path_holder: dict[str, str] = {}

    def fake_load_records(path):
        records_path_holder["path"] = str(path)
        return ["record"], ["invalid"]

    def fake_evaluate(records, *, today, additional_invalid):
        assert records == ["record"]
        assert additional_invalid == ["invalid"]
        assert str(today) == "2049-01-01"
        return SimpleNamespace(ok=True)

    def fake_summary(report):
        assert report.ok is True
        return "summary-output"

    monkeypatch.setattr(smoke.validate_quarantine_ttl, "load_records", fake_load_records)
    monkeypatch.setattr(smoke.validate_quarantine_ttl, "evaluate_records", fake_evaluate)
    monkeypatch.setattr(smoke.validate_quarantine_ttl, "build_summary", fake_summary)

    smoke._quarantine_smoke()

    captured = capsys.readouterr()
    assert "summary-output" in captured.out
    assert records_path_holder["path"].endswith("quarantine.yml")


def test_quarantine_smoke_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        smoke.validate_quarantine_ttl,
        "load_records",
        lambda path: ([], []),
    )
    monkeypatch.setattr(
        smoke.validate_quarantine_ttl,
        "evaluate_records",
        lambda records, *, today, additional_invalid: SimpleNamespace(ok=False),
    )
    monkeypatch.setattr(
        smoke.validate_quarantine_ttl,
        "build_summary",
        lambda report: "summary",
    )

    with pytest.raises(RuntimeError):
        smoke._quarantine_smoke()


def test_main_delegates_to_quarantine_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_quarantine_smoke() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(smoke, "_quarantine_smoke", fake_quarantine_smoke)

    result = smoke.main()

    assert called is True
    assert result == 0


def test_module_entrypoint_runs_main(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(
            Path(smoke.__file__),
            run_name="__main__",
        )

    assert exc.value.code == 0
