"""Integration-style coverage for the trend nl command."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from trend import cli as trend_cli
from trend_analysis.config import DEFAULTS, ConfigPatch, PatchOperation, diff_configs


class _DummyChain:
    def __init__(self, patch: ConfigPatch) -> None:
        self._patch = patch

    def run(self, **_kwargs: object) -> ConfigPatch:
        return self._patch


def test_nl_diff_outputs_expected_patch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        "version: 1\nportfolio:\n  constraints:\n    max_weight: 0.2\n",
        encoding="utf-8",
    )
    patch = ConfigPatch(
        operations=[
            PatchOperation(
                op="set",
                path="portfolio.constraints.max_weight",
                value=0.1,
            )
        ],
        summary="Adjust max weight",
    )
    monkeypatch.setattr(trend_cli, "_build_nl_chain", lambda *_a, **_k: _DummyChain(patch))

    exit_code = trend_cli.main(["nl", "Lower max weight", "--in", str(cfg_path), "--diff"])

    output = capsys.readouterr().out
    expected_diff = diff_configs(
        {"version": 1, "portfolio": {"constraints": {"max_weight": 0.2}}},
        {"version": 1, "portfolio": {"constraints": {"max_weight": 0.1}}},
    )
    assert exit_code == 0
    assert output == expected_diff
    assert "--- before" in output
    assert "-    max_weight: 0.2" in output
    assert "+    max_weight: 0.1" in output
    assert cfg_path.read_text(encoding="utf-8") == (
        "version: 1\nportfolio:\n  constraints:\n    max_weight: 0.2\n"
    )


def test_nl_run_blocks_invalid_config_via_schema_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "invalid.yml"
    patch = ConfigPatch(
        operations=[
            PatchOperation(op="set", path="version", value=""),
        ],
        summary="Invalidate version",
    )
    monkeypatch.setattr(trend_cli, "_build_nl_chain", lambda *_a, **_k: _DummyChain(patch))
    monkeypatch.setattr(
        trend_cli,
        "_run_pipeline",
        lambda *_a, **_k: pytest.fail("Pipeline should not run for invalid config"),
    )

    exit_code = trend_cli.main(
        [
            "nl",
            "Invalidate version",
            "--in",
            str(DEFAULTS),
            "--out",
            str(output_path),
            "--run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Config validation failed" in captured.err
    assert not output_path.exists()


def test_nl_requires_confirmation_for_risky_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "confirmed.yml"
    patch = ConfigPatch(
        operations=[PatchOperation(op="remove", path="portfolio.constraints")],
        summary="Remove constraints",
    )
    called: dict[str, str] = {}

    def _fake_input(prompt: str = "") -> str:
        called["prompt"] = prompt
        return "n"

    monkeypatch.setattr(trend_cli, "_build_nl_chain", lambda *_a, **_k: _DummyChain(patch))
    monkeypatch.setattr(trend_cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", _fake_input)

    exit_code = trend_cli.main(
        ["nl", "Remove constraints", "--in", str(DEFAULTS), "--out", str(output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Update cancelled by user." in captured.err
    assert called
    assert not output_path.exists()
