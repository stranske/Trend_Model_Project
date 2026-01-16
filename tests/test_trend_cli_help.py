from __future__ import annotations

import re

import pytest

from trend import cli


def test_trend_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "usage:" in captured.out
    assert "trend" in captured.out


def test_trend_cli_nl_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["nl", "--help"])
    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "usage:" in captured.out
    assert "nl" in captured.out
    assert "--in" in captured.out
    assert re.search(r"Example:\s+--in\s+config/base\.yml", captured.out)
    assert re.search(r"Example:\s+--out\s+config/updated\.yml", captured.out)
    assert re.search(r'Example:\s+trend\s+nl\s+"Lower\s+max\s+weight"\s+--diff', captured.out)
    assert re.search(r'Example:\s+trend\s+nl\s+"Lower\s+max\s+weight"\s+--dry-run', captured.out)
    assert re.search(r'Example:\s+trend\s+nl\s+"Add\s+CSV\s+path"\s+--run', captured.out)
    assert "--no-confirm" in captured.out
    assert re.search(
        r'Example:\s+trend\s+nl\s+"Remove\s+constraints"\s+--no-\s*confirm',
        captured.out,
    )
    assert "--provider" in captured.out
    assert re.search(r"Example:\s+--provider\s+openai", captured.out)
    assert "--model" in captured.out
    assert re.search(r"Example:\s+--model\s+gpt-4o-mini", captured.out)
    assert "--temperature" in captured.out
    assert re.search(r"Example:\s+--temperature\s+0\.2", captured.out)
    assert re.search(
        r'Example:\s+trend\s+nl\s+"Lower\s+max\s+weight"\s+--explain\s+--diff',
        captured.out,
    )
    assert "Example:" in captured.out
