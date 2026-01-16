"""Tests for rendering the security policy document."""

from __future__ import annotations

from pathlib import Path

from scripts import render_security_policy


def test_render_security_policy_contains_required_sections() -> None:
    payload = render_security_policy.render_security_policy()
    assert payload.startswith("# Security Policy")
    assert "## Security Controls" in payload
    assert "## Potential Vulnerabilities" in payload
    assert "## Mitigation Strategies" in payload
    assert payload.endswith("\n")


def test_main_writes_output(tmp_path: Path) -> None:
    output_path = tmp_path / "SECURITY-DRAFT.md"
    exit_code = render_security_policy.main(["--output", str(output_path)])
    assert exit_code == 0
    contents = output_path.read_text(encoding="utf-8")
    assert contents.startswith("# Security Policy")
