from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

from scripts import evaluate_settings_effectiveness as effectiveness


def test_open_pr_script_emits_real_markdown_fences() -> None:
    script = Path("scripts/open_pr_from_issue.sh").read_text(encoding="utf-8")

    assert "echo '```markdown'" in script
    assert "echo '```'" in script
    assert r"echo '\`\`\`" not in script


def test_quick_check_pipeline_status_is_captured_directly() -> None:
    script = Path("scripts/quick_check.sh").read_text(encoding="utf-8")

    assert "set -o pipefail" in script
    assert "if ! DIFF_FILES=$(git diff --name-only HEAD~1 2>/dev/null); then" in script
    assert "| head -5" not in script


def test_quick_check_continues_when_git_diff_fails(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    git.chmod(0o755)

    result = subprocess.run(
        ["bash", "scripts/quick_check.sh"],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "VIRTUAL_ENV": "test"},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "git diff command failed, but continuing" in result.stdout


def test_release_script_uses_portable_backup_suffix() -> None:
    script = Path("scripts/test-release.sh").read_text(encoding="utf-8")

    assert "sed -i.bak" in script
    assert "sed -i .bak" not in script


def test_literal_string_extraction_uses_constant_only() -> None:
    node = ast.parse('"alpha"', mode="eval").body

    assert effectiveness._extract_literal_str(node) == "alpha"
    source = Path("scripts/evaluate_settings_effectiveness.py").read_text(encoding="utf-8")
    assert "isinstance(node, ast.Str)" not in source
