from __future__ import annotations

import ast
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
    assert "if ! CHANGED_FILES=$(" in script
    assert "if [[ $? -ne 0 ]]" not in script


def test_release_script_uses_portable_backup_suffix() -> None:
    script = Path("scripts/test-release.sh").read_text(encoding="utf-8")

    assert "sed -i.bak" in script
    assert "sed -i .bak" not in script


def test_literal_string_extraction_uses_constant_only() -> None:
    node = ast.parse('"alpha"', mode="eval").body

    assert effectiveness._extract_literal_str(node) == "alpha"
    source = Path("scripts/evaluate_settings_effectiveness.py").read_text(encoding="utf-8")
    assert "ast.Str" not in source
