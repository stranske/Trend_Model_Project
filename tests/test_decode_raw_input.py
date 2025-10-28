from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github/scripts/decode_raw_input.py"


def run_decode(
    tmp_path: Path, *args: str, raw_payload: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Execute the decode script in a temporary working directory."""
    workdir = Path(tmp_path)
    if raw_payload is not None:
        (workdir / "raw_input.json").write_text(raw_payload, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=workdir,
        text=True,
        capture_output=True,
        check=False,
    )


def load_decode_debug(workdir: Path) -> dict:
    return json.loads((workdir / "decode_debug.json").read_text(encoding="utf-8"))


def test_decode_json_rebuilds_enumerators_and_sections(tmp_path: Path) -> None:
    """JSON payloads should be normalised and split for the parser."""
    raw = "1) Pipeline hardening Why Ensure reliability before release Tasks - Add smoke tests - Update docs"
    result = run_decode(tmp_path, raw_payload=json.dumps(raw))
    assert result.returncode == 0, result.stderr

    output = (tmp_path / "input.txt").read_text(encoding="utf-8")
    lines = output.splitlines()
    assert lines[0].startswith("1)")
    # Section headers should be on their own lines to mirror Issues.txt structure
    assert any(line.strip() == "Why" for line in lines)
    assert any(line.strip().startswith("-") for line in lines)

    debug = load_decode_debug(tmp_path)
    assert "raw_input" == debug["source_used"]
    applied = set(debug["applied"])
    assert "enumerators" in applied
    assert "sections" in applied


def test_decode_passthrough_normalises_newlines_and_tabs(tmp_path: Path) -> None:
    """Passthrough mode should preserve text while normalising whitespace."""
    src = tmp_path / "source.txt"
    src.write_text("Line 1\r\nLine\t2", encoding="utf-8")

    result = run_decode(
        tmp_path,
        "--passthrough",
        "--in",
        str(src),
        "--source",
        "repo_file",
    )
    assert result.returncode == 0, result.stderr

    output = (tmp_path / "input.txt").read_text(encoding="utf-8")
    assert output == "Line 1\nLine 2\n"

    debug = load_decode_debug(tmp_path)
    assert debug["source_used"] == "repo_file"
    assert debug["whitespace_normalization"]["tabs"] == 1
    assert debug["rebuilt_newlines"] >= debug["raw_newlines"]


def test_decode_invalid_json_falls_back_to_raw(tmp_path: Path) -> None:
    """Malformed JSON should fall back to the literal payload."""
    payload = "{not-json}"
    result = run_decode(tmp_path, raw_payload=payload)
    assert result.returncode == 0, result.stderr

    output = (tmp_path / "input.txt").read_text(encoding="utf-8")
    assert output == "{not-json}\n"

    debug = load_decode_debug(tmp_path)
    assert debug["source_used"] == "raw_input"
    # No heuristics should trigger when the payload already has newlines
    assert debug["applied"] == []
