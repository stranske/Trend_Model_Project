from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github/scripts/decode_raw_input.py"


def run_decode(
    tmp_path: Path, *args: str, raw_payload: str | None = None
) -> SimpleNamespace:
    """Execute the decode script within the current Python process."""

    workdir = Path(tmp_path)
    if raw_payload is not None:
        (workdir / "raw_input.json").write_text(raw_payload, encoding="utf-8")

    original_cwd = os.getcwd()
    original_argv = sys.argv
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        os.chdir(workdir)
        sys.argv = [str(SCRIPT), *args]
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
        ):
            try:
                runpy.run_path(str(SCRIPT), run_name="__main__")
                return SimpleNamespace(
                    returncode=0,
                    stdout=stdout_buffer.getvalue(),
                    stderr=stderr_buffer.getvalue(),
                )
            except SystemExit as exc:  # pragma: no cover - deliberate CLI passthrough
                code = exc.code if isinstance(exc.code, int) else 1
                return SimpleNamespace(
                    returncode=code,
                    stdout=stdout_buffer.getvalue(),
                    stderr=stderr_buffer.getvalue(),
                )
    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv


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


def test_decode_forced_split_detects_inline_enumerators(tmp_path: Path) -> None:
    """Dense enumerator strings should trigger the forced split heuristic."""
    raw = "1) Alpha topic 2) Beta follow-up"
    result = run_decode(tmp_path, raw_payload=json.dumps(raw))
    assert result.returncode == 0, result.stderr

    output = (tmp_path / "input.txt").read_text(encoding="utf-8")
    # Expect each enumerator to start on its own line after processing.
    assert "\n2)" in output

    debug = load_decode_debug(tmp_path)
    assert "forced_split" in debug["applied"], debug["applied"]
    # Ensure enumerator metrics reflect the reconstructed structure.
    assert debug["rebuilt_enum_count"] >= debug["raw_enum_count"]


def test_decode_whitespace_normalization_reports_counts(tmp_path: Path) -> None:
    """Zero-width and non-breaking whitespace should be normalised and tracked."""
    raw = "\ufeff1) A\u00a0topic\u200b\tWhy Hidden\u200cnotes"
    result = run_decode(tmp_path, raw_payload=json.dumps(raw))
    assert result.returncode == 0, result.stderr

    output = (tmp_path / "input.txt").read_text(encoding="utf-8")
    assert "\ufeff" not in output
    assert "\u00a0" not in output
    assert "\u200b" not in output
    assert "\u200c" not in output

    debug = load_decode_debug(tmp_path)
    normalization = debug["whitespace_normalization"]
    assert normalization["bom"] == 1
    assert normalization["nbsp"] == 1
    assert normalization["zws"] >= 1
    assert normalization["tabs"] == 1


def test_decode_passthrough_without_input_returns(tmp_path: Path) -> None:
    """Passthrough mode with no --in argument should exit quietly."""
    result = run_decode(tmp_path, "--passthrough")
    assert result.returncode == 0
    assert not (tmp_path / "input.txt").exists()
    assert not (tmp_path / "decode_debug.json").exists()


def test_decode_passthrough_missing_file(tmp_path: Path) -> None:
    """Missing passthrough source files should not create outputs."""
    missing = tmp_path / "missing.txt"
    result = run_decode(tmp_path, "--passthrough", "--in", str(missing))
    assert result.returncode == 0
    assert not (tmp_path / "input.txt").exists()
    assert not (tmp_path / "decode_debug.json").exists()


def test_decode_requires_raw_input_file(tmp_path: Path) -> None:
    """Without raw_input.json the decoder should return immediately."""
    result = run_decode(tmp_path)
    assert result.returncode == 0
    assert not (tmp_path / "input.txt").exists()
    assert not (tmp_path / "decode_debug.json").exists()


def test_decode_empty_payload_skips_writing_input(tmp_path: Path) -> None:
    """Empty JSON payloads should not create input.txt but still emit diagnostics."""
    result = run_decode(tmp_path, raw_payload=json.dumps(""))
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "input.txt").exists()

    debug = load_decode_debug(tmp_path)
    assert debug["raw_len"] == 0
    assert debug["applied"] == []
