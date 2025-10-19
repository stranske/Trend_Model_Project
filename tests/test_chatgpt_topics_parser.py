import json
import os
import pathlib
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github/scripts/parse_chatgpt_topics.py"


def run_parser(text: str, env: dict | None = None) -> tuple[int, str, str, list[dict]]:
    """Helper to invoke the parser script in a subprocess for exit code
    semantics."""
    tmp = pathlib.Path("input.txt")
    tmp.write_text(text, encoding="utf-8")
    pathlib.Path("topics.json").unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    return proc.returncode, proc.stdout, proc.stderr


def read_topics() -> list[dict]:
    return json.loads(pathlib.Path("topics.json").read_text(encoding="utf-8"))


def test_parser_success_basic():
    code, out, err, topics = run_parser("1. First topic\n\nWhy\nBecause\n")
    assert code == 0, (code, out, err)
    assert len(topics) == 1
    assert topics[0]["title"] == "First topic"
    assert topics[0]["sections"]["why"].startswith("Because")


def test_parser_multiple_with_labels_and_sections():
    sample = (
        "1) Alpha feature rollout\nLabels: feat:alpha, risk:low\nWhy\nNeed early feedback.\n"
        "\n2: Beta hardening\nTasks\n- Add tests\n- Improve logging\n"
    )
    code, out, err, topics = run_parser(sample)
    assert code == 0
    assert [t["title"] for t in topics] == ["Alpha feature rollout", "Beta hardening"]
    assert topics[0]["labels"] == ["feat:alpha", "risk:low"]
    assert "Add tests" in topics[1]["sections"]["tasks"]


def test_parser_no_numbered_topics_exit_code():
    code, out, err, _topics = run_parser("No numbering here")
    # Expect mapped exit code 3
    assert code == 3, (code, out, err)


def test_parser_fallback_single_topic():
    code, out, err, topics = run_parser(
        "Single blob topic without numbers", env={"ALLOW_SINGLE_TOPIC": "1"}
    )
    assert code == 0, (code, out, err)
    assert len(topics) == 1
    assert topics[0]["title"].startswith("Single blob")


def test_parser_empty_input_exit_code():
    code, out, err, _topics = run_parser("")
    # empty -> exit code 2
    assert code == 2, (code, out, err)


def test_title_cleanup_markdown_and_punctuation():
    code, out, err, topics = run_parser("1) **Title with markdown.**")
    assert code == 0
    assert topics[0]["title"] == "Title with markdown"


def test_alpha_enumeration_and_continuity():
    sample = "A) Alpha topic\n\nB) Beta topic\n\nD) Delta skipped C\n"
    code, out, err, topics = run_parser(sample)
    assert code == 0
    enums = [t["enumerator"] for t in topics]
    assert enums == ["A", "B", "D"]
    continuity = [t["continuity_break"] for t in topics]
    # A (first) ok, B ok, D should flag break
    assert continuity == [False, False, True]


def test_alphanumeric_enumeration():
    sample = "A1) Composite one\nA2) Composite two\nA4) Composite four\n"
    code, out, err, topics = run_parser(sample)
    assert code == 0
    assert [t["enumerator"] for t in topics] == ["A1", "A2", "A4"]
    # For alphanum we do not enforce continuity (so all False)
    assert all(not t["continuity_break"] for t in topics)


def test_lowercase_alpha_enumeration():
    sample = "a) first\nb) second\nd) fourth skipped c\n"
    code, out, err, topics = run_parser(sample)
    assert code == 0
    enums = [t["enumerator"] for t in topics]
    assert enums == ["a", "b", "d"]
    continuity = [t["continuity_break"] for t in topics]
    assert continuity == [False, False, True]
