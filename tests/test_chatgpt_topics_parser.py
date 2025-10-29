import json
import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github/scripts/parse_chatgpt_topics.py"
TOPICS_PATH = pathlib.Path("topics.json")


def run_parser(text: str, env: dict | None = None) -> tuple[int, str, str, list[dict]]:
    """Helper to invoke the parser script in a subprocess for exit code
    semantics."""
    tmp = pathlib.Path("input.txt")
    tmp.write_text(text, encoding="utf-8")
    TOPICS_PATH.unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    return proc.returncode, proc.stdout, proc.stderr, read_topics()


def read_topics() -> list[dict]:
    if not TOPICS_PATH.exists():
        return []
    return json.loads(TOPICS_PATH.read_text(encoding="utf-8"))


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


def test_pipeline_handles_repository_issues_file(tmp_path: pathlib.Path) -> None:
    """End-to-end check that Issues.txt flows through the workflow helpers."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    issues_source = repo_root / "Issues.txt"
    assert issues_source.exists(), "Issues.txt must exist for pipeline test"

    workdir = tmp_path
    passthrough_source = workdir / "Issues.txt"
    passthrough_source.write_text(
        issues_source.read_text(encoding="utf-8"), encoding="utf-8"
    )

    decode_script = repo_root / ".github/scripts/decode_raw_input.py"
    decode_proc = subprocess.run(
        [
            sys.executable,
            str(decode_script),
            "--passthrough",
            "--in",
            str(passthrough_source),
            "--source",
            "repo_file",
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    assert decode_proc.returncode == 0, decode_proc.stderr

    parser_proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    assert parser_proc.returncode == 0, parser_proc.stderr

    topics_path = workdir / "topics.json"
    assert topics_path.exists(), "Parser must emit topics.json"
    topics = json.loads(topics_path.read_text(encoding="utf-8"))

    assert len(topics) >= 2, "Issues.txt should describe multiple topics"
    first = topics[0]
    assert "agent:codex" in first["labels"]
    assert "agents-70-orchestrator.yml" in first["sections"]["tasks"]
    assert "checkout" in first["sections"]["acceptance_criteria"].lower()
    assert "@{agent}" in first["sections"]["implementation_notes"]

    # Ensure every topic captures acceptance criteria to satisfy automation checks.
    assert all(topic["sections"]["acceptance_criteria"].strip() for topic in topics)
