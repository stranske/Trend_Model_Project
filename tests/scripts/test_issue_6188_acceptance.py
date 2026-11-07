"""Keep issue #6188's passing commands discoverable by the Evidence runner."""

from pathlib import Path

from scripts.runner_lib.acceptance_gate import extract_commands

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence/issue-6188-closer-acceptance.md"


def test_evidence_extracts_each_passing_pytest_gate() -> None:
    gates = extract_commands(EVIDENCE.read_text(encoding="utf-8"))

    assert [(gate.criterion, gate.command) for gate in gates] == [
        (0, "python -m pytest tests/test_xdist_loadgroup_compat.py -q"),
        (1, "python -m pytest tests/golden/test_demo.py -n 2 --dist loadgroup -v --tb short"),
        (
            2,
            "python -m pytest tests/test_xdist_loadgroup_compat.py::"
            "test_loadgroup_keeps_xdist_group_on_one_worker -q",
        ),
    ]


def test_deliberate_failure_is_not_an_executable_gate() -> None:
    markdown = EVIDENCE.read_text(encoding="utf-8")
    # The extractor currently rejects environment assignments as command prefixes.
    # Remove that prefix too, so heading placement alone must exclude the probe.
    markdown = markdown.replace("TREND_DISABLE_XDIST_LOADGROUP_COMPAT=1 ", "")
    gates = extract_commands(markdown)

    regression_gates = [
        gate for gate in gates if "::test_loadgroup_keeps_xdist_group_on_one_worker" in gate.command
    ]
    assert len(gates) == 3
    assert len(regression_gates) == 1
    assert regression_gates[0].criterion == 2
