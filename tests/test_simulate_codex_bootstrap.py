import pytest

from tools.simulate_codex_bootstrap import ReadyIssues, main, parse_issue_numbers


def test_ready_issues_outputs_round_trip():
    ready = ReadyIssues([2434, 2560])
    assert ready.issue_numbers_output == "2434,2560"
    assert ready.issue_numbers_json == "[2434, 2560]"
    assert ready.first_issue == "2434"
    assert ready.evaluate_from_json() == "2434"


def test_ready_issues_empty_queue():
    ready = ReadyIssues([])
    assert ready.issue_numbers_output == ""
    assert ready.issue_numbers_json == "[]"
    assert ready.first_issue == ""
    assert ready.evaluate_from_json() == ""


def test_parse_issue_numbers_filters_and_casts():
    numbers = parse_issue_numbers(["2434", "", " 2560 "])
    assert numbers == [2434, 2560]


def test_main_emits_outputs_for_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["2434", "2560"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Ready issues: [2434, 2560]" in captured.out
    assert "issue_numbers output: 2434,2560" in captured.out
    assert "fromJson(steps.ready.outputs.issue_numbers_json)[0] => 2434" in captured.out


def test_main_handles_empty_queue(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Ready issues: []" in captured.out
    assert "issue_numbers output: <empty>" in captured.out
    assert "fromJson(...) result: <empty queue>" in captured.out
