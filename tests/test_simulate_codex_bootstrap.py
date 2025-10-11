from tools.simulate_codex_bootstrap import ReadyIssues, parse_issue_numbers


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
