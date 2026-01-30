from pathlib import Path


def test_issue_ledger_files_are_absent() -> None:
    ledger_files = list(Path(".").glob(".agents/issue-*-ledger.yml"))
    assert not ledger_files
