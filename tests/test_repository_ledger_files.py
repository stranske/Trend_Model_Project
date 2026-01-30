from pathlib import Path


def test_issue_4582_ledger_file_is_absent() -> None:
    assert not Path(".agents/issue-4582-ledger.yml").exists()
