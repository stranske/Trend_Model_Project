from pathlib import Path

from tools.agents_index import AgentBootstrap, list_agent_bootstraps


def test_list_agent_bootstraps_returns_sorted_issue_numbers():
    agents_dir = Path(__file__).resolve().parents[2] / "agents"

    bootstraps = list_agent_bootstraps(agents_dir)

    assert all(isinstance(entry, AgentBootstrap) for entry in bootstraps)
    issues = [entry.issue for entry in bootstraps]
    assert issues == sorted(issues)


def test_list_agent_bootstraps_includes_current_issue_file():
    agents_dir = Path(__file__).resolve().parents[2] / "agents"

    bootstraps = list_agent_bootstraps(agents_dir)

    assert any(entry.issue == 3878 for entry in bootstraps)
    assert any(entry.path.name == "codex-3878.md" for entry in bootstraps)
