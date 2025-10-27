"""
Tests for Issue Bridge workflow trigger conditions.

Ensures that the Issue Bridge workflow correctly triggers when:
1. An issue is created with the agent:codex label
2. An issue is reopened with the agent:codex label
3. The agent:codex label is added to an existing issue
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class TestIssueBridgeTriggers(unittest.TestCase):
    """Validate Issue Bridge workflow trigger conditions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.workflows_dir = cls.project_root / ".github" / "workflows"
        cls.intake_workflow = cls.workflows_dir / "agents-63-issue-intake.yml"

    def _load_workflow(self) -> dict:
        """Load the intake workflow YAML."""
        self.assertTrue(
            self.intake_workflow.exists(),
            "agents-63-issue-intake.yml must exist",
        )
        return yaml.safe_load(self.intake_workflow.read_text(encoding="utf-8"))

    def test_workflow_listens_for_issue_events(self) -> None:
        """Ensure workflow has 'issues' trigger configured."""
        data = self._load_workflow()
        # PyYAML may parse 'on' as boolean True
        triggers = data.get("on", data.get(True, {}))
        self.assertIn(
            "issues",
            triggers,
            "Workflow must listen for issue events",
        )

    def test_workflow_triggers_on_opened_labeled_reopened(self) -> None:
        """Ensure workflow triggers on opened, labeled, and reopened events."""
        data = self._load_workflow()
        # PyYAML may parse 'on' as boolean True
        triggers = data.get("on", data.get(True, {}))
        issue_trigger = triggers.get("issues", {})
        types = set(issue_trigger.get("types", []))

        required_types = {"opened", "labeled", "reopened"}
        self.assertTrue(
            required_types.issubset(types),
            f"Workflow must trigger on {required_types}, got {types}",
        )

    def test_normalize_job_has_correct_condition(self) -> None:
        """Ensure normalize_inputs job has correct trigger condition."""
        data = self._load_workflow()
        jobs = data.get("jobs", {})
        normalize_job = jobs.get("normalize_inputs", {})

        self.assertIn(
            "if",
            normalize_job,
            "normalize_inputs job must have a condition",
        )

        condition = normalize_job["if"]
        self.assertIsInstance(
            condition,
            str,
            "Job condition must be a string",
        )

        # Remove whitespace and newlines for easier checking
        clean_condition = " ".join(condition.split())

        # Check that condition handles non-issue events
        self.assertIn(
            "github.event_name != 'issues'",
            clean_condition,
            "Condition must allow non-issue events (workflow_dispatch/workflow_call)",
        )

        # Check that condition checks the issue's labels array
        self.assertIn(
            "github.event.issue.labels",
            clean_condition,
            "Condition must check the issue's labels array",
        )

        # Check that condition checks for agent:codex label
        self.assertIn(
            "agent:codex",
            clean_condition,
            "Condition must check for agent:codex label",
        )

    def test_condition_handles_opened_with_agent_label(self) -> None:
        """Ensure condition checks issue labels for all issue events."""
        text = self.intake_workflow.read_text(encoding="utf-8")

        # The condition should check the issue's labels array for ALL issue events
        # This ensures that even if a different label triggers the workflow,
        # it will still run if agent:codex is present in the issue's labels
        self.assertIn(
            "github.event.issue.labels",
            text,
            "Condition must check issue labels for all issue events",
        )

    def test_condition_does_not_trigger_on_unlabeled(self) -> None:
        """Ensure condition does NOT trigger on 'unlabeled' events."""
        # Get the normalize_inputs job condition
        data = self._load_workflow()
        jobs = data.get("jobs", {})
        normalize_job = jobs.get("normalize_inputs", {})
        condition = normalize_job.get("if", "")

        # Clean up whitespace
        clean_condition = " ".join(str(condition).split())

        # The condition should NOT include unlabeled
        self.assertNotIn(
            "github.event.action == 'unlabeled'",
            clean_condition,
            "Condition must NOT trigger on 'unlabeled' events",
        )

    def test_condition_logic_structure(self) -> None:
        """Validate the logical structure of the condition."""
        data = self._load_workflow()
        jobs = data.get("jobs", {})
        normalize_job = jobs.get("normalize_inputs", {})
        condition = normalize_job.get("if", "")

        # Clean up for easier parsing
        clean_condition = " ".join(str(condition).split())

        # The simplified condition should have this structure:
        # (not issues) OR (issue has agent:codex label)
        #
        # This means ANY issue event will trigger if the issue has agent:codex,
        # regardless of which specific label triggered the event.
        # This handles the case where multiple labels are added simultaneously
        # and a different label (like agents:keepalive) triggers the workflow.

        # Check for OR operator
        self.assertIn(
            "||",
            clean_condition,
            "Condition should have an OR operator separating the two cases",
        )

    def test_chatgpt_sync_issues_format_works(self) -> None:
        """
        Ensure issues created by chatgpt_sync (which don't initially have
        agent:codex) can be processed when the label is manually added.
        """
        # chatgpt_sync creates issues WITHOUT agent labels, per PR #3090
        # Users then manually add agent:codex from the Issues tab
        # ANY labeled event will trigger the workflow, and it will proceed
        # if agent:codex is present in the issue's labels array

        data = self._load_workflow()
        jobs = data.get("jobs", {})
        normalize_job = jobs.get("normalize_inputs", {})
        condition = normalize_job.get("if", "")

        # Verify the condition checks the issue's labels array
        self.assertIn(
            "github.event.issue.labels",
            condition,
            "Condition must check issue.labels array for agent:codex presence",
        )


if __name__ == "__main__":
    unittest.main()
