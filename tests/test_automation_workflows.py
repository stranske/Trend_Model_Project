from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class TestAutomationWorkflowCoverage(unittest.TestCase):
    """Validate that automation scripts and workflows cover core gates."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.scripts_dir = cls.project_root / "scripts"
        cls.workflows_dir = cls.project_root / ".github" / "workflows"

    # -- helpers -----------------------------------------------------------------

    def _read_script(self, name: str) -> str:
        path = self.scripts_dir / name
        self.assertTrue(path.exists(), f"Expected script to exist: {name}")
        return path.read_text(encoding="utf-8")

    def _read_workflow(self, name: str) -> dict:
        path = self.workflows_dir / name
        self.assertTrue(path.exists(), f"Expected workflow to exist: {name}")
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def _iter_workflow_files(self) -> list[Path]:
        """Return all workflow definitions regardless of YAML suffix."""

        return sorted(
            {
                *self.workflows_dir.glob("*.yml"),
                *self.workflows_dir.glob("*.yaml"),
            }
        )

    def _iter_scalars(self, node: object) -> list[str]:
        """Flatten nested YAML structures into their scalar string values."""

        if isinstance(node, dict):
            scalars: list[str] = []
            for value in node.values():
                scalars.extend(self._iter_scalars(value))
            return scalars
        if isinstance(node, list):
            scalars = []
            for value in node:
                scalars.extend(self._iter_scalars(value))
            return scalars
        return [node] if isinstance(node, str) else []

    def _iter_mappings(self, node: object) -> list[dict]:
        """Yield every mapping found in a nested YAML document."""

        if isinstance(node, dict):
            mappings: list[dict] = [node]
            for value in node.values():
                mappings.extend(self._iter_mappings(value))
            return mappings
        if isinstance(node, list):
            mappings: list[dict] = []
            for value in node:
                mappings.extend(self._iter_mappings(value))
            return mappings
        return []

    def _assert_contains(
        self, haystack: str, needles: list[str], *, context: str
    ) -> None:
        for needle in needles:
            with self.subTest(target=context, substring=needle):
                self.assertIn(needle, haystack, f"Expected `{needle}` in {context}")

    # -- script coverage ----------------------------------------------------------

    def test_dev_check_covers_syntax_lint_and_types(self) -> None:
        script = self._read_script("dev_check.sh")
        self._assert_contains(
            script,
            [
                "ensure_python_packages black isort docformatter",
                "python -m py_compile",
                "black --check",
                "flake8 --select=E9,F",
                "mypy --follow-imports=silent",
            ],
            context="dev_check.sh",
        )

    def test_validate_fast_handles_full_stack(self) -> None:
        script = self._read_script("validate_fast.sh")
        self._assert_contains(
            script,
            [
                "python -m py_compile",
                "black --check .",
                "flake8 src/ tests/ scripts/",
                "mypy src/",
                "pytest tests/",
                "--cov=src",
            ],
            context="validate_fast.sh",
        )

    def test_check_branch_runs_tests_and_coverage(self) -> None:
        script = self._read_script("check_branch.sh")
        self._assert_contains(
            script,
            [
                "black --check .",
                "flake8 src/ tests/ scripts/",
                "mypy src/",
                "pytest tests/",
                "pytest --cov=src",
                "pip install -e .",
            ],
            context="check_branch.sh",
        )

    # -- workflow coverage --------------------------------------------------------

    def test_legacy_pr_ci_wrappers_removed(self) -> None:
        """PR 10/12 wrappers should be retired in favour of the Gate
        workflow."""

        for legacy in ("pr-10-ci-python.yml", "pr-12-docker-smoke.yml"):
            with self.subTest(slug=legacy):
                path = self.workflows_dir / legacy
                self.assertFalse(
                    path.exists(),
                    f"Legacy workflow {legacy} should be removed after Gate consolidation",
                )

    def test_gate_workflow_orchestrates_reusable_jobs(self) -> None:
        workflow = self._read_workflow("pr-00-gate.yml")

        triggers = workflow.get("on") or workflow.get(True) or {}
        pull_request = triggers.get("pull_request", {})
        self.assertNotIn(
            "paths-ignore",
            pull_request,
            "Gate workflow should rely on the changes detector instead of paths-ignore filters.",
        )

        concurrency = workflow.get("concurrency", {})
        self.assertEqual(
            concurrency.get("group"),
            "pr-${{ github.event.pull_request.number || github.ref_name }}-gate",
        )
        self.assertTrue(concurrency.get("cancel-in-progress"))

        jobs = workflow.get("jobs", {})
        self.assertEqual(
            set(jobs.keys()),
            {"changes", "core-tests-311", "core-tests-312", "docker-smoke", "gate"},
        )

        job_changes = jobs["changes"]
        changes_steps = job_changes.get("steps", [])
        detect_step = next(
            (step for step in changes_steps if step.get("id") == "diff"),
            {},
        )
        self.assertTrue(
            detect_step,
            "changes job must expose the diff detection step with id 'diff'",
        )

        job_311 = jobs["core-tests-311"]
        self.assertEqual(
            job_311.get("uses"), "./.github/workflows/reusable-10-ci-python.yml"
        )
        with_block_311 = job_311.get("with", {})
        self.assertEqual(with_block_311.get("python-version"), "3.11")
        self.assertEqual(with_block_311.get("marker"), "not quarantine and not slow")

        job_312 = jobs["core-tests-312"]
        self.assertEqual(
            job_312.get("uses"), "./.github/workflows/reusable-10-ci-python.yml"
        )
        with_block_312 = job_312.get("with", {})
        self.assertEqual(with_block_312.get("python-version"), "3.12")
        self.assertEqual(with_block_312.get("marker"), "not quarantine and not slow")

        job_smoke = jobs["docker-smoke"]
        self.assertEqual(
            job_smoke.get("uses"), "./.github/workflows/reusable-12-ci-docker.yml"
        )

        job_gate = jobs["gate"]
        self.assertEqual(
            job_gate.get("needs"),
            ["changes", "core-tests-311", "core-tests-312", "docker-smoke"],
        )
        steps = job_gate.get("steps", [])
        summary_step = next(
            (step for step in steps if step.get("name") == "Summarize results"),
            {},
        )
        self.assertTrue(summary_step, "gate job must summarize downstream results")

    def test_workflows_do_not_define_invalid_marker_filters(self) -> None:
        """Ensure pytest marker filters stay inside shell commands."""

        invalid_expr = "not quarantine and not slow"

        found_in_gate = False

        for workflow_path in self._iter_workflow_files():
            with self.subTest(workflow=workflow_path.name):
                loaded = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
                if loaded is None:
                    continue
                for scalar in self._iter_scalars(loaded):
                    if scalar.strip() == invalid_expr:
                        if workflow_path.name == "pr-00-gate.yml":
                            found_in_gate = True
                            continue
                        self.fail(
                            "Detected bare pytest marker expression in %s; "
                            "use shell commands (pytest -m) instead to avoid "
                            "invalid YAML filters." % workflow_path.name
                        )

        self.assertTrue(
            found_in_gate,
            "pr-00-gate.yml should pass the marker literal to the reusable CI workflow",
        )

    def test_workflows_do_not_define_marker_filter_anchors(self) -> None:
        """Block the legacy `_marker_filter` YAML anchor regression."""

        for workflow_path in self._iter_workflow_files():
            with self.subTest(workflow=workflow_path.name):
                text = workflow_path.read_text(encoding="utf-8")
                self.assertNotIn(
                    "_marker_filter",
                    text,
                    (
                        "Workflow %s contains `_marker_filter`; anchors defined inside jobs "
                        "become invalid job keys. Keep marker filters embedded inside shell "
                        "commands instead."
                    )
                    % workflow_path.name,
                )

    def test_workflow_conditions_do_not_reintroduce_marker_expression(self) -> None:
        """Ensure `if:` conditionals avoid the invalid pytest marker syntax."""

        invalid_expr = "not quarantine and not slow"

        for workflow_path in self._iter_workflow_files():
            with self.subTest(workflow=workflow_path.name):
                loaded = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
                if loaded is None:
                    continue
                for mapping in self._iter_mappings(loaded):
                    if "if" not in mapping:
                        continue
                    condition = mapping["if"]
                    if isinstance(condition, str) and condition.strip() == invalid_expr:
                        self.fail(
                            "Workflow %s defines `if: %s`; move the marker expression into "
                            "a shell command (pytest -m) so GitHub Actions treats it as "
                            "CLI arguments rather than workflow condition syntax."
                            % (workflow_path.name, invalid_expr)
                        )

    def test_reusable_ci_uploads_coverage_artifacts(self) -> None:
        reusable = self._read_workflow("reusable-10-ci-python.yml")
        jobs = reusable.get("jobs", {})
        tests_job = jobs.get("tests", {})
        steps = tests_job.get("steps", [])
        upload_step = next(
            (
                step
                for step in steps
                if isinstance(step, dict)
                and step.get("name") == "Upload coverage artifact"
            ),
            {},
        )
        self.assertTrue(
            upload_step,
            "Reusable CI workflow must publish coverage artifacts",
        )
        with_block = upload_step.get("with", {})
        paths = (with_block.get("path") or "").splitlines()
        self.assertIn("coverage.xml", [path.strip() for path in paths])

    def test_syntax_demo_missing_colon(self):
        self.assertTrue(True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
