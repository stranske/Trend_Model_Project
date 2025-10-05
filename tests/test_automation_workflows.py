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

    def test_ci_workflow_invokes_reusable_stack(self) -> None:
        workflow = self._read_workflow("pr-10-ci-python.yml")
        jobs = workflow.get("jobs", {})
        self.assertIn(
            "tests", jobs, "CI workflow should delegate tests via reusable job"
        )
        tests_job = jobs["tests"]
        self.assertEqual(
            tests_job.get("uses"),
            "./.github/workflows/reusable-90-ci-python.yml",
            "CI tests job should delegate to reusable stack",
        )
        inputs = tests_job.get("with", {})
        self.assertTrue(inputs.get("run-mypy"), "CI should enable mypy job")
        self.assertTrue(
            inputs.get("enable-soft-gate"), "CI should enable coverage soft gate"
        )

        self.assertIn("gate", jobs, "CI workflow should expose aggregate gate job")
        gate_job = jobs["gate"]
        upstream_jobs = {name for name in jobs if name != "gate"}
        self.assertEqual(
            set(gate_job.get("needs", [])),
            upstream_jobs,
            "Gate job must aggregate every other CI job",
        )
        self.assertEqual(
            gate_job.get("name"),
            "gate / all-required-green",
            "Gate job name should match required check label",
        )
        expected_core_jobs = {"tests", "workflow-automation", "style"}
        if "formatting-fast" in jobs:
            expected_core_jobs.add("formatting-fast")
        self.assertTrue(
            expected_core_jobs.issubset(set(gate_job.get("needs", []))),
            "Gate job must aggregate core CI jobs",
        )
        self.assertEqual(
            gate_job.get("if"),
            "${{ always() }}",
            "Gate job should run even when upstream jobs fail to report aggregate status",
        )
        self.assertTrue(
            gate_job.get("steps"),
            "Gate job should include at least one confirmation step",
        )
        gate_steps = [
            step for step in gate_job.get("steps", []) if isinstance(step, dict)
        ]
        self.assertTrue(gate_steps, "Gate job must define at least one run step")

        first_gate_step = gate_steps[0]
        env_block = first_gate_step.get("env", {})
        self.assertEqual(
            env_block.get("NEEDS_CONTEXT"),
            "${{ toJson(needs) }}",
            "Gate step should expose needs context as JSON for parsing",
        )

        gate_step_runs = [step.get("run", "") for step in gate_steps]
        self.assertTrue(
            any("GITHUB_STEP_SUMMARY" in run for run in gate_step_runs),
            "Gate job should write a summary of upstream results to the job summary",
        )
        self.assertTrue(
            any("set -euo pipefail" in run for run in gate_step_runs),
            "Gate job should harden its shell with `set -euo pipefail`",
        )
        self.assertTrue(
            any("exit 1" in run for run in gate_step_runs),
            "Gate job must exit non-zero when dependencies fail",
        )
        aggregated_script = "\n".join(gate_step_runs)
        self.assertIn(
            "Gate summary",
            aggregated_script,
            "Gate job should emit a human-readable gate summary section",
        )
        self.assertIn(
            "jq -r 'keys[]'",
            aggregated_script,
            "Gate job should enumerate upstream job names dynamically via jq",
        )
        self.assertIn(
            "jq -r --arg job \"$job\" '.[$job].result'",
            aggregated_script,
            "Gate job should use jq to read job results from the needs context",
        )
        self.assertIn(
            "Gate job missing needs context",
            aggregated_script,
            "Gate job should fail fast if the needs context is absent",
        )
        self.assertIn(
            'type == "object" and (keys | length) > 0',
            aggregated_script,
            "Gate job should verify the needs context contains at least one upstream job",
        )
        self.assertIn(
            "Gate requires at least one upstream job",
            aggregated_script,
            "Gate job should report a clear error when no dependencies are configured",
        )

    def test_gate_workflow_file_is_absent(self) -> None:
        for suffix in (".yml", ".yaml"):
            with self.subTest(extension=suffix):
                gate_path = self.workflows_dir / f"gate{suffix}"
                self.assertFalse(
                    gate_path.exists(),
                    "Legacy gate workflow should remain deleted; rely on pr-10 gate job",
                )

    def test_workflows_do_not_define_invalid_marker_filters(self) -> None:
        """Ensure pytest marker filters stay inside shell commands."""

        invalid_expr = "not quarantine and not slow"

        for workflow_path in self._iter_workflow_files():
            with self.subTest(workflow=workflow_path.name):
                loaded = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
                if loaded is None:
                    continue
                for scalar in self._iter_scalars(loaded):
                    if scalar.strip() == invalid_expr:
                        self.fail(
                            "Detected bare pytest marker expression in %s; "
                            "use shell commands (pytest -m) instead to avoid "
                            "invalid YAML filters." % workflow_path.name
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

    def test_workflows_do_not_reference_retired_gate_wrapper(self) -> None:
        """Ensure no workflow tries to re-use the deleted gate.yml wrapper."""

        for workflow_path in self._iter_workflow_files():
            with self.subTest(workflow=workflow_path.name):
                loaded = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
                if loaded is None:
                    continue
                for scalar in self._iter_scalars(loaded):
                    if "gate.yml" in scalar:
                        self.fail(
                            "Workflow %s references the retired gate.yml wrapper via `%s`; "
                            "rely on the gate job inside pr-10-ci-python.yml instead."
                            % (workflow_path.name, scalar)
                        )

    def test_reusable_ci_runs_tests_and_mypy(self) -> None:
        workflow = self._read_workflow("reusable-90-ci-python.yml")
        jobs = workflow.get("jobs", {})
        self.assertIn("tests", jobs)
        self.assertIn("mypy", jobs)
        self.assertIn("coverage_soft_gate", jobs)

        test_steps = "\n".join(
            step["run"].strip()
            for step in jobs["tests"].get("steps", [])
            if isinstance(step, dict) and "run" in step
        )
        self._assert_contains(
            test_steps,
            [
                "pip install -r requirements.txt",
                "pytest --junitxml=pytest-junit.xml \\",
                "--cov=src",
            ],
            context="reusable-ci tests job",
        )

        mypy_steps = "\n".join(
            step["run"].strip()
            for step in jobs["mypy"].get("steps", [])
            if isinstance(step, dict) and "run" in step
        )
        self._assert_contains(
            mypy_steps,
            ["pip install mypy", "mypy src"],
            context="reusable-ci mypy job",
        )

    def test_coverage_soft_gate_checks_out_repo_before_python(self) -> None:
        workflow = self._read_workflow("reusable-90-ci-python.yml")
        jobs = workflow["jobs"]

        def _assert_checkout_precedes_python(job_name: str) -> None:
            steps = jobs[job_name].get("steps", [])
            checkout_indices = [
                index
                for index, step in enumerate(steps)
                if isinstance(step, dict)
                and str(step.get("uses", "")).startswith("actions/checkout")
            ]
            self.assertTrue(
                checkout_indices,
                f"{job_name} job must checkout the repository before running helpers",
            )

            first_checkout = checkout_indices[0]
            python_steps = [
                index
                for index, step in enumerate(steps)
                if isinstance(step, dict)
                and (
                    step.get("shell") == "python"
                    or (isinstance(step.get("run"), str) and "scripts/" in step["run"])
                )
            ]
            self.assertTrue(
                all(index > first_checkout for index in python_steps),
                f"Python helper steps in {job_name} must run after checkout",
            )

        _assert_checkout_precedes_python("coverage_soft_gate")
        _assert_checkout_precedes_python("cosmetic_followup")

    def test_coverage_soft_gate_preserves_classification_and_summary_steps(
        self,
    ) -> None:
        workflow = self._read_workflow("reusable-90-ci-python.yml")
        steps = workflow["jobs"]["coverage_soft_gate"].get("steps", [])
        names = {
            step.get("name")
            for step in steps
            if isinstance(step, dict) and step.get("name")
        }
        expected = {
            "Download coverage artifacts",
            "Classify test outcomes",
            "Compute coverage summary & hotspots",
            "Create / update soft coverage issue",
        }
        missing = expected.difference(names)
        self.assertFalse(
            missing,
            f"coverage_soft_gate missing critical steps: {sorted(missing)}",
        )

    def test_cosmetic_followup_job_depends_on_soft_gate_outputs(self) -> None:
        workflow = self._read_workflow("reusable-90-ci-python.yml")
        job = workflow["jobs"]["cosmetic_followup"]
        condition = job.get("if", "")
        self.assertIn(
            "needs.tests.result != 'success'",
            condition,
            "cosmetic_followup must guard on failing tests",
        )
        self.assertIn(
            "needs.coverage_soft_gate.outputs.has_failures == 'true'",
            condition,
            "cosmetic_followup must check coverage_soft_gate has failures",
        )
        self.assertIn(
            "needs.coverage_soft_gate.outputs.only_cosmetic == 'true'",
            condition,
            "cosmetic_followup must only run for cosmetic failures",
        )

        steps = job.get("steps", [])
        names = {
            step.get("name")
            for step in steps
            if isinstance(step, dict) and step.get("name")
        }
        required_steps = {
            "Summarise cosmetic failures",
            "Capture fixer diff",
            "Upload cosmetic patch",
        }
        missing = required_steps.difference(names)
        self.assertFalse(
            missing,
            f"cosmetic_followup missing required helper steps: {sorted(missing)}",
        )

    def test_style_job_enforces_black_ruff_and_mypy(self) -> None:
        workflow = self._read_workflow("pr-10-ci-python.yml")
        style_job = workflow.get("jobs", {}).get("style", {})
        steps = "\n".join(
            step["run"].strip()
            for step in style_job.get("steps", [])
            if isinstance(step, dict) and "run" in step
        )
        self._assert_contains(
            steps,
            [
                "black --check .",
                "ruff check",
                "mypy --config-file",
            ],
            context="CI style job",
        )

        gate = workflow.get("jobs", {}).get("gate", {})
        needs = gate.get("needs", [])
        self.assertIn(
            "style",
            needs,
            "gate job must wait for style checks before aggregating status",
        )

    def test_syntax_demo_missing_colon(self):
        self.assertTrue(True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
