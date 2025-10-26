from __future__ import annotations

import unittest
from pathlib import Path
from typing import cast

import yaml


class TestAutomationWorkflowCoverage(unittest.TestCase):
    """Validate that automation scripts and workflows cover core gates."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.scripts_dir = cls.project_root / "scripts"
        cls.workflows_dir = cls.project_root / ".github" / "workflows"
        cls.github_scripts_dir = cls.project_root / ".github" / "scripts"

    # -- helpers -----------------------------------------------------------------

    def _read_script(self, name: str) -> str:
        path = self.scripts_dir / name
        self.assertTrue(path.exists(), f"Expected script to exist: {name}")
        return path.read_text(encoding="utf-8")

    def _read_workflow(self, name: str) -> dict:
        path = self.workflows_dir / name
        self.assertTrue(path.exists(), f"Expected workflow to exist: {name}")
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def _read_github_script(self, name: str) -> str:
        path = self.github_scripts_dir / name
        self.assertTrue(
            path.exists(),
            f"Expected GitHub helper script to exist: {name}",
        )
        return path.read_text(encoding="utf-8")

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
            {
                "detect",
                "python-ci",
                "github-scripts-tests",
                "docker-smoke",
                "ledger-validation",
                "summary",
                "autofix",
            },
        )

        job_detect = jobs["detect"]
        detect_steps = job_detect.get("steps", [])
        detect_step = next(
            (step for step in detect_steps if step.get("id") == "diff"),
            {},
        )
        self.assertTrue(
            detect_step,
            "detect job must expose the diff detection step with id 'diff'",
        )

        job_python_ci = jobs["python-ci"]
        self.assertEqual(
            job_python_ci.get("uses"), "./.github/workflows/reusable-10-ci-python.yml"
        )
        with_block_python = job_python_ci.get("with", {})
        self.assertEqual(with_block_python.get("python-versions"), '["3.11", "3.12"]')
        self.assertEqual(with_block_python.get("marker"), "not quarantine and not slow")
        self.assertEqual(with_block_python.get("primary-python-version"), "3.11")

        job_smoke = jobs["docker-smoke"]
        self.assertEqual(
            job_smoke.get("uses"), "./.github/workflows/reusable-12-ci-docker.yml"
        )

        job_gate = jobs["summary"]
        self.assertEqual(
            job_gate.get("needs"),
            [
                "detect",
                "python-ci",
                "docker-smoke",
                "github-scripts-tests",
                "ledger-validation",
            ],
        )
        steps = job_gate.get("steps", [])
        summary_step = next(
            (step for step in steps if step.get("name") == "Summarize results"),
            {},
        )
        self.assertTrue(summary_step, "gate job must summarize downstream results")
        status_step = next(
            (step for step in steps if step.get("name") == "Report Gate commit status"),
            {},
        )
        self.assertTrue(
            status_step,
            "gate job must publish a legacy commit status so branch protection resolves",
        )

    def test_gate_docs_only_handler_reports_fast_pass(self) -> None:
        workflow = self._read_workflow("pr-00-gate.yml")
        gate_job = workflow.get("jobs", {}).get("summary", {})
        self.assertTrue(gate_job, "Gate workflow must define gate job")

        steps = gate_job.get("steps", [])
        docs_step = next(
            (step for step in steps if step.get("id") == "docs_only"), None
        )
        self.assertIsNotNone(
            docs_step, "Gate workflow should expose docs-only handler step"
        )
        self.assertIsInstance(
            docs_step, dict, "docs_only step definition should be a mapping"
        )
        docs_step_mapping = cast(dict[str, object], docs_step)

        with_block = cast(dict[str, object], docs_step_mapping.get("with", {}))
        script_obj = with_block.get("script", "")
        self.assertIsInstance(script_obj, str)
        script = script_obj
        self.assertIn(
            "require('./.github/scripts/gate-docs-only.js')",
            script,
            "Docs-only handler should import the shared helper",
        )
        self.assertIn(
            "handleDocsOnlyFastPass(",
            script,
            "Docs-only handler should delegate to the helper entry point",
        )

        helper_source = self._read_github_script("gate-docs-only.js")
        expected_patterns = {
            "defines success state output": r"state:\s*'success'",
            "defines description output": r"description:\s*message",
            "logs message": r"core\.info\(message\)",
            "includes docs-only fast-pass messaging": r"Gate fast-pass: docs-only change detected; heavy checks skipped\.",
            "writes step summary": r"summary\.addHeading\(summaryHeading,\s*3\)",
        }

        for label, pattern in expected_patterns.items():
            with self.subTest(helper_pattern=label):
                self.assertRegex(
                    helper_source,
                    pattern,
                    msg=f"Helper should {label}",
                )

    def test_gate_cleans_up_legacy_docs_only_comment(self) -> None:
        workflow = self._read_workflow("pr-00-gate.yml")
        gate_job = workflow.get("jobs", {}).get("summary", {})
        self.assertTrue(gate_job, "Gate workflow must define gate job")

        steps = gate_job.get("steps", [])
        cleanup_step = None
        cleanup_names = (
            "Clean up legacy docs-only comment",
            "Remove docs-only fast-pass comment when not needed",
        )
        for step in steps:
            if isinstance(step, dict) and step.get("name") in cleanup_names:
                cleanup_step = step
                break

        self.assertIsNotNone(
            cleanup_step,
            "Gate workflow should remove legacy docs-only comments to stay idempotent",
        )

        condition = (cleanup_step or {}).get("if", "")
        cleanup_step_name = (cleanup_step or {}).get("name")
        expected_condition = (
            "${{ always() }}"
            if cleanup_step_name == "Clean up legacy docs-only comment"
            else "needs.detect.outputs.doc_only != 'true'"
        )
        self.assertEqual(
            condition,
            expected_condition,
            "Cleanup step condition should reflect docs-only lifecycle handling",
        )

        with_block = cast(dict[str, object], (cleanup_step or {}).get("with", {}))
        script_obj = with_block.get("script", "")
        self.assertIsInstance(script_obj, str)
        script = cast(str, script_obj)
        self.assertIn(
            "require('./.github/scripts/comment-dedupe.js')",
            script,
            "Cleanup step should import the shared dedupe helper",
        )
        self.assertIn(
            "removeMarkerComments",
            script,
            "Cleanup step should delegate to removeMarkerComments",
        )

        helper_source = self._read_github_script("comment-dedupe.js")
        expected_cleanup_patterns = {
            "defines marker": r"marker\s*\|\|",
            "defines base message": r"baseMessage\s*\|\|",
            "lists pull request comments": r"github\.rest\.issues\.listComments",
            "detects marker comment": r"comment\.body\.includes\(marker\)",
            "matches legacy prefix": r"trimmed\.startsWith\(legacy\)",
            "removes marker comment": r"github\.rest\.issues\.deleteComment",
        }

        for label, pattern in expected_cleanup_patterns.items():
            with self.subTest(helper_cleanup_pattern=label):
                self.assertRegex(
                    helper_source,
                    pattern,
                    msg=f"Dedupe helper should {label}",
                )

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

    def test_gate_detector_covers_common_docs_patterns(self) -> None:
        workflow = self._read_workflow("pr-00-gate.yml")
        detect_job = workflow.get("jobs", {}).get("detect", {})
        self.assertTrue(detect_job, "Gate workflow must expose detect job")

        steps = detect_job.get("steps", [])
        diff_step = next((step for step in steps if step.get("id") == "diff"), None)
        self.assertIsNotNone(
            diff_step, "Detect job must use diff step to classify changes"
        )

        with_block = cast(dict[str, object], (diff_step or {}).get("with", {}))
        script_obj = with_block.get("script", "")
        self.assertIsInstance(script_obj, str)
        script = cast(str, script_obj)
        self.assertIn(
            "require('./.github/scripts/detect-changes.js')",
            script,
            "Detect step should import the shared change detector",
        )

        detector_source = self._read_github_script("detect-changes.js")
        expected_snippets = {
            "supports doc extensions": ".txt",
            "covers quarto docs": ".qmd",
            "covers doc basenames": "const DOC_BASENAMES = new Set([",
            "handles documentation prefixes": "const DOC_PREFIXES = [",
            "scans nested documentation segments": "const DOC_SEGMENTS = [",
            "contains mkdocs basename": "'mkdocs',",
            "contains docfx basename": "'docfx',",
            "captures windows-style segments": "\\\\docs\\\\",
            "captures manual segment": "/manual/",
            "captures windows manual segment": "\\\\manual\\\\",
        }

        for label, snippet in expected_snippets.items():
            with self.subTest(helper_snippet=label):
                self.assertIn(
                    snippet,
                    detector_source,
                    f"Detector helper should {label}",
                )

    def test_gate_downloads_coverage_with_tolerance(self) -> None:
        workflow = self._read_workflow("pr-00-gate.yml")
        gate_job = workflow.get("jobs", {}).get("summary", {})
        self.assertTrue(gate_job, "Gate workflow must define summary job")

        coverage_steps = [
            step
            for step in gate_job.get("steps", [])
            if isinstance(step, dict)
            and isinstance(step.get("name"), str)
            and step["name"].startswith("Download Gate artifacts")
        ]

        self.assertEqual(
            len(coverage_steps),
            1,
            "Gate summary job should download the consolidated gate artifacts",
        )

        step = coverage_steps[0]
        self.assertTrue(
            step.get("continue-on-error"),
            "Coverage download should tolerate missing artifacts",
        )
        condition = step.get("if", "")
        self.assertIn(
            "needs.detect.outputs.doc_only != 'true'",
            condition,
            "Coverage download must skip docs-only runs",
        )
        self.assertEqual(
            step.get("uses"),
            "actions/download-artifact@v4",
            "Coverage downloads should use actions/download-artifact v4",
        )
        inputs = step.get("with", {})
        self.assertIsInstance(inputs, dict)
        self.assertEqual(inputs.get("pattern"), "gate-*")
        self.assertTrue(inputs.get("merge-multiple"))
        self.assertEqual(inputs.get("path"), "gate_artifacts/downloads")

    def test_gate_summary_reports_job_table(self) -> None:
        workflow = self._read_workflow("pr-00-gate.yml")
        gate_job = workflow.get("jobs", {}).get("summary", {})
        self.assertTrue(gate_job, "Gate workflow must define summary job")

        summarize_step = next(
            (
                step
                for step in gate_job.get("steps", [])
                if isinstance(step, dict) and step.get("id") == "summarize"
            ),
            None,
        )

        self.assertIsNotNone(
            summarize_step,
            "Gate workflow should expose summarize step for results table",
        )

        script = (summarize_step or {}).get("run", "")
        # After extraction, the workflow calls the gate_summary.py helper
        self.assertIn("python .github/scripts/gate_summary.py", script)

        # Verify the helper script exists and contains the expected logic
        gate_summary_script = self._read_github_script("gate_summary.py")
        self.assertIn("| Job | Result |", gate_summary_script)
        self.assertIn(
            "Docs-only change detected; heavy checks skipped",
            gate_summary_script,
            "Gate summary script should append docs-only messaging to the Gate summary",
        )
        self.assertIn(
            "Gate fast-pass: docs-only change detected; heavy checks skipped.",
            gate_summary_script,
            "Gate summary script should set the fast-pass description that matches the marker comment",
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
        normalized_paths = [path.strip() for path in paths]
        self.assertIn(
            "artifacts/coverage",
            normalized_paths,
            "Coverage artifact should bundle the staged artifacts/coverage directory",
        )

    def test_syntax_demo_missing_colon(self):
        self.assertTrue(True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
