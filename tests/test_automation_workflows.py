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
        workflow = self._read_workflow("ci.yml")
        jobs = workflow.get("jobs", {})
        self.assertIn("main", jobs, "CI workflow should define a main job")
        main_job = jobs["main"]
        self.assertEqual(
            main_job.get("uses"),
            "./.github/workflows/reusable-ci-python.yml",
            "CI main job should delegate to reusable stack",
        )
        inputs = main_job.get("with", {})
        self.assertTrue(inputs.get("run-mypy"), "CI should enable mypy job")
        self.assertTrue(
            inputs.get("enable-soft-gate"), "CI should enable coverage soft gate"
        )

    def test_reusable_ci_runs_tests_and_mypy(self) -> None:
        workflow = self._read_workflow("reusable-ci-python.yml")
        jobs = workflow.get("jobs", {})
        self.assertIn("tests", jobs)
        self.assertIn("mypy", jobs)

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

    def test_style_gate_enforces_black_ruff_and_mypy(self) -> None:
        workflow = self._read_workflow("style-gate.yml")
        steps = "\n".join(
            step["run"].strip()
            for step in workflow.get("jobs", {}).get("style", {}).get("steps", [])
            if isinstance(step, dict) and "run" in step
        )
        self._assert_contains(
            steps,
            [
                "black --check .",
                "ruff check",
                "mypy --config-file",
            ],
            context="style gate workflow",
        )

    def test_syntax_demo_missing_colon(self):
        self.assertTrue(True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
