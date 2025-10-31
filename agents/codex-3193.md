# Codex Agent Configuration Documentation

This document describes how the Codex agent is configured for issue [#3193](https://github.com/stranske/Trend_Model_Project/issues/3193) and outlines how to extend the associated bootstrap tasks.

## Purpose
The Codex agent ensures that follow-up automation for issue #3193 is discoverable and repeatable. It captures the intended workflow for tracking test coverage gaps so contributors can quickly resume the effort.

## Configuration Overview
- **Location:** `agents/codex-3193.md`
- **Issue Link:** [#3193](https://github.com/stranske/Trend_Model_Project/issues/3193)
- **Primary Goal:** Raise per-file test coverage to at least 95% and guarantee essential functionality is exercised by automated tests.

## Bootstrap Scope
The initial bootstrap request added this documentation file so that future follow-up changes have context. No functional code paths are altered by this PR.

## Setup Instructions
1. Create or activate the project virtual environment using `./scripts/setup_env.sh`.
2. Install any additional tooling required for coverage analysis (e.g., `pip install -r requirements.txt`).
3. Run the coverage helper script (`./scripts/run_tests.sh --cov`) to capture the latest coverage report before starting new work.

## Workflow Guidance
- Use the coverage summary to identify files below the 95% target.
- Address coverage gaps incrementally—focus on one module or related group of functions at a time.
- Document each increment of improved coverage in the PR description and checklist so reviewers can verify progress.

## Additional Resources
- [Testing summary](../TESTING_SUMMARY.md)
- [Coverage summary](../coverage-summary.md)
- [Contribution guidelines](../CONTRIBUTING.md)

---
*Last updated: 2024-06-14*
