# Coverage Trend Monitoring Plan

## Scope & Key Constraints
- Monitor code coverage changes within the existing CI workflows without introducing hard gating beyond current requirements.
- Parse coverage results from the generated XML report and compare against a committed baseline coverage summary.
- Ensure the solution is GitHub Actions–friendly, leverages existing tooling (Python + actions/github-script@v7), and avoids introducing heavyweight dependencies.
- Maintain compatibility with both pull request workflows and the default branch (baseline updates must be manageable and auditable).
- Provide “soft” notifications (job summary + optional PR comment) without failing CI when coverage drops within the defined tolerance window.
- Keep stored artifacts lightweight and adhere to repository storage constraints; large history storage should be avoided.

## Acceptance Criteria / Definition of Done
1. CI workflow uploads a coverage trend artifact that includes the latest coverage metrics and a historical baseline for comparison.
2. Workflow step parses coverage XML, computes delta against the baseline, and appends a formatted trend line to the GitHub Actions job summary.
3. If coverage decreases by more than the configured threshold (X percentage points), the workflow posts a PR comment containing a 🔶 warning while keeping the job status successful.
4. Documentation updates describe how to refresh the baseline, interpret the trend artifact, and adjust the threshold.
5. Appropriate automated checks (unit test or workflow-level validation) exist to ensure the parser behaves predictably (e.g., handling missing files, malformed XML).

## Initial Task Checklist
- [ ] Audit existing CI workflows to identify the stage that generates coverage XML and determine artifact availability/paths.
- [ ] Define the baseline storage format (e.g., JSON/CSV in repo) and establish conventions for updating it when coverage improves.
- [ ] Implement a Python utility to parse coverage XML, compute coverage metrics, and compare them against the baseline.
- [ ] Wire the utility into the GitHub Actions workflow, capturing outputs for both job summary updates and conditional warning comments via actions/github-script@v7.
- [ ] Add artifact upload steps for the coverage trend data and ensure retention settings meet project expectations.
- [ ] Document usage instructions, including baseline refresh workflow and interpretation of soft alerts.
