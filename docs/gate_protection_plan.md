# Gate Workflow Branch Protection Plan

## Scope and Key Constraints
- Enforce the `gate` GitHub Actions workflow as a required status check on the default branch (currently `main`).
- Remove any legacy required status checks that overlap or conflict with `gate` (e.g., older `CI` workflows).
- Enable the "Require branches to be up to date" option so merges must include the latest default-branch commits.
- Preserve existing job names inside `gate.yml` (`core tests (3.11)`, `core tests (3.12)`, `docker smoke`, `gate`) to avoid downstream automation regressions.
- Communicate the new requirement in contributor-facing docs (CONTRIBUTING.md) without altering unrelated guidance.

## Acceptance Criteria / Definition of Done
- Default branch protection rule lists `gate` as a required status check and prevents merging until it succeeds.
- Any deprecated required checks (e.g., `CI`) are removed from the protection rule.
- "Require branches to be up to date" is enabled for the default branch protection rule.
- CONTRIBUTING.md explicitly instructs contributors that the `gate` check must pass before merging.
- A test pull request shows the `gate` check as required and blocks merge when failing or pending.

## Initial Task Checklist
1. Audit current branch protection settings for the default branch and note existing required checks.
2. Update branch protection:
   - Add `gate` as a required status check.
   - Remove obsolete required checks.
   - Enable "Require branches to be up to date" if not already set.
3. Verify recent `gate` workflow runs to confirm job naming and stability.
4. Create or update documentation (CONTRIBUTING.md) to mention the required `gate` check.
5. Open a validation pull request to confirm the `gate` check appears as required and blocks merge until passing.
6. Record findings and screenshots/logs (if applicable) demonstrating the protection rule and validation PR behavior.

## Implementation Summary

- Added `tools/enforce_gate_branch_protection.py` to interrogate and update the default branch protection rule via the GitHub
  REST API so the `Gate / gate` context remains the sole required check while "Require branches to be up to date" stays enabled.
- The helper defaults to `GITHUB_REPOSITORY`/`DEFAULT_BRANCH` and can run in dry-run mode to audit the current contexts before
  applying changes.
- Contributors without direct settings access can now request an owner to run the script with a fine-grained `GITHUB_TOKEN`
  instead of navigating the UI, ensuring infrastructure as code parity for the protection rule. When no protection rule exists,
  the helper bootstraps one with the required status checks so teams can enable gate in a single `--apply` invocation.
- Scheduled automation (`.github/workflows/enforce-gate-branch-protection.yml`) executes the helper nightly and on-demand,
  applying fixes whenever the optional `BRANCH_PROTECTION_TOKEN` secret is configured.

## Automation Requirements

- Create a fine-grained PAT with the **Administration: Branches** scope (repo → settings → developer settings → fine-grained
  personal access tokens). Attach it to the repository as the `BRANCH_PROTECTION_TOKEN` secret so the enforcement workflow can
  manage branch protection.
- The workflow runs on a nightly cron and via the `workflow_dispatch` trigger. When the secret is absent the job exits early
  with an informational log, allowing maintainers to opt in when they are ready to enforce gate automatically.
- Manual runs surface the audit diff in the workflow logs, mirroring the script's dry-run output before applying updates.

## Usage Notes

Run a dry check to review the current branch protection rule:

```bash
GITHUB_TOKEN=ghp_xxx python tools/enforce_gate_branch_protection.py --repo stranske/Trend_Model_Project --branch main
```

Expected dry-run output when the rule is correct:

```
Repository: stranske/Trend_Model_Project
Branch:     main
Current contexts: Gate / gate
Desired contexts: Gate / gate
Current 'require up to date': True
Desired 'require up to date': True
No changes required.
```

To automate auditing without applying fixes, add `--check`. The command exits with
status code `1` when drift is detected, making it suitable for CI monitors:

```bash
GITHUB_TOKEN=ghp_xxx python tools/enforce_gate_branch_protection.py --repo stranske/Trend_Model_Project --branch main --check
```

Apply corrections (if the dry run indicates drift or the rule is missing):

```bash
GITHUB_TOKEN=ghp_xxx python tools/enforce_gate_branch_protection.py --repo stranske/Trend_Model_Project --branch main --apply
```

The script patches `required_status_checks` in-place and leaves other branch protection toggles untouched. When the rule is
absent the helper creates one with GitHub’s default toggles (admins not enforced, reviews/restrictions unset) before ensuring
`Gate / gate` is the required context. Use `--context` to temporarily allow additional contexts or `--no-clean` to preserve
existing extras while asserting Gate.

## Validation Checklist

- After enforcing the rule, run `gh api repos/stranske/Trend_Model_Project/branches/main/protection/required_status_checks` to
  confirm the payload lists only `Gate / gate` and has `"strict": true`.
- Trigger the "Enforce Gate Branch Protection" workflow via `workflow_dispatch` (with `BRANCH_PROTECTION_TOKEN` configured) and
  verify the run logs either report "No changes required." or "Update successful." along with the enforced contexts.
- Open a throwaway pull request from an outdated branch to confirm the merge box displays `Required — Gate / gate` and blocks
  merges until the workflow finishes successfully.
- Capture screenshots or console transcripts for the repository automation log (attach to the issue or link in meeting notes).
