# Gate Branch Protection Validation Runbook

This runbook documents how to demonstrate that the Gate workflow is enforced as a required
status check on the default branch. Follow these steps whenever branch protection settings are
updated or during periodic audits.

## Prerequisites
- Repository administrator access (or a fine-grained personal access token with **Administration → Branches** scope).
- [GitHub CLI](https://cli.github.com/) `gh` tool or a shell session where the `tools/enforce_gate_branch_protection.py` helper
  script can run.
- A scratch branch to use for the validation pull request.

## 1. Audit Current Protection Rule
1. Export the status check configuration:
   ```bash
   gh api repos/stranske/Trend_Model_Project/branches/main/protection/required_status_checks \
     --jq '{strict: .strict, contexts: .contexts}'
   ```
   - Confirm the output shows `{"strict":true,"contexts":["Gate / gate"]}`.
   - If `gh` is unavailable, run the helper in dry-run mode:
     ```bash
     GITHUB_TOKEN=ghp_xxx python tools/enforce_gate_branch_protection.py --repo stranske/Trend_Model_Project --branch main
     ```
     Ensure it prints `No changes required.`

2. If the audit reveals legacy contexts, enforce the rule with the helper:
   ```bash
   GITHUB_TOKEN=ghp_xxx python tools/enforce_gate_branch_protection.py \
     --repo stranske/Trend_Model_Project --branch main --apply
   ```
   Record the console output in the issue log.

## 2. Create Validation Pull Request
1. Check out a new branch and make a trivial change (e.g., update a comment) so a pull request can be opened.
2. Open a **draft** pull request targeting `main`.
3. From the Actions tab, cancel the Gate workflow run mid-flight or push a commit that deliberately fails one of the jobs
   (for example, run `pytest -k always_fail` and commit the failing test).
4. Confirm the PR shows **Required — Gate / gate** in the status area and the merge button is disabled with wording similar to
   `Merging is blocked`.
5. Capture a screenshot or copy the status text into the issue notes.

## 3. Resolve the Failure
1. Push a fix that allows the Gate workflow to pass (e.g., revert the failing test or re-run the cancelled job).
2. Wait for the workflow to finish successfully.
3. Verify the status area now shows **Required — Gate / gate** with a green check and the merge button becomes available.
4. Update the issue or documentation with the pass/fail timestamps and the URL of the validation PR for traceability.

## 4. Close Out
- Merge or close the draft PR without merging into `main` (if using a deliberately failing change, force-push to remove it).
- Attach artifacts (logs, screenshots) to the issue tracker entry for future audits.
- If automation tokens were rotated during the process, re-run the helper with `--check` to ensure the rule remains intact.

## Expected Evidence
- JSON payload or helper output confirming `strict: true` and `contexts: ["Gate / gate"]`.
- Screenshot or textual proof of the PR blocked on a failing Gate run.
- Follow-up proof showing the Gate run passed and merging was permitted.
- Links to the validation PR and workflow runs recorded in issue #2495.
