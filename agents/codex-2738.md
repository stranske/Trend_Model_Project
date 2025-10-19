<!--
  Bootstrap file for Codex work on [Issue #2738](https://github.com/stranske/Trend_Model_Project/issues/2738).

  Expected deliverables:
  - Verify the repository ruleset that should block deletions/renames of the protected agent workflows (`agents/agent63_pair.yml`, `agents/agent63_pair_secondary.yml`, `agents/agent70_orchestrator.yml`).
  - Update documentation (`docs/ci/AGENTS_POLICY.md`) with the active ruleset name, configuration summary, and a reproducible verification path.

  Use this file to track project scope, constraints, and task progress for resolving Issue #2738.
-->

# Repository Ruleset Validation Plan (Issue #2738)

## Scope & Key Constraints
- **Focus on ruleset behavior**: Confirm that the default-branch ruleset blocks deletions or renames of the three protected workflow files and restricts edits per policy.
- **Out-of-repo configuration**: Ruleset changes may require GitHub UI or API calls; capture evidence but do not commit secrets or policy JSON into the repo.
- **Non-disruptive testing**: Perform verification from an isolated branch using push attempts that intentionally fail without affecting production workflows.
- **Documentation requirements**: Any verified configuration updates must be recorded in `docs/ci/AGENTS_POLICY.md` with clear guidance for future audits.
- **Evidence capture**: Maintain logs, screenshots, or command transcripts proving enforcement; store them in issue/PR references rather than large binary assets in the repo.

## Acceptance Criteria / Definition of Done
1. Demonstrate (with links or captured output) that attempted deletions and renames of each protected workflow are rejected by the repository ruleset.
2. If protections are missing or insufficient, document the misconfiguration, update the ruleset accordingly, and provide proof of the fix.
3. Update `docs/ci/AGENTS_POLICY.md` to include:
   - Ruleset name/ID and relevant conditions.
   - Step-by-step UI path and/or API command for re-validating protection.
   - Date-stamped verification notes.
4. Record verification details (e.g., API responses, command snippets) either within the documentation appendix or linked artifacts so reviewers can reproduce the checks.
5. Obtain review sign-off confirming that ruleset protections align with policy and that documentation is complete.

## Task Checklist & Status
- [x] Enumerate current protected workflows and confirm file paths.
- [x] Query repository rulesets to determine existing protections and enforcement status.
- [ ] Attempt deletes on each protected workflow from a throwaway branch; capture rejection evidence. _(Blocked: repository ruleset currently disabled, so enforcement cannot be validated yet.)_
- [ ] Attempt renames on each protected workflow from a throwaway branch; capture rejection evidence. _(Blocked pending rule reactivation.)_
- [ ] Adjust the repository ruleset (UI/API) to block deletions/renames and restrict edits to CODEOWNERS, then retest. _(Requires maintainer/admin access to GitHub rulesets.)_
- [x] Summarize verification steps, ruleset metadata, and evidence references in `docs/ci/AGENTS_POLICY.md`.
- [ ] Share collected logs/screenshots in Issue #2738 for historical record.

## Progress Log
- **2025-09-05** – Used the public REST API to list repository rulesets. Confirmed `Tests Pass to Merge` (ID `7880490`) is disabled and lacks `restrict_file_updates` entries, so deletion/rename protection is inactive. Captured command output in `docs/ci/agents_ruleset_verification.md` for reproducibility.
