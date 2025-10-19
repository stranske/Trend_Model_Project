# Agents Workflow Protection Policy

**Purpose.** Preserve the Agents 63 pair (issue bridge + ChatGPT sync) and the
Agents 70 orchestrator as always-on automation. The policy explains why the
workflows are treated as "unremovable," the layers that enforce the guardrails,
and the narrow scenarios where changes are allowed.

## Scope and purpose
- **Agents 70 orchestrator** — single dispatch surface for all consumer
  automations. Retiring it strands downstream workflows, so availability is a
  release gate.
- **Agents 63 – Codex issue bridge** — converts labelled issues into working
  branches and bootstrap PRs.
- **Agents 63 – ChatGPT issue sync** — keeps curated topic files in lock-step
  with GitHub issues.

These workflows are coupled: the orchestrator depends on the bridge for intake,
and both Agents 63 files assume the orchestrator will service their dispatches.
Disrupting any one of them breaks the automation topology.

## Protection layers
1. **CODEOWNERS review** – `.github/CODEOWNERS` lists the three workflows under
   maintainer ownership. GitHub will not merge a change without Code Owner
   approval and branch protection keeps the requirement enabled.
2. **Repository ruleset** – the default-branch ruleset (named
   `Tests Pass to Merge`, ruleset ID `7880490`) is the enforcement surface for
   deleting or renaming these workflows. The rule **must** include a
   `restrict_file_updates` block with the following paths and both
   "Block deletions" and "Block renames" toggled on:
   - `.github/workflows/agents-63-chatgpt-issue-sync.yml`
   - `.github/workflows/agents-63-codex-issue-bridge.yml`
   - `.github/workflows/agents-70-orchestrator.yml`

   > **2025-09-05 status:** the ruleset is currently **disabled** with no
   > `restrict_file_updates` entries. Reactivate it in the GitHub UI or via the
   > REST API before attempting verification pushes. Maintainers can bypass the
   > rule in emergencies once it is active; everyone else should receive an
   > immediate push rejection.
3. **Agents Guard workflow** – `health-45-agents-guard.yml` (surfaced as the
   "Agents Critical Guard" check) fails when these files change without the
   required label or when paths disappear. Gate branch protection lists the
   status as required, so failures block merges.
4. **Branch protection** – the default branch requires Gate and Agents Guard to
   report success, plus **Require review from Code Owners**. This combination
   prevents force pushes or merges that sidestep the protections above.

## Allowlisted change reasons and label process
Only the following scenarios justify edits. If your proposal does not fit, open
an ops discussion before touching the files.

- **Security response** – rotating a compromised secret, pinning a patched
  Action version, or patching a supply-chain CVE that affects execution.
- **Reliability fix** – addressing a regression that stops dispatch, branch
  creation, or orchestrator fan-out.
- **Policy maintenance** – updating links, documentation pointers, or metadata
  to keep references accurate after broader restructuring.
- **Platform migration** – adapting to a GitHub Actions deprecation announced by
  GitHub (for example, new runner images or permissions requirements).

When one of these scenarios applies:

1. File or link to the tracking issue describing the incident and expected fix.
2. A maintainer adds the `agents:allow-change` label to the pull request **after**
   confirming the change reason fits the allowlist.
3. Ensure the PR body includes the incident or policy context plus rollback
   steps.
4. Secure Code Owner review before merge; the label does not bypass review.
5. Remove the label once merged so the guardrail resumes full enforcement on
   future PRs.

## Troubleshooting
- **Guard check failing for “missing agents:allow-change label”.** Add the label
  (maintainers only) or revert the workflow edits. The check re-evaluates once
  the label is present.
- **CODEOWNERS review still required.** Ping the maintainer group listed in
  `.github/CODEOWNERS`. Draft reviews do not satisfy branch protection.
- **Ruleset rejection on push.** Confirm you are operating on a maintainer-owned
  branch or request a maintainer to apply a temporary bypass while they assist
  with the change.
- **Unexpected dispatch failures post-merge.** Review the orchestrator run in
  Actions → `Agents 70 Orchestrator`. Most issues stem from missing
  permissions or mismatched input contracts introduced in the edit.

Document any exception in the linked issue so future investigations have a
single source of truth.

## Quick ruleset verification

- **UI path** – `Settings → Code security and analysis → Rulesets → Tests Pass to
  Merge`. Confirm the ruleset is `Active`, then expand the `Restrict file
  updates` section to verify the three workflow paths, with both "Block
  deletions" and "Block renames" toggled on.
- **API check** – run

  ```bash
  curl -s "https://api.github.com/repos/stranske/Trend_Model_Project/rulesets/7880490" \
    | jq '{name, enforcement, scope: .conditions.ref_name.include,
           file_rules: [(.rules[] | select(.type == "restrict_file_updates"))]
           }'
  ```

  The JSON must report `"enforcement": "active"` and a `file_rules` array whose
  `parameters` include the three workflow paths with `block_deletions` and
  `block_renames` set to `true`. If the ruleset is disabled or missing the file
  rules, coordinate with a repository admin to restore protection before
  running destructive push tests. Verification output is logged in
  `docs/ci/agents_ruleset_verification.md`.
