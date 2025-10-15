# Agents Workflow Protection Policy

The agents orchestrator and bridge workflows are contract-critical automation.
They must remain available unless a maintainer deliberately overrides the
protections described below.

## Covered files

- `.github/workflows/agents-63-chatgpt-issue-sync.yml`
- `.github/workflows/agents-63-codex-issue-bridge.yml`
- `.github/workflows/agents-70-orchestrator.yml`

## Protection layers

1. **CODEOWNERS gate** – the files above are owned by `@stranske`. Pull
   requests cannot merge changes to them without Code Owner approval. In the
   default branch protection rule, enable **Require review from Code Owners** so
   GitHub enforces the restriction automatically.
2. **Repository ruleset** – create (or update) a ruleset that targets the
   default branch and blocks deletions or renames of the covered workflows. Use
   the protected file patterns listed above and set the bypass mode to
   *Maintainers only* so emergencies still require a deliberate maintainer
   action. Administrators can provision the ruleset through the repository
   settings (**Settings → Code security and analysis → Rulesets → New ruleset →
   Branch**). Capture screenshots of the configuration in the incident log so
   auditors can confirm the guardrail is in place.
3. **CI guardrail** – the `Agents Critical Guard` check (see
   `.github/workflows/agents-critical-guard.yml`) fails any pull request that
   deletes or renames a protected file. The check documents override steps and
   prevents merges unless a maintainer explicitly bypasses the branch protection
   rule. Mark this status check as **required** on the default branch so the
   enforcement cannot be skipped accidentally.

## Emergency override procedure

1. Confirm the incident really requires editing or temporarily removing one of
   the protected workflows.
2. A maintainer with admin access temporarily adjusts the repository ruleset
   (or applies a bypass) via **Settings → Code security and analysis → Rulesets**
   and, if required, toggles the `Agents Critical Guard` status check in branch
   protection.
3. Apply and review the change in a dedicated pull request. Code Owner approval
   remains required even when a maintainer performs the edits.
4. Re-enable the ruleset block and restore the CI guard immediately after the
   change merges. Document the incident and restoration steps in the associated
   issue or runbook entry.

## Verification checklist

- The default branch protection lists **Require review from Code Owners** and
   includes the files above. The `Agents Critical Guard` status check is marked
   as required.
- The repository ruleset shows the three workflows in its “Protected file
  patterns” section with **Block deletion** and **Block rename** enabled.
- A maintainer can describe the override procedure without referencing this
  document (spot check during ops reviews).
