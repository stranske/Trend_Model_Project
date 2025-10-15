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
   requests cannot merge changes to them without Code Owner approval. The default
   branch protection must also enable **Require review from Code Owners** so the
   rule is enforced.
2. **Repository ruleset** – create (or update) a ruleset that targets the
   default branch and blocks deletions or renames of the covered workflows. Set
   the bypass mode to *Maintainers only* so that emergencies still require a
   deliberate maintainer action.

## Emergency override procedure

1. Confirm the incident really requires editing or temporarily removing one of
   the protected workflows.
2. A maintainer with admin access disables the repository ruleset’s block or
   grants themselves a temporary bypass via the repository settings
   (**Settings → Code security and analysis → Rulesets**).
3. Apply and review the change in a dedicated pull request. Code Owner approval
   remains required even when a maintainer performs the edits.
4. Re-enable the ruleset block immediately after the change merges. Document the
   incident and restoration steps in the associated issue or runbook entry.

## Verification checklist

- The default branch protection lists **Require review from Code Owners** and
  includes the files above.
- The repository ruleset shows the three workflows in its “Protected file
  patterns” section with **Block deletion** and **Block rename** enabled.
- A maintainer can describe the override procedure without referencing this
  document (spot check during ops reviews).
