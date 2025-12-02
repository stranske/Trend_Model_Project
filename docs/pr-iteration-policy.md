## PR Iteration Policy

Date: 2025-09-20

Requirement: Each logical batch of automation / workflow hygiene changes MUST be raised as a **new pull request** after the previous PR merges. Branches used for merged PRs are deleted; do **not** reopen or force-push old branches.

Guidelines:
1. Start from up-to-date `main` (`git fetch origin && git checkout main && git pull`).
2. Create a fresh branch named with scope and date, e.g. `ci/automation-tidy-20250920`.
3. Commit focused changes only (avoid mixing unrelated refactors).
4. Open PR with concise summary: context, changes, risk, rollback.
5. After merge, delete branch (allow GitHub auto-delete) and begin next iteration from updated `main`.

Rationale:
- Guarantees clean diff review surfaces.
- Avoids stale branch state & orphaned workflow_run triggers.
- Ensures linear audit trail for compliance & governance docs.

Enforcement Aids (optional future work):
- Add a lightweight action to fail if a PR branch matches a previously merged branch name hash (currently archived guard replaced by policy documentation).

---
Maintainer Note: Update this file if policy changes or automation enforcement is reintroduced.