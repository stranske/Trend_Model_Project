# Code Ownership and Review Expectations

This repository uses a single CODEOWNERS routing profile to keep reviews fast
and predictable. The current mappings are intentionally coarse grained so that
any change touching the critical execution paths automatically notifies the core
maintainer.

## Ownership Map

| Path Pattern        | Owners      | Notes |
|---------------------|-------------|-------|
| `/src/**`           | `@stranske` | Core engine and model code. |
| `/tests/**`         | `@stranske` | Unit and integration coverage for the engine. |
| `/.github/**`       | `@stranske` | Automation, workflows, and policy configuration. |
| `/agents/**`        | `@stranske` | Agent bootstrap files and task playbooks. |
| `*` (fallback)      | `@stranske` | Any file not captured by the patterns above. |

## Review and Auto-Merge Workflow

- **Pull Request reviews** – Whenever a PR touches one of the paths above the
  listed owner is automatically requested as a reviewer. This keeps code reviews
  aligned with the areas of expertise documented here.
- **Low-risk lanes** – Once the CODEOWNER has approved, low-risk changes that
  satisfy repository checks are eligible for auto-merge. No additional manual
  steps are required.
- **Shared context** – Contributors should reference this document when opening
  PRs so they know which maintainer will be looped in and what areas are
  considered high-priority for review.

If ownership ever needs to expand beyond a single maintainer, update both this
file and `.github/CODEOWNERS` in the same commit so the documentation and routing
logic stay synchronized.
