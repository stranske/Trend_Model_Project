# Code Ownership & Review Expectations

This document complements `.github/CODEOWNERS` and clarifies responsibilities and review flow.

## Coverage Summary
The repository uses a single primary owner presently:

| Path Group | Pattern(s) | Owner(s) |
|------------|------------|----------|
| Core library & engine | `/src/**` | @stranske |
| Streamlit UI & app layer | `/streamlit_app/**`, `/app/**` | @stranske |
| Configuration & demos | `/config/**`, `/demo/**` | @stranske |
| Assets | `/assets/**` | @stranske |
| Agents automation specs | `/agents/**` | @stranske |
| Tests | `/tests/**` | @stranske |
| Tooling & scripts | `/scripts/**`, `/tools/**`, `/.devcontainer/**` | @stranske |
| Documentation & notebooks | `/docs/**`, `/notebooks/**`, `/Old/**`, `/*.ipynb` | @stranske |
| CI/CD & workflows | `/.github/workflows/**`, `Dockerfile`, `docker-compose.yml` | @stranske |

## Objectives
1. Route PRs automatically to maintainers for required review.
2. Enable safe auto‑merge for low‑risk lanes once reviewed.
3. Provide a transparent process for proposing changes to ownership.

## Review SLAs (Guidelines)
| Risk Label | Target First Response | Merge Eligibility |
|------------|-----------------------|-------------------|
| `risk:low` | < 24h | Auto‑merge after required approval and status checks green. |
| `risk:medium` | < 48h | Manual merge by owner after deeper review. |
| `risk:high` | < 72h | Explicit owner approval; may require follow‑up design notes. |

## Auto‑Merge Policy
Low‑risk PRs (`risk:low` + `automerge` label) may be auto‑merged by automation once:
- All required status checks pass.
- At least one CODEOWNER approval is present (if branch protection requires it) OR no additional approvers are mandated by protection rules.

## Adding / Changing Ownership
1. Open a PR modifying `.github/CODEOWNERS` and this document.
2. Provide rationale (bus factor, domain expertise, load balancing).
3. Ping existing owner(s) for explicit acknowledgement in PR comments.

## Escalation
If an urgent fix awaits review beyond SLA, mention the owner directly or (if established) escalate via a designated label (e.g., `needs:expedite`).

## Future Evolution
- Introduce secondary owners for core directories as contributors become regular maintainers.
- Add a health check that verifies every top-level directory has a CODEOWNERS match.

_Last updated: 2025-09-19 (implements Issue #1203)_
