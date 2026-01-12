# Template Setup Guide for Workflow Adoption

This guide explains how to adopt the automation workflows from this repository in other similar projects. It highlights variables, secrets, and optional features you can customize.

Last updated: 2025-09-17

## 1. Repository Variables (Settings → Variables → Actions)

Set these variables to tailor behavior (all have sensible defaults if omitted):

- `AUTOMERGE_LABEL` (default: `automerge`)
- `RISK_LABEL` (default: `risk:low`)
- `AGENT_FROM_LABEL` (default: `from:copilot`)
- `AGENT_FROM_LABEL_ALT` (default: `from:codex`)
- `AGENT_LABEL` (default: `agent:copilot`)
- `AGENT_LABEL_ALT` (default: `agent:codex`)
- `AUTOFIX_LABEL` (always `autofix:clean`; add this label to PRs to enable the Gate summary autofix routines)
- `APPROVE_PATTERNS` (default: `src/**,docs/**,tests/**,**/*.md`)
- `MAX_LINES_CHANGED` (default: `1000`)
- `CI_PY_VERSIONS` (default: `["3.11","3.12"]`)
- `COV_MIN` (default: `85`)
- `REGISTRY` (default: `ghcr.io`)
- `IMAGE_NAME` (default: `<owner>/trend-model`) – override per project
- `HEALTH_PORT` (default: `8000`)
- `HEALTH_PATH` (default: `/health`)

Tip: you can manage these centrally using repo templates or org-level defaults.

## 2. Secrets

### GitHub Apps (Rate Limit Pools)

GitHub Apps have separate rate limit pools (5000 req/hr each). Configure multiple apps to increase effective capacity:

- `WORKFLOWS_APP_ID` / `WORKFLOWS_APP_PRIVATE_KEY`: Primary app for autofix-loop
- `GH_APP_ID` / `GH_APP_PRIVATE_KEY`: Secondary app for issue-intake, autofix, bot-comment-handler
- `KEEPALIVE_APP_ID` / `KEEPALIVE_APP_PRIVATE_KEY`: Dedicated app for keepalive-loop (isolated pool)

**Current distribution:**
| App | Rate Pool | Workflows |
|-----|-----------|-----------|
| KEEPALIVE_APP | 5000/hr | keepalive-loop |
| WORKFLOWS_APP | 5000/hr | autofix-loop |
| GH_APP | 5000/hr | issue-intake, autofix, bot-comment-handler |

To add additional apps:
1. Create a new GitHub App at https://github.com/settings/apps/new
2. Grant permissions: `contents: write`, `pull_requests: write`, `actions: write`
3. Install on your repository
4. Add `KEEPALIVE_APP_ID` and `KEEPALIVE_APP_PRIVATE_KEY` secrets
5. Update `agents-keepalive-loop.yml` to use `KEEPALIVE_APP_*` instead of `WORKFLOWS_APP_*`

### Personal Access Tokens

- `OWNER_PR_PAT` (optional): Personal Access Token to author PRs as a human (Codex bootstrap create-mode).
- `SERVICE_BOT_PAT` (optional): Service bot token for labeling/assignment when needed across forks. **This is the primary PAT fallback for rate limit resilience** - ensure it's from a separate account.
- `GITHUB_TOKEN`: Provided by GitHub Actions – ensure it has Read/Write permissions in repo settings (Actions → General).

### Rate Limit Architecture

Each token type has an independent rate limit pool:
- GitHub App installations: 5000/hr per app
- PATs: 5000/hr per user account
- GITHUB_TOKEN: 1000/hr per repository (shared across all workflow runs)

**Combined capacity:** Up to 15,000 requests/hour across all three app pools when all tokens are configured. Actual effective capacity depends on request distribution across workflows.

The keepalive workflow automatically falls back to `SERVICE_BOT_PAT` when the GitHub App token generation fails (e.g., rate limited, not installed, or misconfigured).

Security posture: The `pull_request_target` workflows in this template do not checkout or execute fork code. They only mutate labels/approvals using base-repo context.

## 3. Branch and Trigger Notes

- Default branch is auto-resolved where needed; update branch filters if your default branch isn’t `main`.
- CI runs on `pull_request` and `push` to default branch.
- Docker pushes only on `push` to the default branch (adjust if your flow differs).

## 4. Autofix (fork-friendly)

- Same-repo PRs: fixes are committed & pushed automatically.
- Fork PRs: an `autofix.patch` artifact is uploaded and a PR comment explains how to apply locally:
  ```bash
  git am < autofix.patch
  git push origin HEAD:<branch>
  ```
- Composite action runs `ruff`, `black`, `isort`, `docformatter`, and optionally `scripts/auto_type_hygiene.py` when present.

## 5. Manual Review + Cosmetic Repair

- Automated merge-manager flows were retired with Issue #2190. Approvals and merges are now handled manually once Gate succeeds.
- `maint-45-cosmetic-repair.yml` provides an optional manual helper to re-run pytest, apply formatting fixes via
  `scripts/ci_cosmetic_repair.py`, and open a labelled repair PR when hygiene updates are required.
- Keep the `ci:green` label in sync with the latest Gate status manually; the Gate summary job surfaces aggregated results and will
  highlight mismatches in its summary comment.

## 6. Docker Workflow

- Customize container registry, image, port, and health endpoint via variables.
- The workflow builds, tests, smoke-tests, and pushes the image on branch pushes.

## 7. Known Safe Defaults / Optional Modules

- Guard: No PR branch reuse – prevents accidental branch reuse across merged PRs.
- Diagnostics (preflight/verify) – optional; enable as needed.

## 8. Adoption Checklist

1. Create repository variables listed in Section 1.
2. Add optional secrets in Section 2 (or skip if not needed).
3. Verify default branch in workflows; adjust branch filters.
4. If using Docker, set `IMAGE_NAME` and verify `HEALTH_*`.
5. Run a small test PR to confirm CI and autofix flows.

## 9. Troubleshooting

- Merge automation removed:
  - Issue #2190 retired `.github/workflows/maint-45-merge-manager.yml`; rely on manual reviews or add a custom workflow if automation is reintroduced.
- Labels not auto-applied on fork PRs:
  - The labeler workflow was removed in Issue #2190; apply labels manually or add a bespoke workflow if needed.
- Autofix doesn’t push on forks:
  - This is intentional – download the patch artifact and apply locally.

---

For updates, see `docs/ops/codex-bootstrap-facts.md` and this file’s history.
