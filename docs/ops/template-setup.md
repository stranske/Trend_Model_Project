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
- `AUTOFIX_OPT_IN_LABEL` (default: `autofix`)
- `APPROVE_PATTERNS` (default: `src/**,docs/**,tests/**,**/*.md`)
- `MAX_LINES_CHANGED` (default: `1000`)
- `CI_PY_VERSIONS` (default: `["3.11","3.12"]`)
- `COV_MIN` (default: `80`)
- `REGISTRY` (default: `ghcr.io`)
- `IMAGE_NAME` (default: `<owner>/trend-model`) – override per project
- `HEALTH_PORT` (default: `8000`)
- `HEALTH_PATH` (default: `/health`)

Tip: you can manage these centrally using repo templates or org-level defaults.

## 2. Secrets

- `OWNER_PR_PAT` (optional): Personal Access Token to author PRs as a human (Codex bootstrap create-mode).
- `SERVICE_BOT_PAT` (optional): Service bot token for labeling/assignment when needed across forks.
- `GITHUB_TOKEN`: Provided by GitHub Actions – ensure it has Read/Write permissions in repo settings (Actions → General).

Security posture: The `pull_request_target` workflows in this template do not checkout or execute fork code. They only mutate labels/approvals using base-repo context.

## 3. Branch and Trigger Notes

- Default branch is auto-resolved where needed; update branch filters if your default branch isn’t `phase-2-dev`.
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

## 5. Auto-approve & Enable Auto-merge

- Auto-approve and enable auto-merge are gated by labels and file-change constraints.
- Customize labels via variables above. Adjust approvable file patterns and size cap via `APPROVE_PATTERNS` and `MAX_LINES_CHANGED`.
- Auto-merge uses squash by default; can be changed in workflow inputs if desired.

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
5. Run a small test PR to confirm labeling, auto-approve, enable auto-merge, and autofix flows.

## 9. Troubleshooting

- Auto-merge step fails with branch context error:
  - Ensure `enable-automerge.yml` passes `pull-request-number` (template does).
- Labels not applied on fork PRs:
  - Ensure `label-agent-prs.yml` exists and `pull_request_target` triggers are enabled.
- Autofix doesn’t push on forks:
  - This is intentional – download the patch artifact and apply locally.

---

For updates, see `docs/ops/codex-bootstrap-facts.md` and this file’s history.
