# Repo Ops Facts — Codex Bootstrap

This page captures the established, validated facts about the Codex bootstrap and workflow wiring in this repository. It is the single source of truth so we don’t keep re‑asking for already‑confirmed details.

Last updated: 2025‑09‑16

## Branches and Event Semantics
- Default branch: `phase-2-dev`.
- Issue label events execute workflows from the default branch.
- We do not reuse existing PR branches for fixes; each run creates a fresh branch from default.

## Trigger Labels
- Primary label: `agent:codex` (case‑sensitive).
- Alias supported: `agents:codex` (legacy).

## PR Hygiene and Activation
- PRs are created as non‑draft by default.
- The first PR comment includes `@codex start`.
- PRs and the source issue are auto‑assigned to: `chatgpt-codex-connector`, `stranske-automation-bot` (best‑effort; failures are logged as warnings).
- PRs are labeled with `agent:codex`.

## Tokens, Permissions, and Secrets
- Preferred token: `SERVICE_BOT_PAT` (scopes: `repo`, `workflows`, `pull_requests`, `issues`).
- Fallback token: `GITHUB_TOKEN` (requires Actions → Workflow permissions set to “Read and write”).
- The workflow is safe without referencing `secrets.*` inside step‑level `if:` conditions; token decisions are passed via `env`/inputs.

## Workflows and Actions
- Bridge workflow: `.github/workflows/codex-issue-bridge.yml`.
  - Triggers on Issues: `opened`, `labeled`, `reopened`, and `workflow_dispatch` with optional `test_issue`.
  - Safe checkout of default branch before performing operations.
  - Prefers local composite `.github/actions/codex-bootstrap-lite`; falls back to an inline path if the composite fails.
- Composite action: `.github/actions/codex-bootstrap-lite/action.yml`.
  - PAT‑first, fallback to `GITHUB_TOKEN` if allowed.
  - Creates a new branch with deterministic prefix `agents/codex-issue-<num>-<runid>`.
  - Seeds a bootstrap file under `agents/` and opens a non‑draft PR.
  - Labels/assigns and posts `@codex start` in PR comments.

## Diagnostic Guarantees
- Event summary step logs the action, label, and issue number at run start.
- On failure, the run logs indicate whether branch creation, PR creation, labeling, assignment, or commenting failed and with which token.

## Known Good Behaviours (Confirmed)
- PR #1025 merged: the fixed bridge workflow is present on `phase-2-dev`.
- Label path is dual‑trigger safe and avoids undefined contexts when dispatched.
- Safe checkout prevents missing‑ref errors.

## Expectations for Future Work
- Keep this page up to date if labels, assignees, or token policy changes.
- If org rules change, document the new constraints here (e.g., PAT usage restrictions).
