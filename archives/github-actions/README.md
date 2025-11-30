# Archived GitHub Actions

Composite actions that are no longer referenced by any active workflow.

## Archive Structure

```
archives/github-actions/
├── YYYY-MM-DD-action-name/    # Archived action directory
│   └── action.yml
└── README.md
```

## Archived on 2025-11-30

### `apply-autofix`
- **Purpose**: Wrapper that ran `autofix` action then committed/pushed changes
- **Reason**: Not used by any workflow; `reusable-18-autofix.yml` handles the full autofix loop directly
- **Dependencies**: Used `autofix` action internally

### `autofix-commit-push`  
- **Purpose**: Similar to `apply-autofix` but with simpler commit message handling
- **Reason**: Not used by any workflow; duplicate of `apply-autofix` functionality
- **Dependencies**: Used `autofix` action internally

### `update-residual-history`
- **Purpose**: Appended autofix residual classification to `ci/autofix/history.json`
- **Reason**: Not used by any workflow; residual tracking was removed from autofix flow
- **Dependencies**: Required `scripts/update_residual_history.py` (also unused)

### `codex-bootstrap`
- **Purpose**: Original verbose Codex bootstrap action with complex fallback logic
- **Reason**: Replaced by `codex-bootstrap-lite` which is simpler and actively used
- **Superseded by**: `.github/actions/codex-bootstrap-lite/`

## Active Actions (kept in `.github/actions/`)

| Action | Used By | Purpose |
|--------|---------|---------|
| `autofix` | `build-pr-comment`, archived wrappers | Core formatting action (ruff, black, isort) |
| `build-pr-comment` | `reusable-18-autofix.yml` | Builds PR comment from autofix results |
| `codex-bootstrap-lite` | `reusable-16-agents.yml` | Minimal Codex PR bootstrap |
| `signature-verify` | `health-43-ci-signature-guard.yml` | Verify CI signature files |
