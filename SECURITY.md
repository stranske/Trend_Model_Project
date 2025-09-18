# Security Policy

## Reporting a Vulnerability

If you discover a security issue, please email security@example.com.
We will respond as soon as possible.

## pull_request_target Hardening (Issue #1140)

The repository uses `pull_request_target` only for narrowly scoped automation that:
1. Does not build or execute contributor code.
2. Performs metadata actions (labeling, guarded auto-approval) with least privileges.
3. Restricts checkout to a single JSON allowlist/rules file via sparse checkout.
4. Sets `persist-credentials: false` to prevent unintended token reuse.

Defensive assertion steps fail the workflow if sparse checkout unexpectedly expands.
All build/test operations run in standard `pull_request` workflows without elevated secrets.

Any change introducing code execution in these workflows requires prior security review.
