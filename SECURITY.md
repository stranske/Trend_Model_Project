# Security Policy

## Reporting a Vulnerability

If you discover a security issue, please email security@example.com.
We will respond as soon as possible.

## Security Controls

The Trend Model Project implements multiple layers of security controls to protect against unauthorized changes and ensure data integrity. All configuration changes that could affect portfolio behavior are subject to a centralized risk guard middleware that intercepts potentially dangerous operations before they reach the application layer. This middleware validates that high-risk changes—such as removing validation constraints, increasing leverage limits, or modifying broad-scope settings—require explicit confirmation from the caller. Additionally, the API enforces strict type checking on all request payloads, rejecting malformed or unexpected input types with informative 400-level error responses. Authentication and authorization are handled at the infrastructure level, with all API endpoints requiring valid credentials before processing requests.

## Potential Vulnerabilities

Configuration injection represents the primary attack surface for this application, where malicious actors could attempt to modify portfolio constraints or validation rules through crafted API requests. Without proper guards, an attacker could disable turnover limits, remove volatility caps, or inject unknown configuration keys that bypass validation. Another potential vulnerability exists in the narrative generation pipeline, where dynamically constructed text could inadvertently include forward-looking statements that violate compliance requirements. The `pull_request_target` workflow trigger also presents a risk if not properly hardened, as it runs with elevated privileges and could potentially execute untrusted code from forked repositories.

## Mitigation Strategies

To mitigate configuration injection risks, the application employs a deny-by-default approach where all high-risk patch operations are blocked unless the caller explicitly sets `confirm_risky=True`. The risk detection logic identifies constraint removals, leverage increases, validation disables, broad-scope edits, and unknown keys, flagging them for mandatory confirmation. Unknown or ambiguous configuration keys trigger the `needs_review` flag and provide close-match suggestions to help users identify typos. For narrative generation, a deterministic scanner checks all output text against a predefined list of forbidden forward-looking phrases before delivery. The `pull_request_target` workflows are hardened through sparse checkout restrictions, credential persistence disabling, and defensive assertions that fail the workflow if unexpected files are checked out. All security-sensitive changes require prior review before merging.

## pull_request_target Hardening (Issue #1140)

The repository uses `pull_request_target` only for narrowly scoped automation that:
1. Does not build or execute contributor code.
2. Performs metadata actions (labeling, guarded auto-approval) with least privileges.
3. Restricts checkout to a single JSON allowlist/rules file via sparse checkout.
4. Sets `persist-credentials: false` to prevent unintended token reuse.

Defensive assertion steps fail the workflow if sparse checkout unexpectedly expands.
All build/test operations run in standard `pull_request` workflows without elevated secrets.

Any change introducing code execution in these workflows requires prior security review.
