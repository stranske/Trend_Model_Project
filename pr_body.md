<!-- pr-preamble:start -->
> **Source:** Issue #4323

<!-- pr-preamble:end -->

## Scope
Harden tool execution by enforcing sandboxed paths, structured ToolResult outputs, JSON logging with redaction, and configurable rate limiting.

## Tasks
- [x] Enforce sandbox path resolution (block traversal and symlink escapes) before tool execution.
- [x] Implement `ToolResult` as a Pydantic model with `status`, `message`, `data`, and `elapsed_ms` fields; return it from tool operations.
- [x] Emit structured JSON logs for tool calls with redacted sensitive parameters and output summaries.
- [x] Add rate limiting per tool with configurable limits and tests for exceeded limits.
- [x] Add unit tests covering sandbox traversal, symlink escapes, JSON logging, and rate limit enforcement.

## Acceptance Criteria
- [x] Tool sandbox rejects paths outside allowed roots, including traversal components and symlinks.
- [x] Tool operations return `ToolResult` with `status`, `message`, `data`, and `elapsed_ms` fields.
- [x] Logs are emitted as valid JSON with timestamp, request_id, tool name, redacted parameters, output summary, and status.
- [x] Rate limiting halts execution when limits are exceeded and is configurable per tool.
- [x] Unit tests cover sandbox enforcement, symlink traversal, JSON log validation, and rate limit exceedance.
