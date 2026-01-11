# Keepalive Status — PR #4339

> **Status:** In progress — tasks remaining.

## Progress updates
- Round 1: Added structured NL operation log model and log persistence helpers with tests.
- Round 2: Added optional LangSmith tracing helpers, wired NL invocations, and documented setup.
- Round 3: Added LangSmith trace runs for NL calls/replay with tests.
- Round 4: Propagated NL request IDs through apply/validate/run logging and added CLI log coverage.

## Scope
Add NL operation observability to capture prompts, outputs, and replay diagnostics.

## Tasks
### Define Structured Log Format for NL Operations
- [x] Define a structured log format for NL operations:
- [x] ```python
- [x] class NLOperationLog(BaseModel):
- [x] request_id: str
- [x] timestamp: datetime
- [x] operation: str  # "nl_to_patch", "apply_patch", "validate", "run"
- [x] input_hash: str  # Hash of inputs for dedup
- [x] prompt_template: str
- [x] prompt_variables: dict
- [x] model_output: str | None
- [x] parsed_patch: ConfigPatch | None
- [x] validation_result: ValidationResult | None
- [x] error: str | None
- [x] duration_ms: float
- [x] model_name: str
- [x] temperature: float
- [x] token_usage: dict | None
- [x] ```
### Implement Log Persistence
- [x] Write logs to `.trend_nl_logs/` directory.
- [x] Create one log file per day with the format `nl_ops_<date>.jsonl`.
- [x] Implement log rotation to keep only the last 30 days of logs.
### Add Replay Capability
- [x] Implement `trend nl replay <log_file> --entry <n>`:
- [x] - Re-run the exact prompt.
- [x] - Compare outputs.
### Implement Optional LangSmith Tracing
- [x] Check for `LANGSMITH_API_KEY`.
- [x] Enable tracing if the key is present.
- [x] Send traces to LangSmith.
- [x] Document LangSmith setup in the README.
### Add Request Correlation
- [x] Generate `request_id` at the entry point.
- [x] Pass `request_id` through all operations.
- [x] Include `request_id` in all logs and errors.
### Implement Log Analysis Tools
- [x] Implement `trend nl logs` to list recent operations.
- [x] Implement `trend nl logs --failures` to show only failures.
- [x] Implement `trend nl logs --stats` to display summary statistics.

## Acceptance criteria
- [x] All NL operations are logged with a structured format.
- [x] A failed run can be reproduced from logged artifacts.
- [x] Logs include specific fields such as `request_id`, `timestamp`, `operation`, and `error` to provide enough context for debugging.
- [x] Old logs are automatically rotated, keeping only the last 30 days.
- [ ] LangSmith integration successfully transmits traces and verifies trace visibility in the LangSmith dashboard.
- [x] CLI tools (`trend nl logs`, `trend nl logs --failures`, `trend nl logs --stats`) function as expected.
