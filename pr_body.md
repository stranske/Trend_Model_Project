<!-- pr-preamble:start -->
> **Source:** Issue #4323

<!-- pr-preamble:end -->

## Scope
Harden tool execution by enforcing sandboxed paths, structured ToolResult outputs, JSON logging with redaction, and configurable rate limiting.

## Tasks
- [x] Enhance the file sandboxing function to reject any file paths that resolve to directories outside the allowed list. This includes validating against absolute paths, relative paths containing '..' components, and ensuring symlink resolution is performed prior to the sandbox check.
- [x] Implement the ToolResult class as a Pydantic model with standardized fields (e.g., status, message, and data) and refactor all tool operations to return an instance of this model.
- [x] Update the logging functionality to output logs in a validated JSON format. Ensure that each log entry contains required fields such as timestamp, parameters (with any sensitive data redacted), output summary, and request ID.
- [x] Implement a rate limiting mechanism within the tool call workflow. The mechanism should use configurable limits and actively prevent runaway loops by halting execution or throwing an error when limits are exceeded. Include tests to simulate excessive calls.
- [x] Design and add comprehensive unit tests for sandbox enforcement that explicitly cover path traversal attacks. Test cases should include attempts to use '..' in paths, absolute paths to locations outside allowed directories, and symlink traversal scenarios.
- [x] Add unit tests to verify that log outputs are valid JSON objects and contain all required fields with correct data types and redactions applied.

## Acceptance Criteria
- [x] The file sandboxing function rejects any file paths that resolve to directories outside the allowed list, including paths with '..' components and symlink resolutions.
- [x] The ToolResult class is implemented as a Pydantic model with fields: status, message, and data. All tool operations return an instance of this model.
- [x] Logs are output in a validated JSON format containing timestamp, parameters (redacted), output summary, and request ID.
- [x] A rate limiting mechanism is implemented that enforces configurable execution limits per tool and halts execution when limits are exceeded.
- [x] Log outputs are verified to be valid JSON objects containing all required fields with correct data types and redactions.
