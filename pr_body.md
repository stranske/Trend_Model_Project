<!-- pr-preamble:start -->
> **Source:** Issue #4186

<!-- pr-preamble:end -->

## Scope
Add path traversal validation to config path resolution.

## Changes
- Updated path validation to check resolved paths stay within project boundaries
- Added tests for path traversal detection

## Testing
- All existing tests pass
- New tests verify path traversal is blocked
