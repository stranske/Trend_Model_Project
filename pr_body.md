<!-- pr-preamble:start -->
> **Source:** Issue #4180

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
The NL layer must output a structured, validated format—not freeform text that gets parsed with regex. A typed `ConfigPatch` schema gives us:
- Predictable model outputs that can be validated
- Clear operation semantics (set/remove/append/merge)
- Risk flagging for dangerous changes
- Audit trail for what was changed and why

#### Tasks
- [x] Design patch operation types:
- [x] - `set` - Set a value at a path (create if missing)
- [x] - `remove` - Delete a key at a path
- [x] - `append` - Add item to a list
- [x] - `merge` - Deep merge a dict at a path
- [x] Define `PatchOperation` Pydantic model:
- [x] ```python
- [x] class PatchOperation(BaseModel):
- [x] op: Literal["set", "remove", "append", "merge"]
- [x] path: str  # JSONPointer or dotpath
- [x] value: Any | None = None  # Required for set/append/merge
- [x] rationale: str | None = None  # LLM's explanation
- [x] ```
- [x] Define `ConfigPatch` Pydantic model:
- [x] ```python
- [x] class ConfigPatch(BaseModel):
- [x] operations: list[PatchOperation]
- [x] risk_flags: list[RiskFlag] = []
- [x] summary: str  # Human-readable summary of changes
- [x] ```
- [x] Define `RiskFlag` enum/model for dangerous changes:
- [x] - `REMOVES_CONSTRAINT` - Removing position/turnover limits
- [x] - `INCREASES_LEVERAGE` - Vol target above threshold
- [x] - `REMOVES_VALIDATION` - Disabling safety checks
- [x] - `BROAD_SCOPE` - Changes affect many keys
- [x] Add JSON Schema export method for the patch format
- [x] Write comprehensive unit tests

#### Acceptance criteria
- [x] `ConfigPatch` model validates correct patches
- [x] `ConfigPatch` model rejects malformed patches with clear errors
- [x] JSON Schema can be exported for LLM prompting
- [x] Risk flags are populated for dangerous operations
- [x] Unit tests cover:
- [x] - All operation types
- [x] - Path validation (valid/invalid paths)
- [x] - Risk flag detection
- [x] - Edge cases (empty patch, nested paths, list operations)
- [x] - `set` operations with null values
- [x] - `remove` operations with explicit null values

<!-- auto-status-summary:end -->
