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
- [ ] - `set` - Set a value at a path (create if missing)
- [ ] - `remove` - Delete a key at a path
- [ ] - `append` - Add item to a list
- [ ] - `merge` - Deep merge a dict at a path
- [ ] Define `PatchOperation` Pydantic model:
- [ ] ```python
- [ ] class PatchOperation(BaseModel):
- [ ] op: Literal["set", "remove", "append", "merge"]
- [ ] path: str  # JSONPointer or dotpath
- [ ] value: Any | None = None  # Required for set/append/merge
- [ ] rationale: str | None = None  # LLM's explanation
- [ ] ```
- [ ] Define `ConfigPatch` Pydantic model:
- [ ] ```python
- [ ] class ConfigPatch(BaseModel):
- [ ] operations: list[PatchOperation]
- [ ] risk_flags: list[RiskFlag] = []
- [ ] summary: str  # Human-readable summary of changes
- [ ] ```
- [ ] Define `RiskFlag` enum/model for dangerous changes:
- [ ] - `REMOVES_CONSTRAINT` - Removing position/turnover limits
- [ ] - `INCREASES_LEVERAGE` - Vol target above threshold
- [ ] - `REMOVES_VALIDATION` - Disabling safety checks
- [ ] - `BROAD_SCOPE` - Changes affect many keys
- [x] Add JSON Schema export method for the patch format
- [x] Write comprehensive unit tests

#### Acceptance criteria
- [ ] `ConfigPatch` model validates correct patches
- [ ] `ConfigPatch` model rejects malformed patches with clear errors
- [ ] JSON Schema can be exported for LLM prompting
- [x] Risk flags are populated for dangerous operations
- [x] Unit tests cover:
- [ ] - All operation types
- [ ] - Path validation (valid/invalid paths)
- [ ] - Risk flag detection
- [ ] - Edge cases (empty patch, nested paths, list operations)

<!-- auto-status-summary:end -->
