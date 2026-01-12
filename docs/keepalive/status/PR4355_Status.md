# Keepalive Status — PR #4355

> **Status:** In progress — tasks remaining.

## Progress updates
- Round 1: Added side-by-side YAML diff rendering with HTML diff table, included raw YAML with syntax highlighting, and verified via `tests/app/test_model_page_helpers.py`.

## Blockers
- None noted.

## Scope
Enable interactive config changes in the Streamlit Model page with preview diffs, apply/apply+run actions, safety checks, and change history.

## Tasks
- [x] Add "Config Chat" panel to Streamlit layout:
- [x] Sidebar or expandable section
- [x] Text input for instructions
- [x] Send button
- [x] Implement config viewer component:
- [x] Show current config summary (key sections)
- [x] Collapsible sections for detail
- [x] Highlight recently changed values
- [x] Implement diff preview component:
- [x] Side-by-side or unified diff view
- [x] Syntax highlighting for YAML
- [x] Color-coded additions/removals
- [x] Add action buttons:
- [x] "Preview" - Show diff without applying
- [x] "Apply" - Apply patch to session config
- [x] "Apply + Run" - Apply (verify: confirm completion in repo)
- [x] execute analysis (verify: confirm completion in repo)
- [ ] "Revert" - Undo last change
- [ ] Implement safety confirmations:
- [ ] Modal dialog for risky changes
- [x] Show risk flags and explanations (verify: confirm completion in repo)
- [ ] Require explicit confirmation
- [x] Add loading states:
- [x] Spinner during LLM call
- [ ] Progress bar for analysis
- [ ] Estimated time remaining
- [ ] Implement change history:
- [ ] List of recent changes
- [ ] Click to revert to any point
- [ ] Show diffs between versions
- [x] Add error display:
- [ ] Validation errors in red
- [ ] Suggestions for fixes
- [ ] Link to relevant docs

## Acceptance criteria
- [ ] User can enter NL instruction and see preview diff
- [ ] User can apply changes without running
- [ ] User can apply and run in one action
- [ ] Risky changes show warning dialog
- [ ] Validation errors are displayed clearly
- [ ] Changes can be reverted
- [ ] UI is responsive during LLM calls
