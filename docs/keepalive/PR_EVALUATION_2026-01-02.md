# PR Evaluation Report — 2026-01-02

> **Purpose**: Comprehensive evaluation of acceptance criteria completion for PRs #4111, #4112, #4113, #4114 to identify gaps and inform keepalive improvements.

---

## Executive Summary

| PR | Source Issue | Status | AC Complete | Tasks Complete | Assessment |
|----|--------------|--------|-------------|----------------|------------|
| #4112 | #4109 | In Progress | 0/3 | 2/5 | **Partial** — Documentation added but AC not formally checked |
| #4113 | #4110 | In Progress | 1/7 | 6/13 | **Partial** — Script created but workflow/CI not wired |
| #4111 | #4108 | In Progress | 3/5 | 1/5 | **Partial** — Tests added, core tasks incomplete |
| #4114 | #4107 | In Progress | 1/3 | 1/4 | **Good progress** — Implementation done, needs validation |

---

## Detailed PR Evaluations

### PR #4112 — `long_only` constraint documentation (Issue #4109)

#### Tasks Status
| Status | Task |
|--------|------|
| ✅ | Investigate which weighting schemes could produce negative weights |
| ❌ | If none: Add documentation explaining `long_only` context |
| ❌ | If applicable: Verify constraint is applied in those schemes |
| ❌ | Consider removing setting if it's never relevant |
| ✅ | Add appropriate test or documentation |

#### Acceptance Criteria Status
| Status | Criterion |
|--------|-----------|
| ❌ | Setting has clear documented purpose |
| ❌ | If applicable: disabling `long_only` allows negative weights |
| ❌ | If not applicable: Consider removing from UI to reduce confusion |

#### Evidence from PR Changes
1. **UserGuide.md updated** — Added explanation that `long_only` is relevant for custom_weights or `robust_mv` with `min_weight < 0`
2. **Streamlit help text updated** — Clarified when `long_only` matters
3. **New test** — `test_pipeline_long_only_clips_negative_weight_engine_weights` creates a test engine with negative weights
4. **Test for robust_mv** — `test_negative_weights_with_short_min_weight` verifies short weights work

#### Gap Analysis
- **Documentation is present** but AC checkboxes weren't updated in PR body
- The work IS functionally complete:
  - ✅ Setting now has documented purpose (in UserGuide and UI help)
  - ✅ Disabling `long_only` allows negative weights (proven by new test)
  - N/A for removal consideration since it now has clear purpose

**Verdict**: Work appears COMPLETE but PR body wasn't updated. Keepalive failed to extract completion signals from code changes.

---

### PR #4113 — Settings effectiveness evaluation framework (Issue #4110)

#### Tasks Status
| Status | Task |
|--------|------|
| ✅ | Create `scripts/evaluate_settings_effectiveness.py` |
| ❌ | Extracts all settings from `streamlit_app/pages/2_Model.py` |
| ✅ | Defines baseline configuration and meaningful test variations |
| ❌ | Runs paired simulations for each setting |
| ❌ | Computes difference metrics (weights, returns, Sharpe, turnover, etc.) |
| ❌ | Applies statistical tests for significance |
| ❌ | Outputs structured results (JSON/CSV) |
| ❌ | Create `.github/workflows/settings-effectiveness.yml` |
| ✅ | Executes the evaluation script |
| ✅ | Generates markdown report |
| ❌ | Opens/updates tracking issue with results |
| ❌ | Fails CI if effectiveness drops below threshold |
| ✅ | Create `docs/settings/EFFECTIVENESS_REPORT.md` template |
| ✅ | Update `scripts/test_settings_wiring.py` to integrate with framework |
| ✅ | Add mode-aware testing |

#### Acceptance Criteria Status
| Status | Criterion |
|--------|-----------|
| ✅ | Script identifies all settings from UI automatically |
| ❌ | Each setting tested with at least one meaningful variation |
| ❌ | Results categorized as: EFFECTIVE, MODE_SPECIFIC, NO_EFFECT, ERROR |
| ✅ | Report shows overall effectiveness rate (target: >80%) |
| ❌ | Per-category breakdown |
| ❌ | List of non-effective settings with reasons |
| ❌ | Recommendations for each non-effective setting |
| ❌ | Workflow runs successfully in CI |
| ❌ | Threshold-based CI failure when effectiveness drops |
| ❌ | Documentation updated with evaluation methodology |

#### Evidence from PR Changes
1. **740-line script created** — `scripts/evaluate_settings_effectiveness.py` is substantial
2. **Status file created** — `docs/keepalive/status/PR4113_Status.md` tracks progress
3. **Test added** — `tests/scripts/test_evaluate_settings_effectiveness.py`
4. **Many cosmetic fixes** — Lots of whitespace/formatting changes

#### Gap Analysis
- The script exists and is comprehensive (AST parsing, statistical tests, JSON/CSV output)
- However, workflow YAML was NOT created
- The AC checkboxes in PR body don't match actual implementation state
- Several marked ✅ in PR body that aren't reflected in changes

**Verdict**: Significant implementation done, but workflow integration missing. PR body checkboxes INCONSISTENT with actual state.

---

### PR #4111 — Condition threshold & safe_mode fallback (Issue #4108)

#### Tasks Status
| Status | Task |
|--------|------|
| ❌ | Verify `condition_threshold` is checked against actual matrix condition numbers |
| ❌ | Verify `safe_mode` fallback is triggered when threshold exceeded |
| ❌ | Add logging/diagnostics when fallback occurs |
| ❌ | Create test case that triggers the fallback (synthetic ill-conditioned data) |
| ✅ | Add wiring test to verify settings have effect |

#### Acceptance Criteria Status
| Status | Criterion |
|--------|-----------|
| ❌ | Low `condition_threshold` (e.g., 1.0) triggers fallback with realistic data |
| ✅ | Different `safe_mode` values produce different fallback weights |
| ❌ | Diagnostic info indicates when fallback was used |
| ✅ | Settings wiring tests pass |
| ✅ | No regression in existing tests |

#### Evidence from PR Changes
1. **robust_weighting.py modified** — Changed `>` to `>=` for threshold check
2. **Added `fallback_used` diagnostic** — New field in diagnostics dict
3. **test_api_run_simulation.py** — Added tests for robustness config
4. **test_robust_weighting_integration.py** — Added `test_fallback_emits_warning`

#### Gap Analysis
- Tests expect `details["weight_engine_diagnostics"]` with specific keys that may not exist
- Tests expect `details["weight_engine_fallback"]` but this isn't added to result dict
- The implementation adds `fallback_used` to engine diagnostics but doesn't expose it at the API level

**Verdict**: Implementation INCOMPLETE. Tests were added but they likely fail because the API doesn't expose the diagnostic info they expect.

---

### PR #4114 — min_tenure_periods implementation (Issue #4107)

#### Tasks Status
| Status | Task |
|--------|------|
| ❌ | Trace setting from UI through Config to exit logic |
| ✅ | Track holding duration per fund (via `holdings_tenure` dict) |
| ✅ | Block exit before minimum tenure met (via `_min_tenure_protected`) |
| ✅ | Add wiring test to verify setting reduces early exits |

#### Acceptance Criteria Status
| Status | Criterion |
|--------|-----------|
| ❌ | Higher min_tenure_periods reduces early exit rate |
| ❌ | Funds held for at least min_tenure periods |
| ✅ | Settings wiring test passes |

#### Evidence from PR Changes
1. **engine.py heavily modified** — Added `min_tenure_n`, `holdings_tenure`, `_min_tenure_protected`, `_min_tenure_guard`
2. **Test added** — `test_min_tenure_blocks_early_exits` validates the behavior
3. **`holding_tenure` output** — Results now include tenure tracking

#### Gap Analysis
- Implementation looks complete and well-tested
- Test verifies that:
  - `B` is in period2 holdings with `min_tenure_n=2` but NOT without
  - Tenure counts increment correctly (1 → 2)
  - After tenure met, fund CAN be dropped
- AC checkboxes not updated despite work being done

**Verdict**: Implementation appears COMPLETE. Tests validate all criteria. PR body not updated.

---

## Pattern Analysis: Why Keepalive Is Grinding to a Halt

### Issue 1: Checkbox-based progress tracking doesn't reflect actual work

**Evidence**: PR #4112 and #4114 have work complete but checkboxes unchecked.

**Root cause**: Codex makes code changes but fails to update PR body checkboxes. The keepalive automation reads checkboxes and sees no progress.

### Issue 2: Scope/Tasks extraction is format-sensitive

**Evidence**: `parseScopeTasksAcceptanceSections` in `issue_scope_parser.js` relies on specific markdown patterns.

**Root cause**: If Codex reformats the PR body or the status summary block changes, parsing fails.

### Issue 3: No automated validation of actual code changes against acceptance criteria

**Evidence**: Keepalive can't tell that `test_negative_weights_with_short_min_weight` satisfies "disabling long_only allows negative weights".

**Root cause**: The system lacks semantic understanding of what code changes mean in relation to AC.

### Issue 4: Tests exist but may not pass

**Evidence**: PR #4111 adds tests expecting API response fields that may not exist.

**Root cause**: No pre-commit validation that tests actually pass.

---

## Recommendations for Keepalive Improvements

### Immediate Fixes (can do now)

1. **For PR #4112**: Update PR body to mark AC complete — work is done
2. **For PR #4114**: Update PR body to mark tasks/AC complete — implementation is solid
3. **For PR #4113**: Create the missing workflow YAML
4. **For PR #4111**: Fix the API to expose diagnostics, or fix tests to match actual API

### Medium-term: Improve Progress Detection

See section below on LangChain-based improvements.

---

## LangChain Integration Proposals

### Proposal 4: Resilient Task/Scope Extraction

**Problem**: Current `issue_scope_parser.js` is regex-based and format-sensitive.

**Solution Architecture**:
```python
# tools/keepalive_llm_extractor.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

class ExtractedTask(BaseModel):
    text: str = Field(description="The task description")
    is_complete: bool = Field(description="Whether the task checkbox is checked")
    confidence: float = Field(description="Confidence in extraction (0-1)")

class ScopeExtraction(BaseModel):
    scope: str = Field(description="The scope/objective of the work")
    tasks: List[ExtractedTask] = Field(description="List of tasks")
    acceptance_criteria: List[ExtractedTask] = Field(description="List of acceptance criteria")
    extraction_notes: str = Field(description="Any issues encountered during extraction")

def extract_scope_tasks_acceptance(body: str) -> ScopeExtraction:
    """LLM-based extraction with fallback to regex."""
    parser = PydanticOutputParser(pydantic_object=ScopeExtraction)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are extracting structured information from a GitHub PR/Issue body.
Extract the Scope, Tasks, and Acceptance Criteria sections.
For tasks and acceptance criteria, determine if each checkbox is checked (complete).
A checkbox is checked if it shows [x] or [X], unchecked if [ ].

Be tolerant of:
- Different heading formats (# Header, **Header**, Header:)
- Nested lists
- Missing sections
- Malformed checkboxes

{format_instructions}"""),
        ("human", "{body}")
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | parser
    
    return chain.invoke({
        "body": body,
        "format_instructions": parser.get_format_instructions()
    })
```

**Integration Point**: Called from `keepalive_loop.js` when regex parsing fails.

### Proposal 5: Composable Prompt Assembly with State

**Problem**: Hand-rolled templates in `keepalive_instruction_template.js` are brittle.

**Solution Architecture**:
```python
# tools/keepalive_prompt_builder.py
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class KeepaliveContext:
    pr_number: int
    round: int
    tasks_total: int
    tasks_complete: int
    ci_status: str  # success, failure, pending
    last_error: Optional[str]
    attempted_tasks: List[str]  # Tasks tried in previous rounds
    codex_log_summary: Optional[str]  # Extracted from Codex session log

class KeepalivePromptBuilder:
    def __init__(self):
        self.memory = ConversationBufferWindowMemory(k=5, return_messages=True)
        
        self.base_template = ChatPromptTemplate.from_messages([
            ("system", self._system_prompt()),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{instruction}")
        ])
    
    def _system_prompt(self) -> str:
        return """You are directing an AI coding agent to complete tasks on a PR.
Your role is to:
1. Identify the next incomplete task
2. Provide clear, actionable instructions
3. Track what has been attempted
4. Redirect if previous attempts failed"""
    
    def build_instruction(self, ctx: KeepaliveContext, scope_data: dict) -> str:
        """Build instruction based on context and what's been tried."""
        
        # Routing logic
        if ctx.ci_status == "failure":
            return self._build_ci_failure_instruction(ctx, scope_data)
        elif ctx.tasks_complete == ctx.tasks_total:
            return self._build_verification_instruction(ctx, scope_data)
        else:
            return self._build_task_instruction(ctx, scope_data)
    
    def _build_ci_failure_instruction(self, ctx: KeepaliveContext, scope_data: dict) -> str:
        """Special prompt for CI failure scenarios."""
        return f"""CI has failed. Before proceeding with new tasks:

1. Review the failure: {ctx.last_error or 'Check workflow logs'}
2. Fix the failing tests/checks
3. Verify the fix locally if possible

Previous attempts in this session: {ctx.attempted_tasks}

After fixing CI, continue with remaining tasks from:
{scope_data.get('tasks', 'No tasks found')}"""
    
    def update_memory(self, round_summary: str):
        """Track what happened in each round."""
        self.memory.save_context(
            {"input": f"Round {round_summary}"},
            {"output": "Acknowledged"}
        )
```

### Proposal 6: LLM Failure Triage Layer

**Problem**: `post_ci_summary.py` shows what failed but not what to do.

**Solution Architecture**:
```python
# tools/ci_failure_triage.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Optional
import re

class FailureTriageResult:
    def __init__(self, 
                 failing_job: str,
                 error_type: str,
                 root_cause: str,
                 suggested_fix: str,
                 relevant_files: list[str],
                 playbook_link: Optional[str]):
        self.failing_job = failing_job
        self.error_type = error_type
        self.root_cause = root_cause
        self.suggested_fix = suggested_fix
        self.relevant_files = relevant_files
        self.playbook_link = playbook_link

def triage_ci_failure(
    job_name: str,
    log_content: str,
    max_log_lines: int = 200
) -> FailureTriageResult:
    """Analyze CI failure and suggest fix."""
    
    # Truncate log to relevant portion
    lines = log_content.split('\n')
    if len(lines) > max_log_lines:
        # Find error context
        error_idx = next(
            (i for i, line in enumerate(lines) 
             if 'error' in line.lower() or 'failed' in line.lower()),
            len(lines) - max_log_lines
        )
        start = max(0, error_idx - max_log_lines // 2)
        lines = lines[start:start + max_log_lines]
    
    truncated_log = '\n'.join(lines)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a CI/CD debugging expert. Analyze the failure log and provide:
1. The type of error (test failure, type error, import error, coverage, etc.)
2. The likely root cause (one sentence)
3. A specific fix suggestion (actionable steps)
4. Which files likely need changes

Be concise. Focus on the actual error, not warnings."""),
        ("human", """Job: {job_name}

Log excerpt:
```
{log_content}
```

Provide analysis.""")
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(prompt.format(
        job_name=job_name,
        log_content=truncated_log
    ))
    
    # Parse response into structured result
    return _parse_triage_response(job_name, response.content)

def _parse_triage_response(job_name: str, response: str) -> FailureTriageResult:
    """Parse LLM response into structured result."""
    # Implementation would extract structured fields from response
    # Could also use function calling or output parser
    ...
```

### New Proposal: Codex Log Analyzer

**Problem**: Codex's work logs contain valuable information about what was attempted, what failed, and actual progress that isn't reflected in checkbox updates.

**Solution Architecture**:
```python
# tools/codex_log_analyzer.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

class TaskProgress(BaseModel):
    task_text: str = Field(description="The task from the original list")
    status: str = Field(description="complete, partial, attempted_failed, not_started")
    evidence: str = Field(description="What in the log shows this status")
    files_modified: List[str] = Field(description="Files touched for this task")

class CodexLogAnalysis(BaseModel):
    tasks_completed: List[TaskProgress] = Field(description="Tasks with evidence of completion")
    tasks_attempted: List[TaskProgress] = Field(description="Tasks attempted but not completed")
    blockers_encountered: List[str] = Field(description="Errors or blockers Codex hit")
    suggested_next_action: str = Field(description="What should happen next")
    checkbox_updates_needed: List[str] = Field(description="Checkboxes that should be marked complete")

def analyze_codex_session(
    session_log: str,
    original_tasks: List[str],
    original_acceptance: List[str]
) -> CodexLogAnalysis:
    """Analyze Codex session log to determine actual progress."""
    
    parser = PydanticOutputParser(pydantic_object=CodexLogAnalysis)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are analyzing a Codex coding agent's session log to determine actual progress.

Compare what Codex did against the original task list and acceptance criteria.
Look for:
- Files created or modified
- Tests added or updated
- Errors encountered
- Commits made
- Success/failure messages

Tasks may be COMPLETE even if Codex didn't explicitly check the checkbox.
Look for evidence in the actual work done.

{format_instructions}"""),
        ("human", """Original Tasks:
{tasks}

Original Acceptance Criteria:
{acceptance}

Codex Session Log:
{log}

Analyze progress.""")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Use larger model for log analysis
    chain = prompt | llm | parser
    
    return chain.invoke({
        "tasks": "\n".join(f"- {t}" for t in original_tasks),
        "acceptance": "\n".join(f"- {a}" for a in original_acceptance),
        "log": session_log,
        "format_instructions": parser.get_format_instructions()
    })
```

**Integration with Keepalive**:
```javascript
// In keepalive_loop.js, after Codex round completes:

async function analyzeCodexProgress({ prNumber, sessionLog, tasks, acceptance }) {
  // Call Python tool via subprocess or API
  const analysis = await runPythonTool('codex_log_analyzer', {
    session_log: sessionLog,
    original_tasks: tasks,
    original_acceptance: acceptance
  });
  
  // Auto-update PR body checkboxes based on analysis
  if (analysis.checkbox_updates_needed.length > 0) {
    await updatePRCheckboxes(prNumber, analysis.checkbox_updates_needed);
  }
  
  // If blockers found, adjust next round's instruction
  if (analysis.blockers_encountered.length > 0) {
    return buildBlockerResolutionInstruction(analysis);
  }
  
  return null; // Continue normal flow
}
```

---

## Implementation Priority

| Priority | Proposal | Effort | Impact |
|----------|----------|--------|--------|
| 1 | **Codex Log Analyzer** | Medium | High — directly addresses checkbox update gap |
| 2 | **Resilient Task Extraction (4)** | Low | Medium — prevents "no-checklists" failures |
| 3 | **CI Failure Triage (6)** | Low | Medium — speeds up debugging |
| 4 | **Composable Prompts (5)** | High | Medium — cleaner but not urgent |

---

## Immediate Action Items

1. ✅ Create this tracking document
2. ⏳ Update PR #4112 body to reflect completed work
3. ⏳ Update PR #4114 body to reflect completed work  
4. ⏳ Create settings-effectiveness.yml workflow for PR #4113
5. ⏳ Fix PR #4111 to either add missing API exposure or fix tests
6. ⏳ Prototype Codex Log Analyzer tool

---

*Document created: 2026-01-02*
*Next review: After immediate action items complete*
