<!-- pr-preamble:start -->
> **Source:** Issue #4202

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
LangChain and Pydantic have complex version interdependencies that can cause subtle runtime failures if not carefully managed. LangChain v1 requires Python 3.10+ and Pydantic v2 for best compatibility. Deciding these versions upfront prevents dependency hell during development and protects against accidental drift.

#### Tasks
- [x] Review current Python version requirements in `pyproject.toml`
- [ ] Decide LangChain version strategy:
- [ ] - **Recommended**: LangChain v1.x (stable, well-documented)
- [ ] - Pin to minor version range (e.g., `langchain>=1.0,<1.1`)
- [ ] Decide Pydantic version strategy:
- [ ] - **Recommended**: Pydantic v2-only (v1 compatibility mode has edge cases)
- [ ] - Verify existing codebase is v2-compatible
- [x] Add `langchain`, `langchain-core`, `langchain-community` to optional dependencies
- [x] Create `[llm]` extras group in pyproject.toml
- [ ] Add CI check that fails if:
- [ ] - Python < 3.10 is used with LLM extras
- [ ] - Pydantic v1 is resolved when LLM extras are installed
- [x] Document version requirements in README or DEPENDENCY_QUICKSTART.md

#### Acceptance criteria
- [x] `pyproject.toml` updated with pinned LangChain versions in `[project.optional-dependencies]`
- [x] `pip install -e ".[llm]"` installs compatible LangChain + Pydantic
- [x] CI job exists that validates dependency compatibility
- [ ] README documents Python 3.10+ requirement for NL features
- [x] No Pydantic v1/v2 compatibility warnings when importing langchain

<!-- auto-status-summary:end -->
