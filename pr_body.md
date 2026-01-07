<!-- pr-preamble:start -->
> **Source:** Issue #4180

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
This is the core NL component: a LangChain chain that takes a user instruction and current config, then produces a valid `ConfigPatch` as structured output. Using a deterministic chain (not an agent) gives us predictable, auditable behavior.

#### Tasks
- [x] Design prompt template with sections:
- [x] - System prompt explaining the task
- [x] - Current config (or relevant excerpt)
- [x] - Allowed schema (from Issue 4)
- [x] - Safety rules (don't invent keys, flag risks)
- [x] - User instruction
- [ ] Implement `ConfigPatchChain` using LangChain:
- [x] Define the class structure. (verify: confirm completion in repo)
- [ ] Define methods. (verify: confirm completion in repo)
- [ ] Implement the run method. (verify: confirm completion in repo)
- [ ] Integrate with LangChain. (verify: confirm completion in repo)
- [ ] Configure model settings for reliability:
- [ ] Set low temperature. (verify: confirm completion in repo)
- [ ] Enforce structured output. (verify: confirm completion in repo)
- [ ] Determine max tokens for patch size. (verify: confirm completion in repo)
- [ ] Implement schema injection:
- [ ] Include compact schema in prompt. (verify: confirm completion in repo)
- [ ] Limit to relevant sections based on instruction. (verify: confirm completion in repo)
- [ ] Add self-check step:
- [ ] Validate patch references known keys. (verify: confirm completion in repo)
- [ ] Flag unknown keys for human review. (verify: confirm completion in repo)
- [ ] Implement rejection criteria based on unknown keys. (verify: confirm completion in repo)
- [ ] Implement retry logic:
- [ ] Implement retry on parsing failure. (verify: confirm completion in repo)
- [ ] Include previous error in retry prompt. (verify: confirm completion in repo)
- [ ] Add provider abstraction:
- [ ] Support OpenAI. (verify: confirm completion in repo)
- [ ] Support Anthropic Claude. (verify: confirm completion in repo)
- [ ] Support local Ollama. (verify: confirm completion in repo)

#### Acceptance criteria
- [ ] Chain produces valid ConfigPatch for a defined set of common instructions.
- [ ] Define specific test cases and metrics for structured output parsing reliability (≥95% success rate).
- [ ] Unknown keys are flagged, not silently included.
- [ ] Local eval harness exists for testing prompts.
- [ ] Temperature and model settings are configurable.
- [ ] Chain works with at least 2 LLM providers.

<!-- auto-status-summary:end -->
