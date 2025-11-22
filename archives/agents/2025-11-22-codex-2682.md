<!-- bootstrap for codex on issue #2682

Source Issue: https://github.com/stranske/Trend_Model_Project/issues/2682
Topic GUID: c1017df4-8810-5fca-b072-f5bd11c13093

## Scope / Key Constraints
- Remove the legacy Agents 61/62 consumer workflows and any remaining shim logic without disturbing the Agents 63 bridge or the Orchestrator jobs.
- Update documentation, runbooks, and references that mention Agents 61/62 while preserving historical context where required (e.g., archives) by clearly marking them as retired.
- Ensure the Actions catalogue and workflow guardrails reflect only the supported Agents 63 + Orchestrator topology; avoid changes to unrelated CI pipelines or agent versions.

## Acceptance Criteria / Definition of Done
- The `.github/workflows/agents-61-consumer-compat.yml` and `.github/workflows/agents-62-consumer.yml` files are deleted and no other workflows reference the retired consumers.
- Repository content (docs, comments, configs, tests) no longer references "Agents 61" or "Agents 62" except in archival notes explicitly stating their retirement.
- Documentation (WORKFLOW_SYSTEM.md, AGENTS_POLICY.md, and any affected runbooks) states that Agents 63 + Orchestrator are the only supported workflow surfaces.
- Automated checks and the GitHub Actions UI display only Agents 63 and Orchestrator workflows for the agents family.

## Initial Task Checklist
- [ ] Remove the deprecated Agents 61/62 workflow files from `.github/workflows/`.
- [ ] `rg -i "Agents 6[12]"` across the repo and clean up or reword each reference; add retirement notes where historical context is necessary.
- [ ] Update `docs/ci/WORKFLOW_SYSTEM.md` and `docs/AGENTS_POLICY.md` to call out the retirement and the supported topology.
- [ ] Verify runbooks, tests, and guardrails (e.g., consolidation tests) do not expect the old consumers; adjust assertions if needed.
- [ ] Confirm Actions tab/CI manifests list only Agents 63 bridge and Orchestrator jobs, documenting any follow-up if external configuration is required.
-->
