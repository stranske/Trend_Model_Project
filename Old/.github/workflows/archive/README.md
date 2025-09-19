This directory contains deprecated or archived workflow prototypes retained for historical reference.

Files here are not loaded by GitHub Actions because the path includes an additional nesting under `Old/` and/or `archive/`.

Issue #1140: Hardened `pull_request_target` workflows; legacy experimental files were relocated or removed to avoid accidental activation.

Removed on 2025-09-18 (cleanup pass):
- assign-to-agent.yml.rewrite
- assign-to-agent-legacy.yml
- assign-to-agent.yml
- codex-assign-minimal.yml
- verify-codex-bootstrap.yml

Rationale:
These were superseded by consolidated bootstrap and labeling workflows. Keeping inert duplicates increases audit surface and cognitive load. Removal confirmed no active references in docs (aside from historical listing) or other workflows.