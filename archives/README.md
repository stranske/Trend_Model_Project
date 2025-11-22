# Archives overview

This directory holds long-term references that are no longer part of the active codebase but should remain available for historical context.

## Classification of items moved from `retired/`
- **Permanently archived:**
  - `archives/retired/agents-64-pr-comment-commands.yml` — preserved as a record of the former PR comment command workflow and its replacement (`agents-pr-meta.yml`).
  - `archives/retired/agents-74-pr-body-writer.yml` — kept to document how PR body synchronization used to work before consolidation into `agents-pr-meta.yml`.
- **Removal candidates:** None at this time. The `retired/` directory now serves only as a staging area for newly removed assets awaiting review.

## Placement policy
- Store artifacts directly in `archives/` when they provide long-term historical reference or implementation detail that may be consulted again (e.g., superseded workflows, legacy configs, or data snapshots).
- Move items into `retired/` when they are freshly removed, under evaluation for deletion, or still awaiting a decision about long-term retention.
- Once an item in `retired/` is confirmed as a useful reference, relocate it into the parallel path under `archives/`, and leave a brief pointer in `retired/` summarizing the move.
