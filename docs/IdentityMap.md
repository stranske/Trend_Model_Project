# Identity Map

Trend run manifests preserve raw selected fund labels in `selected_funds` and add a
deterministic `selected_entities` block for downstream joins.

```yaml
identity:
  entities:
    - canonical_id: fund:aqr-managed-futures
      display_name: AQR Managed Futures
      aliases:
        - AQR MF
        - AQR Managed Futures
  universes:
    - config/universe/core.yml
```

Resolution is exact after case and whitespace normalization. The resolver checks
`display_name`, `canonical_id`, and every alias. It does not use fuzzy matching,
network calls, or LLMs.

Unmatched labels are explicit:

```json
{
  "label": "Unmapped Fund",
  "canonical_id": "unknown:Unmapped Fund",
  "display_name": "Unmapped Fund",
  "resolved": false
}
```

When multiple raw selected labels resolve to the same canonical ID, the manifest
emits one `selected_entities` entry with a `labels` list. `selected_funds` remains
unchanged for backward compatibility.

Universe files can seed aliases through `identity.universes`. Each member becomes
a deterministic `fund:<normalized-member>` canonical ID unless an explicit entity
entry overrides it.
