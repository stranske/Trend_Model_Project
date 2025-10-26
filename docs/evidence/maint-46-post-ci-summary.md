## Automated Status Summary
**Head SHA:** ebeec96b10458fe6c1674d1165965dedad67d720
**Latest Runs:** ✅ success — [Gate (#101)](https://example.test/gate/101)
**Required:** core tests (3.11): ✅ success, core tests (3.12): ✅ success, docker smoke: ✅ success, gate: ✅ success

| Workflow / Job | Result | Logs |
|----------------|--------|------|
| Gate / core tests (3.11) | ✅ success | [logs](https://example.test/gate/101/py311) |
| Gate / core tests (3.12) | ✅ success | [logs](https://example.test/gate/101/py312) |
| Gate / docker smoke | ✅ success | [logs](https://example.test/gate/101/docker) |
| Gate / gate | ✅ success | [logs](https://example.test/gate/101/gate) |

### Coverage Overview
- Coverage delta: head 91.23% | base 90.50% | Δ +0.73 pp | drop 0.00 pp | threshold 1.00 pp | status ok
- Coverage (jobs): 91.23% | Δ +0.12 pp
- Coverage (worst job): 88.56% | Δ -0.34 pp
- Coverage history entries: 12
| Runtime | Coverage | Δ vs 3.11 |
| --- | --- | --- |
| 3.11 | 91.23% | — |
| 3.12 | 91.05% | -0.18 pp |

### Screenshot artifact
The rendered Gate summary preview is stored as a Base64 text artifact (`maint-46-post-ci-summary.b64`) to avoid binary diffs.
Recreate the PNG locally with:

```bash
base64 -d docs/evidence/maint-46-post-ci-summary.b64 > docs/evidence/maint-46-post-ci-summary.png
```

This preserves the evidence while keeping the repository text-only for easier reviews.
