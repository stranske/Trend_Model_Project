# Actionlint Checks Reference

This project uses [actionlint](https://github.com/rhysd/actionlint) for GitHub Actions workflow validation.

## Workflow

**File**: `.github/workflows/health-42-actionlint.yml`

The actionlint workflow runs on pull requests to validate all workflow YAML files.

## Documentation

For the complete list of checks performed by actionlint, see:
- [Actionlint Checks Documentation](https://github.com/rhysd/actionlint/blob/main/docs/checks.md)
- [Actionlint Usage Guide](https://github.com/rhysd/actionlint#readme)

## Configuration

- **Allowlist**: `.github/actionlint-allowlist.txt` - patterns to ignore
- **Workflow**: `health-42-actionlint.yml` - CI integration

## Common Issues

### Expression syntax errors
```yaml
# Wrong
if: ${{ github.event.pull_request.merged == true }}
# Right (no quotes needed in if:)
if: github.event.pull_request.merged == true
```

### Missing permissions
```yaml
permissions:
  contents: read
  pull-requests: write
```

### Deprecated commands
Replace `set-output` with `$GITHUB_OUTPUT`:
```yaml
# Deprecated
echo "::set-output name=value::$VALUE"
# Current
echo "value=$VALUE" >> "$GITHUB_OUTPUT"
```

## Running Locally

```bash
# Install actionlint
brew install actionlint  # macOS
# or
go install github.com/rhysd/actionlint/cmd/actionlint@latest

# Run on all workflows
actionlint .github/workflows/*.yml
```
