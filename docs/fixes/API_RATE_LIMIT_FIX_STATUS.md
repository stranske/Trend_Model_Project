# API Rate Limit Fix Status

> **Last Updated:** 2026-02-02
> **Current Status:** � In Progress – Fix PR Created
> **Active PR:** [Workflows #1183](https://github.com/stranske/Workflows/pull/1183) – comprehensive rate limit remediation
> **Blocking Issue:** Token load balancer missing dependencies in consumer repo contexts

---

## Latest Progress

**PR #1183 created** in Workflows repo with:
1. New unified `setup-api-client` action that combines npm install + token export
2. Comprehensive remediation plan in `docs/fixes/RATE_LIMIT_REMEDIATION_PLAN.md`
3. Pinned @octokit/* versions for consistency

**Remaining tasks:**
- [ ] Apply setup-api-client action to agents-keepalive-loop.yml
- [ ] Update status messaging to show "🛑 Agent Stopped" when rate-limited  
- [ ] Add to consumer repo sync manifest
- [ ] Create test PRs to verify

---

## Executive Summary

The workflow system has been enhanced with a multi-token load balancer to distribute API calls across ~25,000 requests/hour (5 GitHub Apps + 2 PATs). However, **recent PRs #4647, #4648, and #4641 still hit rate limits and stall** because:

1. **Token registry initializes with 0 tokens** in some job contexts (missing secrets/deps)
2. **Missing `@octokit/rest` package** when token_load_balancer.js runs in consumer repos
3. **Fallback to GITHUB_TOKEN** when registry empty → single 5,000/hr pool exhausted
4. **No recovery mechanism** when all tokens exhausted mid-run
5. **Keepalive summary falsely reports agent as "running"** even when rate-limited

### End Goal

Multiple PRs should continue work **simultaneously** without rate limit interference. If limits ARE hit:
1. The system MUST NOT falsely report the agent is "running"
2. A clear status of ALL API capacity sources MUST be displayed
3. Work should ONLY stop if ALL apps/PATs are at or near capacity
4. A handoff document MUST explain where the fix process stands for the next bot

---

## Failure Analysis: PRs #4647, #4648, #4641

### Symptom (from workflow annotations)
```
Rate limit hit while updating PR #4647: GitHub API rate limit exceeded. Reset at: 2026-02-01T13:30:15.000Z. Remaining: 0
Mark agent running: No tokens available with required capabilities and capacity
Token registry returned no available token
Failed to check rate limit for GITHUB_TOKEN: Cannot find package '@octokit/rest'
```

### Root Causes Identified

| Problem | Impact | Status |
|---------|--------|--------|
| `@octokit/rest` not installed in consumer repo context | Token load balancer can't check rate limits | 🔴 Open |
| Token registry initializes with 0 tokens in some jobs | No tokens available → falls back to exhausted GITHUB_TOKEN | 🔴 Open |
| Secrets not passed to downstream jobs | PATs/App credentials unavailable | 🔴 Open |
| Keepalive summary job uses GITHUB_TOKEN directly | Summary fails when installation token exhausted | 🔴 Open |
| No multi-token fallback during mid-execution | Once started with one token, can't switch | 🟡 Partial |

### Timeline from Logs (Run 21563133208)

| Time (UTC) | Event | Tokens |
|------------|-------|--------|
| 12:47:24 | Evaluate keepalive loop starts | 7 tokens initialized |
| 12:47:25 | WORKFLOWS_APP selected (4942 remaining, 98.8%) | Working |
| 12:47:27 | Rate limit check: 30354/35000 remaining | Healthy |
| 12:47:42 | Mark agent running job starts | **0 tokens initialized** |
| 12:48:43 | Codex runs, starts generating output | Using GITHUB_TOKEN |
| 12:54:47 | Token registry only has 1 token | GITHUB_TOKEN only |
| 12:54:47 | **FAILURE:** Cannot find package '@octokit/rest' | deps missing |
| 12:55:21 | Max retries reached, rate limit exhausted | All requests failing |
| 12:55:31 | Update keepalive summary starts | **0 tokens initialized** |
| 12:56:06 | PR update fails, summary reports falsely | Agent appears "running" |

---

## What Works ✅

1. **Token Load Balancer Architecture** (PR #1008)
   - Multi-token registry supporting PATs + GitHub Apps
   - Proactive token selection based on remaining capacity
   - Token specialization for exclusive tasks

2. **Retry Wrapper with Token Awareness** (PRs #1079, #1080, #1135, #1139)
   - Exponential backoff on rate limit errors
   - Header-based rate limit detection
   - Pagination support with rate awareness

3. **API Rate Diagnostic Workflow** (health-75-api-rate-diagnostic.yml)
   - Scheduled checks every 30 minutes
   - Reports all token pool capacities
   - Historical trend tracking

4. **High-Volume API Audit** (docs/fixes/high-volume-workflows-api-audit.md)
   - Complete inventory of API calls by workflow
   - Line-number precision for remediation

---

## What Doesn't Work ❌

### 1. Consumer Repo Dependency Gap

**Problem:** Token load balancer requires `@octokit/rest` but consumer repos don't install it.

**Evidence:**
```
Failed to check rate limit for GITHUB_TOKEN: Cannot find package '@octokit/rest' 
  imported from .workflows-lib/.github/scripts/token_load_balancer.js
```

**Location:** `.github/scripts/token_load_balancer.js` line ~190
**Fix needed:** Add `npm install @octokit/rest` step before any job using the load balancer

### 2. Inconsistent Token Registry Initialization

**Problem:** Some jobs initialize with 7 tokens, others with 0.

**Evidence:**
```
Evaluate keepalive loop: Token registry initialized with 7 tokens
Mark agent running: Token registry initialized with 0 tokens
Update keepalive summary: Token registry initialized with 0 tokens
```

**Cause:** Secrets/environment not propagating to all jobs
**Fix needed:** Add export-load-balancer-tokens action to EVERY job that makes API calls

### 3. Summary Reports False "Running" State

**Problem:** When rate limited, the keepalive summary still says agent is "running."

**Evidence:** PR #4647 shows agent blocked but summary doesn't reflect this
**Fix needed:** Summary job must check rate limit status and report accurately

### 4. No Graceful Degradation

**Problem:** When one token exhausted, system doesn't automatically try alternatives.

**Current behavior:** Retry with same exhausted token 5 times → fail
**Needed behavior:** Detect exhaustion → switch to next available token → continue

---

## PR History: Rate Limit Fixes (PR #1008 → Present)

### Phase 1: Foundation (Jan 21-22)

| PR | Title | Outcome |
|----|-------|---------|
| #1008 | feat: Add dynamic token load balancer | ✅ Core architecture |
| #1013 | feat: rate limit notification + label sync | ✅ Alert mechanism |
| #1014 | fix: improve rate limit handling in retry label | ✅ agent:retry support |
| #1028 | Fix/keepalive status and restart instructions | ✅ Documentation |

### Phase 2: Wrapper Integration (Jan 22-23)

| PR | Title | Outcome |
|----|-------|---------|
| #1051 | Add retry wrappers to agents autofix loop | ✅ Autofix protected |
| #1052 | Add retry wrapper to belt dispatcher | ✅ Belt protected |
| #1077 | Wrap autofix-loop API calls with retries | ✅ More coverage |
| #1079 | Integrate retry helpers in keepalive loop | ✅ Keepalive protected |
| #1080 | Integrate retry helpers in reusable codex run | ✅ Codex protected |
| #1082 | sync api-rate-limit helpers to consumer workflows | ✅ Consumer sync |

### Phase 3: Diagnostics & Stability (Jan 26-27)

| PR | Title | Outcome |
|----|-------|---------|
| #1126-1134 | Stabilize API rate diagnostic | ✅ Health-75 working |
| #1135 | Add rate-limit-aware retry to high-frequency scripts | ✅ Script coverage |
| #1139 | Add paginate.iterator support to wrapper | ✅ Pagination fixed |

### Phase 4: Load Balancer Expansion (Jan 28-31)

| PR | Title | Outcome |
|----|-------|---------|
| #1142 | Standardize token export for retry helpers | ✅ Token passing |
| #1146 | Verify token load sharing | ✅ Verification workflow |
| #1148 | Sync export-load-balancer-tokens to consumers | ⚠️ Partial - missing in some jobs |
| #1161 | Stabilize keepalive load balancing | ⚠️ Partial - still gaps |
| #1170 | Install octokit deps | ⚠️ Partial - not all contexts |

### Phase 5: Remediation Attempts (Feb 1-2)

| PR | Title | Outcome |
|----|-------|---------|
| #1172 | Expand load balancer coverage | ⚠️ Incomplete |
| #1173 | Add token-aware retry to issue workflows | ⚠️ Incomplete |
| #1177-1182 | Various fixes for retry/deps | 🔴 Still failing |

---

## Remaining Work Items

### Critical (Blocking PR Progress)

1. **Add `npm install @octokit/rest @octokit/plugin-retry` before load balancer use**
   - Location: Every job using token_load_balancer.js
   - Workflows: agents-keepalive-loop.yml, reusable-codex-run.yml, etc.
   - Consumer template: templates/consumer-repo/.github/workflows/

2. **Add export-load-balancer-tokens action to ALL API-calling jobs**
   - Currently missing from: mark-running, update-summary, post-work
   - Must pass secrets properly

3. **Fix keepalive summary false-positive reporting**
   - Check rate limit status before declaring "agent running"
   - Report accurate capacity across all token pools
   - Include rate limit reset times

4. **Implement mid-execution token switching**
   - Monitor x-ratelimit-remaining headers
   - Switch to alternate token when < 100 remaining
   - Don't wait for complete exhaustion

### Important (Reliability)

5. **Add concurrency groups with cancel-in-progress**
   - Prevent duplicate workflow runs
   - See: docs/ops/debouncing-run-counts.md for full list

6. **Implement GraphQL batching for multi-fetch operations**
   - PR context can be fetched in 1 call instead of 4+
   - See: pr-context-graphql.js (already implemented, needs deployment)

### Nice to Have (Optimization)

7. **Path filters to skip irrelevant changes**
8. **Central API response caching**
9. **Workflow consolidation to reduce run count**

---

## Handoff Instructions for Next Bot

### If You're Continuing This Work:

1. **Read this document first** - understand what's been tried
2. **Check the Workflows repo** for any PRs merged after this document
3. **Run the API diagnostic** to see current token status:
   ```bash
   gh workflow run "Health 75 API Rate Diagnostic" --repo stranske/Workflows
   ```
4. **Look at recent failures** to see if the same patterns continue:
   ```bash
   gh run list --repo stranske/Trend_Model_Project --workflow "Agents Keepalive Loop" --limit 5
   ```

### Key Files to Modify (in Workflows repo):

| File | What Needs Changing |
|------|---------------------|
| `.github/workflows/agents-keepalive-loop.yml` | Add npm install step, export-load-balancer-tokens to all jobs |
| `.github/workflows/reusable-codex-run.yml` | Same as above |
| `.github/scripts/token_load_balancer.js` | Better error handling when deps missing |
| `.github/scripts/keepalive_loop.js` | Mid-execution token switching |
| `.github/scripts/keepalive_state.js` | Accurate status reporting |
| `templates/consumer-repo/.github/workflows/` | All templates need load balancer setup |

### Testing Your Fix:

1. Make changes in Workflows repo
2. Run the sync workflow to push to consumer repos
3. Create a test PR in Trend_Model_Project with `agent:codex` label
4. Watch the keepalive loop - check:
   - Does it initialize with 7+ tokens?
   - Does it switch tokens when one is low?
   - Does the summary accurately reflect status?

### Don't Make These Mistakes:

- ❌ Don't assume one token type is available - check all
- ❌ Don't retry with the same exhausted token
- ❌ Don't report "agent running" without checking rate limits
- ❌ Don't forget to sync changes to consumer repo templates

---

## Monitoring Commands

### Check Current Rate Limits
```bash
# All token pools
gh api rate_limit --jq '.resources.core | "\(.remaining)/\(.limit) remaining, resets \(.reset | todate)"'

# Via diagnostic workflow
gh run view $(gh run list --repo stranske/Workflows --workflow "Health 75 API Rate Diagnostic" --limit 1 --json databaseId -q '.[0].databaseId') --repo stranske/Workflows --log | grep -E "remaining|capacity"
```

### Check Recent Keepalive Failures
```bash
gh run list --repo stranske/Trend_Model_Project --workflow "Agents Keepalive Loop" --status failure --limit 10
```

### Check Token Registry Initialization
```bash
gh run view <run_id> --repo stranske/Trend_Model_Project --log | grep "Token registry initialized"
```

---

## Related Documentation

- [Workflows/docs/fixes/high-volume-workflows-api-audit.md](https://github.com/stranske/Workflows/blob/main/docs/fixes/high-volume-workflows-api-audit.md) - Complete API call inventory
- [Workflows/docs/ops/RATE_LIMIT_MANAGEMENT.md](https://github.com/stranske/Workflows/blob/main/docs/ops/RATE_LIMIT_MANAGEMENT.md) - Architecture overview
- [Workflows/docs/keepalive/KEEPALIVE_TROUBLESHOOTING.md](https://github.com/stranske/Workflows/blob/main/docs/keepalive/KEEPALIVE_TROUBLESHOOTING.md) - General keepalive debugging

---

## Success Criteria

This fix effort is COMPLETE when:

- [ ] Multiple PRs can run keepalive simultaneously without rate limit failures
- [ ] Token registry consistently initializes with all available tokens
- [ ] No `@octokit/rest` import errors in any workflow context
- [ ] Keepalive summary accurately reports "rate limited" when blocked
- [ ] System automatically switches tokens before exhaustion
- [ ] All token pools show in status when work stops
- [ ] Handoff document is updated with resolution
