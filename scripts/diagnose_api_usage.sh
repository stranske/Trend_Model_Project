#!/bin/bash
# API Usage Diagnostic - Shell wrapper
# Checks rate limits for the current GitHub token

set -e

echo "🔍 API Usage Diagnostic Report (Quick Check)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo

# Check if gh CLI is available
if ! command -v gh &> /dev/null; then
    echo "❌ Error: gh CLI is not installed"
    echo "   Install it from: https://cli.github.com/"
    exit 1
fi

# Check authentication
if ! gh auth status &> /dev/null; then
    echo "❌ Error: Not authenticated with GitHub"
    echo "   Run: gh auth login"
    exit 1
fi

echo "📊 Current Token Rate Limit Status"
echo "───────────────────────────────────────────────────────────────────────────────"
echo

# Get rate limit info
RATE_LIMIT=$(gh api rate_limit 2>&1)

if [ $? -ne 0 ]; then
    echo "❌ Error fetching rate limit: $RATE_LIMIT"
    exit 1
fi

# Parse using jq if available, otherwise use basic parsing
if command -v jq &> /dev/null; then
    CORE_LIMIT=$(echo "$RATE_LIMIT" | jq -r '.resources.core.limit')
    CORE_USED=$(echo "$RATE_LIMIT" | jq -r '.resources.core.used')
    CORE_REMAINING=$(echo "$RATE_LIMIT" | jq -r '.resources.core.remaining')
    CORE_RESET=$(echo "$RATE_LIMIT" | jq -r '.resources.core.reset')
    
    SEARCH_LIMIT=$(echo "$RATE_LIMIT" | jq -r '.resources.search.limit')
    SEARCH_USED=$(echo "$RATE_LIMIT" | jq -r '.resources.search.used')
    SEARCH_REMAINING=$(echo "$RATE_LIMIT" | jq -r '.resources.search.remaining')
    SEARCH_RESET=$(echo "$RATE_LIMIT" | jq -r '.resources.search.reset')
    
    GRAPHQL_LIMIT=$(echo "$RATE_LIMIT" | jq -r '.resources.graphql.limit')
    GRAPHQL_USED=$(echo "$RATE_LIMIT" | jq -r '.resources.graphql.used')
    GRAPHQL_REMAINING=$(echo "$RATE_LIMIT" | jq -r '.resources.graphql.remaining')
    GRAPHQL_RESET=$(echo "$RATE_LIMIT" | jq -r '.resources.graphql.reset')
    
    # Calculate percentages
    CORE_PERCENT_USED=$(echo "scale=1; ($CORE_USED * 100) / $CORE_LIMIT" | bc 2>/dev/null || echo "0")
    CORE_PERCENT_REMAINING=$(echo "scale=1; ($CORE_REMAINING * 100) / $CORE_LIMIT" | bc 2>/dev/null || echo "100")
    
    # Convert reset time
    CORE_RESET_DATE=$(date -d "@$CORE_RESET" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -r "$CORE_RESET" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "Unknown")
    CURRENT_TIME=$(date +%s)
    MINUTES_UNTIL_RESET=$(echo "($CORE_RESET - $CURRENT_TIME) / 60" | bc 2>/dev/null || echo "Unknown")
    
    # Determine status (using bc for comparison)
    if [ $(echo "$CORE_PERCENT_REMAINING >= 80" | bc -l 2>/dev/null) -eq 1 ]; then
        STATUS="✅ healthy"
    elif [ $(echo "$CORE_PERCENT_REMAINING >= 50" | bc -l 2>/dev/null) -eq 1 ]; then
        STATUS="⚠️  moderate"
    elif [ $(echo "$CORE_PERCENT_REMAINING >= 20" | bc -l 2>/dev/null) -eq 1 ]; then
        STATUS="🔶 low"
    elif [ $(echo "$CORE_PERCENT_REMAINING >= 5" | bc -l 2>/dev/null) -eq 1 ]; then
        STATUS="🔴 critical"
    else
        STATUS="🚨 exhausted"
    fi
    
    echo "Core API:"
    echo "  Status:     $STATUS"
    echo "  Limit:      $CORE_LIMIT requests/hour"
    echo "  Used:       $CORE_USED ($CORE_PERCENT_USED%)"
    echo "  Remaining:  $CORE_REMAINING ($CORE_PERCENT_REMAINING%)"
    echo "  Resets:     $CORE_RESET_DATE (in ~$MINUTES_UNTIL_RESET minutes)"
    echo
    
    echo "Search API:"
    echo "  Limit:      $SEARCH_LIMIT requests/minute"
    echo "  Used:       $SEARCH_USED"
    echo "  Remaining:  $SEARCH_REMAINING"
    echo
    
    echo "GraphQL API:"
    echo "  Limit:      $GRAPHQL_LIMIT points/hour"
    echo "  Used:       $GRAPHQL_USED"
    echo "  Remaining:  $GRAPHQL_REMAINING"
    echo
    
    # Recommendations
    echo "💡 Recommendations"
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo
    
    if [[ "$STATUS" == *"exhausted"* ]] || [[ "$STATUS" == *"critical"* ]]; then
        echo "🚨 CRITICAL: Rate limit is critically low or exhausted!"
        echo "   • Consider pausing non-critical workflows"
        echo "   • Wait $MINUTES_UNTIL_RESET minutes for rate limit reset"
        echo "   • Configure additional tokens (PATs or GitHub Apps) for load distribution"
        echo
    elif [[ "$STATUS" == *"low"* ]]; then
        echo "⚠️  WARNING: Rate limit is running low"
        echo "   • Monitor usage closely"
        echo "   • Consider configuring additional tokens for load distribution"
        echo
    else
        echo "✅ Status is healthy - no immediate action needed"
        echo
    fi
    
    # Check if this is likely GITHUB_TOKEN (installation token)
    CURRENT_USER=$(gh api user --jq '.login' 2>/dev/null || echo "unknown")
    if [[ "$CURRENT_USER" == *"[bot]"* ]] || [[ "$CORE_LIMIT" == "1000" ]]; then
        echo "ℹ️  Detection: This appears to be a GitHub Actions installation token"
        echo "   Installation tokens have limited rate limits (typically 1000/hr)"
        echo "   Consider configuring dedicated PATs or GitHub Apps for high-volume operations"
        echo
    fi
    
else
    echo "⚠️  jq not available - showing raw output:"
    echo "$RATE_LIMIT"
    echo
    echo "Install jq for formatted output: https://stedolan.github.io/jq/"
fi

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "Report generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo
echo "For detailed multi-token analysis, run:"
echo "  node .github/scripts/diagnose_api_usage.js"
