/**
 * API Usage Diagnostic Script
 * 
 * Evaluates API rate limit status across all configured tokens
 * and provides recommendations for workflow token allocation.
 * 
 * Usage:
 *   node .github/scripts/diagnose_api_usage.js
 *   
 * With GitHub Actions context:
 *   GITHUB_TOKEN=${{secrets.GITHUB_TOKEN}} \
 *   SERVICE_BOT_PAT=${{secrets.SERVICE_BOT_PAT}} \
 *   ... \
 *   node .github/scripts/diagnose_api_usage.js
 *   
 * Or using gh CLI (simpler):
 *   ./scripts/diagnose_api_usage.sh
 */

const { execSync } = require('child_process');
const fs = require('fs');

// Token configuration - maps env var names to token metadata
const TOKEN_CONFIG = [
  {
    id: 'GITHUB_TOKEN',
    envVar: 'GITHUB_TOKEN',
    type: 'Installation',
    description: 'Default Actions token (installation-scoped)',
    workflows: ['Most workflows by default'],
  },
  {
    id: 'SERVICE_BOT_PAT',
    envVar: 'SERVICE_BOT_PAT',
    type: 'PAT',
    description: 'Bot account PAT for automation',
    workflows: ['Bot comments', 'Labels', 'Autofix commits'],
  },
  {
    id: 'ACTIONS_BOT_PAT',
    envVar: 'ACTIONS_BOT_PAT',
    type: 'PAT',
    description: 'Bot account PAT for workflow dispatch',
    workflows: ['Workflow dispatch', 'Belt conveyor'],
  },
  {
    id: 'CODESPACES_WORKFLOWS',
    envVar: 'CODESPACES_WORKFLOWS',
    type: 'PAT',
    description: 'Owner PAT for cross-repo operations',
    workflows: ['Cross-repo sync', 'Dependabot automerge', 'Label sync'],
  },
  {
    id: 'OWNER_PR_PAT',
    envVar: 'OWNER_PR_PAT',
    type: 'PAT',
    description: 'Owner PAT for PR creation',
    workflows: ['PR creation as owner'],
  },
  {
    id: 'WORKFLOWS_APP',
    envVar: 'WORKFLOWS_APP_ID',
    privateKeyVar: 'WORKFLOWS_APP_PRIVATE_KEY',
    type: 'GitHub App',
    description: 'General-purpose GitHub App',
    workflows: ['Autofix', 'General workflow operations'],
  },
  {
    id: 'KEEPALIVE_APP',
    envVar: 'KEEPALIVE_APP_ID',
    privateKeyVar: 'KEEPALIVE_APP_PRIVATE_KEY',
    type: 'GitHub App',
    description: 'Dedicated App for keepalive (isolated rate limit)',
    workflows: ['Keepalive loop'],
  },
  {
    id: 'GH_APP',
    envVar: 'GH_APP_ID',
    privateKeyVar: 'GH_APP_PRIVATE_KEY',
    type: 'GitHub App',
    description: 'Bot comment handler, issue intake',
    workflows: ['Issue intake', 'Comment handling'],
  },
];

/**
 * Mint a GitHub App installation token using gh CLI
 */
async function mintAppToken(appId, privateKey) {
  // Note: This requires additional setup and is complex to do via CLI
  // For now, return null and note that App tokens need special handling
  return null;
}

/**
 * Check rate limit for a token using gh CLI
 */
async function checkRateLimit(token) {
  try {
    const result = execSync(
      `gh api rate_limit --header "Authorization: Bearer ${token}"`,
      {
        encoding: 'utf8',
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, GITHUB_TOKEN: token },
      }
    );
    
    const data = JSON.parse(result);
    const core = data.resources.core;
    
    const percentUsed = core.limit > 0 
      ? ((core.used / core.limit) * 100).toFixed(1)
      : 0;
    
    const percentRemaining = core.limit > 0
      ? ((core.remaining / core.limit) * 100).toFixed(1)
      : 0;
    
    const resetDate = new Date(core.reset * 1000);
    const minutesUntilReset = Math.ceil((resetDate - new Date()) / 60000);
    
    return {
      limit: core.limit,
      used: core.used,
      remaining: core.remaining,
      percentUsed,
      percentRemaining,
      reset: resetDate.toISOString(),
      minutesUntilReset,
      status: getStatus(percentRemaining),
    };
  } catch (error) {
    return {
      error: error.message,
      status: 'error',
    };
  }
}

/**
 * Get status label based on remaining percentage
 */
function getStatus(percentRemaining) {
  const percent = parseFloat(percentRemaining);
  if (percent >= 80) return 'healthy';
  if (percent >= 50) return 'moderate';
  if (percent >= 20) return 'low';
  if (percent >= 5) return 'critical';
  return 'exhausted';
}

/**
 * Get status emoji
 */
function getStatusEmoji(status) {
  switch (status) {
    case 'healthy': return '✅';
    case 'moderate': return '⚠️';
    case 'low': return '🔶';
    case 'critical': return '🔴';
    case 'exhausted': return '🚨';
    case 'unavailable': return '❌';
    case 'error': return '❌';
    default: return '❔';
  }
}

/**
 * Diagnose all tokens
 */
async function diagnoseAllTokens() {
  console.log('🔍 API Usage Diagnostic Report');
  console.log('═'.repeat(80));
  console.log('');
  
  const results = [];
  
  for (const config of TOKEN_CONFIG) {
    console.log(`Checking ${config.id}...`);
    
    let token;
    let status = 'unavailable';
    let rateLimit = null;
    
    // Get token value
    if (config.type === 'GitHub App') {
      const appId = process.env[config.envVar];
      const privateKey = process.env[config.privateKeyVar];
      
      if (appId && privateKey) {
        try {
          token = await mintAppToken(appId, privateKey);
          rateLimit = await checkRateLimit(token);
          status = rateLimit.status;
        } catch (error) {
          rateLimit = { error: error.message };
          status = 'error';
        }
      }
    } else {
      token = process.env[config.envVar];
      if (token) {
        rateLimit = await checkRateLimit(token);
        status = rateLimit.status;
      }
    }
    
    results.push({
      ...config,
      available: !!token,
      status,
      rateLimit,
    });
  }
  
  console.log('');
  console.log('📊 Token Status Summary');
  console.log('═'.repeat(80));
  console.log('');
  
  // Table header
  console.log('Token                 Type            Status      Used    Remaining  Reset');
  console.log('─'.repeat(80));
  
  // Table rows
  for (const result of results) {
    if (!result.available) {
      console.log(
        `${result.id.padEnd(20)} ${result.type.padEnd(15)} ❌ Not configured`
      );
      continue;
    }
    
    if (result.rateLimit?.error) {
      console.log(
        `${result.id.padEnd(20)} ${result.type.padEnd(15)} ❌ Error: ${result.rateLimit.error}`
      );
      continue;
    }
    
    const rl = result.rateLimit;
    const emoji = getStatusEmoji(result.status);
    const usedStr = `${rl.used}/${rl.limit}`.padEnd(11);
    const remainingStr = `${rl.remaining} (${rl.percentRemaining}%)`.padEnd(15);
    const resetStr = `${rl.minutesUntilReset}min`;
    
    console.log(
      `${result.id.padEnd(20)} ${result.type.padEnd(15)} ${emoji} ${result.status.padEnd(10)} ${usedStr} ${remainingStr} ${resetStr}`
    );
  }
  
  console.log('');
  console.log('📝 Token Usage by Workflow');
  console.log('═'.repeat(80));
  console.log('');
  
  for (const result of results) {
    if (!result.available) continue;
    
    const emoji = getStatusEmoji(result.status);
    console.log(`${emoji} ${result.id} (${result.status})`);
    console.log(`   Description: ${result.description}`);
    console.log(`   Workflows:`);
    for (const workflow of result.workflows) {
      console.log(`     • ${workflow}`);
    }
    if (result.rateLimit && !result.rateLimit.error) {
      console.log(`   Rate Limit: ${result.rateLimit.remaining}/${result.rateLimit.limit} remaining (${result.rateLimit.percentRemaining}%)`);
    }
    console.log('');
  }
  
  // Generate recommendations
  console.log('💡 Recommendations');
  console.log('═'.repeat(80));
  console.log('');
  
  const critical = results.filter(r => r.status === 'critical' || r.status === 'exhausted');
  const low = results.filter(r => r.status === 'low');
  const unavailable = results.filter(r => !r.available);
  
  if (critical.length > 0) {
    console.log('🚨 CRITICAL: The following tokens are critically low or exhausted:');
    for (const token of critical) {
      console.log(`   • ${token.id}: ${token.rateLimit?.remaining || 0} remaining`);
      console.log(`     Workflows affected: ${token.workflows.join(', ')}`);
      if (token.rateLimit?.minutesUntilReset) {
        console.log(`     Reset in: ${token.rateLimit.minutesUntilReset} minutes`);
      }
    }
    console.log('   Action: Consider pausing non-critical workflows or waiting for reset');
    console.log('');
  }
  
  if (low.length > 0) {
    console.log('⚠️  WARNING: The following tokens have low capacity:');
    for (const token of low) {
      console.log(`   • ${token.id}: ${token.rateLimit?.remaining || 0} remaining (${token.rateLimit?.percentRemaining || 0}%)`);
    }
    console.log('   Action: Monitor usage and consider token rotation');
    console.log('');
  }
  
  if (unavailable.length > 0) {
    console.log('ℹ️  INFO: The following tokens are not configured:');
    for (const token of unavailable) {
      console.log(`   • ${token.id}: ${token.description}`);
      console.log(`     Would handle: ${token.workflows.join(', ')}`);
    }
    console.log('   Note: Workflows may fall back to GITHUB_TOKEN, increasing load');
    console.log('');
  }
  
  // Specific recommendation for keepalive loop
  const keepaliveApp = results.find(r => r.id === 'KEEPALIVE_APP');
  const githubToken = results.find(r => r.id === 'GITHUB_TOKEN');
  
  if (!keepaliveApp?.available && githubToken?.status !== 'healthy') {
    console.log('🔧 KEEPALIVE RECOMMENDATION:');
    console.log('   The keepalive loop may be using GITHUB_TOKEN, which is showing stress.');
    console.log('   Configure KEEPALIVE_APP to provide an isolated rate limit pool:');
    console.log('     1. Create a GitHub App with pull_requests:write permission');
    console.log('     2. Install the app on this repository');
    console.log('     3. Add KEEPALIVE_APP_ID and KEEPALIVE_APP_PRIVATE_KEY secrets');
    console.log('     4. Update keepalive workflow to use the app token');
    console.log('');
  }
  
  // Overall health
  const healthy = results.filter(r => r.status === 'healthy').length;
  const total = results.filter(r => r.available).length;
  
  console.log('📈 Overall Health');
  console.log('═'.repeat(80));
  console.log(`Tokens available: ${total}/${TOKEN_CONFIG.length}`);
  console.log(`Tokens healthy: ${healthy}/${total} (${((healthy/total)*100).toFixed(0)}%)`);
  
  if (critical.length > 0) {
    console.log('Status: 🚨 CRITICAL - Immediate action required');
  } else if (low.length > 0) {
    console.log('Status: ⚠️  WARNING - Monitor closely');
  } else {
    console.log('Status: ✅ HEALTHY - All systems operational');
  }
  
  console.log('');
  console.log('═'.repeat(80));
  console.log('Report generated:', new Date().toISOString());
  
  return results;
}

// Run diagnostic if called directly
if (require.main === module) {
  diagnoseAllTokens()
    .then(() => process.exit(0))
    .catch(error => {
      console.error('Fatal error:', error);
      process.exit(1);
    });
}

module.exports = {
  diagnoseAllTokens,
  checkRateLimit,
  mintAppToken,
};
