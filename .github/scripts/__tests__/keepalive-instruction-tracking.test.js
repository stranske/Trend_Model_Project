'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { runKeepalivePostWork } = require('../keepalive_post_work.js');

/**
 * Test the instruction tracking functionality added for Issue #3532
 * This ensures that keepalive workers re-run when new instructions arrive
 * even if the head SHA is unchanged.
 */

const buildMockCore = () => {
  const outputs = {};
  const warnings = [];
  const info = [];
  const summaryEntries = [];
  
  return {
    setOutput: (name, value) => { outputs[name] = value; },
    warning: (message) => { warnings.push(message); },
    info: (message) => { info.push(message); },
    summary: {
      addHeading: (text) => ({ addTable: () => ({ addRaw: () => ({ addEOL: () => ({ write: async () => {} }) }) }) }),
      addTable: () => ({ write: async () => {} }),
      addRaw: (text) => ({ addEOL: () => ({ write: async () => {} }) }),
      addEOL: () => ({ write: async () => {} }),
      write: async () => {},
    },
    outputs,
    warnings,
    info,
  };
};

const buildMockGithub = ({ 
  headSha = 'sha123', 
  existingState = {}, 
  labelNames = ['agents:keepalive'], 
  fork = false 
} = {}) => {
  let stateComment = null;
  
  return {
    rest: {
      pulls: {
        get: async () => ({
          data: {
            head: { sha: headSha, ref: 'codex/issue-1' },
            base: { ref: 'main' },
            user: { login: 'stranske-automation-bot' },
            head: {
              sha: headSha,
              ref: 'codex/issue-1',
              repo: { fork }
            }
          }
        })
      },
      issues: {
        listLabelsOnIssue: async () => ({
          data: labelNames.map(name => ({ name }))
        }),
        listComments: async () => ({
          data: stateComment ? [stateComment] : []
        }),
        createComment: async ({ body }) => {
          stateComment = {
            id: 12345,
            body,
            html_url: 'https://github.com/test/repo/issues/1#issuecomment-12345'
          };
          return { data: stateComment };
        },
        updateComment: async ({ comment_id, body }) => {
          if (stateComment && stateComment.id === comment_id) {
            stateComment.body = body;
          }
          return { data: { id: comment_id, body } };
        }
      }
    },
    paginate: async (fn, params) => {
      const result = await fn(params);
      return Array.isArray(result?.data) ? result.data : [];
    }
  };
};

const buildMockContext = () => ({
  repo: { owner: 'testowner', repo: 'testrepo' }
});

test('instruction tracking persists new comment/head tuple', async () => {
  const core = buildMockCore();
  const github = buildMockGithub({ headSha: 'abc123' });
  const context = buildMockContext();
  
  const env = {
    TRACE: 'test-trace',
    ROUND: '1',
    PR_NUMBER: '100',
    COMMENT_ID: '98765',
    COMMENT_URL: 'https://github.com/test/repo/issues/1#issuecomment-98765',
    AGENT_STATE: 'done'
  };

  await runKeepalivePostWork({ core, github, context, env });

  // Check that the instruction was tracked
  assert(core.info.some(msg => msg.includes('updated: comment=98765')));
  assert.equal(core.outputs.success, 'false'); // Should fail due to no actual head movement in test
});

test('instruction tracking detects same comment/head combination', async () => {
  const core = buildMockCore();
  const github = buildMockGithub({ 
    headSha: 'abc123',
    existingState: {
      last_instruction: {
        comment_id: 98765,
        head_sha: 'abc123',
        processed_at: '2025-01-15T10:00:00Z'
      }
    }
  });
  const context = buildMockContext();
  
  const env = {
    TRACE: 'test-trace',
    ROUND: '1',
    PR_NUMBER: '100',
    COMMENT_ID: '98765', // Same comment ID as stored
    COMMENT_URL: 'https://github.com/test/repo/issues/1#issuecomment-98765',
    AGENT_STATE: 'done'
  };

  await runKeepalivePostWork({ core, github, context, env });

  // Should reuse existing instruction tracking
  assert(core.info.some(msg => msg.includes('reused: comment=98765')));
});

test('instruction tracking detects new comment with same head', async () => {
  const core = buildMockCore();
  const github = buildMockGithub({ 
    headSha: 'abc123',
    existingState: {
      last_instruction: {
        comment_id: 11111, // Different comment ID
        head_sha: 'abc123', // Same head SHA
        processed_at: '2025-01-15T10:00:00Z'
      }
    }
  });
  const context = buildMockContext();
  
  const env = {
    TRACE: 'test-trace',
    ROUND: '1',
    PR_NUMBER: '100',
    COMMENT_ID: '98765', // New comment ID
    COMMENT_URL: 'https://github.com/test/repo/issues/1#issuecomment-98765',
    AGENT_STATE: 'done'
  };

  await runKeepalivePostWork({ core, github, context, env });

  // Should update instruction tracking due to new comment
  assert(core.info.some(msg => msg.includes('updated: comment=98765')));
});

test('instruction tracking handles missing comment gracefully', async () => {
  const core = buildMockCore();
  const github = buildMockGithub({ headSha: 'abc123' });
  const context = buildMockContext();
  
  const env = {
    TRACE: 'test-trace',
    ROUND: '1',
    PR_NUMBER: '100',
    // No COMMENT_ID provided
    AGENT_STATE: 'done'
  };

  await runKeepalivePostWork({ core, github, context, env });

  // Should handle missing comment gracefully
  assert(core.info.some(msg => msg.includes('unavailable; proceeding without comment context')));
});