'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { upsertSummaryComment, extractAnchor } = require('../comment-dedupe');

function createCore() {
  return {
    infoMessages: [],
    warnings: [],
    failures: [],
    info(message) {
      this.infoMessages.push(message);
    },
    warning(message) {
      this.warnings.push(message);
    },
    setFailed(message) {
      this.failures.push(message);
    },
  };
}

test('extractAnchor parses tokens', () => {
  const anchor = extractAnchor('Hello <!-- maint-46-post-ci: pr=42 head=abcdef --> world', /<!--\s*maint-46-post-ci:([^>]*)-->/i);
  assert.deepEqual(anchor, { raw: '<!-- maint-46-post-ci: pr=42 head=abcdef -->', pr: '42', head: 'abcdef' });
});

test('updates existing matching comment', async () => {
  const core = createCore();
  const actions = [];
  const comments = [
    { id: 1, body: '<!-- maint-46-post-ci: pr=10 head=abc -->\nold body' },
  ];
  const github = {
    paginate: async () => comments,
    rest: {
      issues: {
        updateComment: async (payload) => { actions.push({ type: 'update', payload }); },
        createComment: async () => { throw new Error('should not create'); },
      },
    },
  };

  const body = '<!-- maint-46-post-ci: pr=10 head=abc -->\nnew body';
  const result = await upsertSummaryComment({ github, context: { repo: {}, }, core }, { body, prNumber: 10 });
  assert.equal(result.action, 'updated');
  assert.equal(actions.length, 1);
  assert.equal(actions[0].payload.body, body);
});

test('creates new comment when none match', async () => {
  const core = createCore();
  const actions = [];
  const github = {
    paginate: async () => [{ id: 1, body: 'no marker' }],
    rest: {
      issues: {
        updateComment: async () => { throw new Error('should not update'); },
        createComment: async (payload) => {
          actions.push({ type: 'create', payload });
          return { data: { id: 99 } };
        },
      },
    },
  };

  const body = '<!-- maint-46-post-ci: pr=5 head=ffff -->\nsummary';
  const result = await upsertSummaryComment({ github, context: { repo: {} }, core }, { body, prNumber: 5 });
  assert.equal(result.action, 'created');
  assert.equal(actions.length, 1);
  assert.equal(actions[0].payload.body, body);
});

test('skips update when bodies match', async () => {
  const core = createCore();
  const github = {
    paginate: async () => [{ id: 7, body: '<!-- maint-46-post-ci: pr=3 head=abc -->\nsummary' }],
    rest: { issues: { updateComment: async () => { throw new Error('should not update'); }, createComment: async () => { throw new Error('should not create'); } } },
  };
  const body = '<!-- maint-46-post-ci: pr=3 head=abc -->\nsummary';
  const result = await upsertSummaryComment({ github, context: { repo: {} }, core }, { body, prNumber: 3 });
  assert.equal(result.action, 'noop');
});
