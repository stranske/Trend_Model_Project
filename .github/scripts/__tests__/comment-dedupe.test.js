'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  extractAnchor,
  findExistingComment,
  upsertSummaryComment,
} = require('../comment-dedupe.js');

test('extractAnchor parses PR and head identifiers', () => {
  const anchor = extractAnchor('hello <!-- maint-46-post-ci: pr=42 head=abcdef123456 --> world');
  assert.deepEqual(anchor, { raw: '<!-- maint-46-post-ci: pr=42 head=abcdef123456 -->', pr: '42', head: 'abcdef123456' });
  assert.equal(extractAnchor('no marker'), null);
});

test('findExistingComment prefers matching anchor over legacy markers', () => {
  const comments = [
    { id: 1, body: 'legacy <!-- maint-46-post-ci: pr=41 head=deadbeef -->' },
    { id: 2, body: 'new <!-- maint-46-post-ci: pr=42 head=feedface -->' },
    { id: 3, body: 'plain <!-- maint-46-post-ci: -->' },
  ];
  const targetAnchor = { pr: '42', head: 'feedface' };
  const match = findExistingComment(comments, targetAnchor, '<!-- maint-46-post-ci:');
  assert.equal(match.id, 2);

  const fallback = findExistingComment([{ id: 4, body: 'text <!-- maint-46-post-ci:' }], null, '<!-- maint-46-post-ci:');
  assert.equal(fallback.id, 4);
});

test('upsertSummaryComment updates existing comment with matching anchor', async () => {
  const updates = [];
  const github = {
    rest: {
      issues: {
        listComments: () => {},
        updateComment: async (payload) => {
          updates.push(payload);
          return {};
        },
        createComment: async () => {
          throw new Error('Should not create new comment when anchor matches');
        },
      },
    },
    paginate: async () => [
      { id: 100, body: 'old <!-- maint-46-post-ci: pr=5 head=abc -->' },
    ],
  };
  const info = [];
  const warnings = [];
  const core = {
    info: (msg) => info.push(msg),
    warning: (msg) => warnings.push(msg),
  };
  const context = { repo: { owner: 'octo', repo: 'example' } };

  const body = 'new body <!-- maint-46-post-ci: pr=5 head=abc -->';
  const result = await upsertSummaryComment({ github, context, core, issueNumber: 5, body });
  assert.equal(result.updated, true);
  assert.equal(result.created, false);
  assert.equal(updates[0].comment_id, 100);
  assert.equal(updates[0].body, body);
  assert.equal(warnings.length, 0);
  assert.ok(info.some((line) => line.includes('Updated existing')));
});

test('upsertSummaryComment warns when PR number missing', async () => {
  const github = {
    paginate: async () => [],
    rest: { issues: {} },
  };
  const warnings = [];
  const core = {
    warning: (msg) => warnings.push(msg),
  };
  const result = await upsertSummaryComment({ github, context: {}, core, issueNumber: null, body: 'body' });
  assert.equal(result.updated, false);
  assert.equal(result.created, false);
  assert.ok(warnings[0].includes('PR number'));
});
