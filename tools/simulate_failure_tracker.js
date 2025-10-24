#!/usr/bin/env node

const assert = require('assert');
const maintPostCi = require('../.github/scripts/maint-post-ci.js');

function createCore(state) {
  const summary = {
    addHeading() {
      return this;
    },
    addRaw() {
      return this;
    },
    async write() {
      return undefined;
    },
  };
  return {
    info(message) {
      state.log.push(String(message));
    },
    summary,
  };
}

function extractOccurrences(body) {
  const match = body.match(/Occurrences:\s*(\d+)/i);
  return match ? parseInt(match[1], 10) : 0;
}

function createGithub(state, options) {
  const { searchHasIssue } = options;
  const failedJobs = [
    {
      id: 9001,
      name: 'Tests',
      conclusion: 'failure',
      html_url: `https://ci.example/jobs/${state.runIndex + 1}`,
      steps: [
        { name: 'pytest', conclusion: 'failure' },
      ],
    },
  ];

  return {
    rest: {
      actions: {
        async listJobsForWorkflowRun() {
          return { data: { jobs: failedJobs } };
        },
      },
      issues: {
        async getLabel({ name }) {
          if (!state.availableLabels.has(name)) {
            throw Object.assign(new Error('Not Found'), { status: 404 });
          }
          return { data: { name } };
        },
        async createLabel({ name }) {
          state.availableLabels.add(name);
          state.log.push(`label created: ${name}`);
          return { data: { name } };
        },
        async get({ issue_number }) {
          if (!state.issue || state.issue.number !== issue_number) {
            throw new Error(`Issue ${issue_number} not found in simulation.`);
          }
          return {
            data: {
              number: state.issue.number,
              title: state.issue.title,
              body: state.issue.body,
              labels: Array.from(state.issue.labels).map((name) => ({ name })),
            },
          };
        },
        async update({ issue_number, body, title }) {
          if (!state.issue || state.issue.number !== issue_number) {
            throw new Error(`Update called for unexpected issue ${issue_number}.`);
          }
          state.issue.body = body;
          state.issue.title = title;
          state.issue.updated_at = new Date(state.now).toISOString();
          return { data: state.issue };
        },
        async listComments({ issue_number }) {
          const comments = state.comments.filter((c) => c.issue_number === issue_number);
          return {
            data: comments.map((c) => ({ body: c.body, created_at: c.created_at })),
          };
        },
        async createComment({ issue_number, body }) {
          const created = {
            issue_number,
            body,
            created_at: new Date(state.now).toISOString(),
          };
          state.comments.push(created);
          state.log.push(`comment appended to #${issue_number}`);
          return { data: created };
        },
        async addLabels({ issue_number, labels }) {
          if (!state.issue || state.issue.number !== issue_number) {
            throw new Error(`addLabels called for unexpected issue ${issue_number}.`);
          }
          labels.forEach((label) => state.issue.labels.add(label));
          state.log.push(`labels added to #${issue_number}: ${labels.join(', ')}`);
          return { data: { labels: Array.from(state.issue.labels) } };
        },
        async listForRepo() {
          if (!state.issue) {
            return { data: [] };
          }
          return {
            data: [
              {
                number: state.issue.number,
                title: state.issue.title,
                created_at: state.issue.created_at,
                pull_request: null,
              },
            ],
          };
        },
        async create({ title, body, labels }) {
          const number = state.issue ? state.issue.number : state.nextIssueNumber;
          state.issue = {
            number,
            title,
            body,
            labels: new Set(labels),
            created_at: new Date(state.now).toISOString(),
            updated_at: new Date(state.now).toISOString(),
          };
          state.issueCreationCount += 1;
          state.log.push(`issue #${number} created`);
          return { data: state.issue };
        },
      },
      search: {
        async issuesAndPullRequests() {
          if (searchHasIssue && state.issue) {
            return {
              data: {
                items: [
                  {
                    number: state.issue.number,
                    title: state.issue.title,
                  },
                ],
              },
            };
          }
          return { data: { items: [] } };
        },
      },
    },
    async request() {
      return { data: Buffer.from('') };
    },
  };
}

async function main() {
  const state = {
    availableLabels: new Set(['ci-failure', 'ci', 'devops', 'priority: medium']),
    baseTime: new Date('2025-01-01T00:00:00Z'),
    comments: [],
    issue: null,
    issueCreationCount: 0,
    log: [],
    nextIssueNumber: 321,
    now: new Date('2025-01-01T00:00:00Z'),
    runIndex: 0,
  };

  const runs = [
    { label: 'First failure', offsetHours: 0, searchHasIssue: false },
    { label: 'Second failure', offsetHours: 2, searchHasIssue: true },
    { label: 'Third failure', offsetHours: 4, searchHasIssue: true },
  ];

  const realDateNow = Date.now;

  try {
    process.env.PR_NUMBER = '1234';
    for (let i = 0; i < runs.length; i += 1) {
      const run = runs[i];
      state.runIndex = i;
      state.now = new Date(state.baseTime.getTime() + run.offsetHours * 3600 * 1000);
      Date.now = () => state.now.getTime();

      const core = createCore(state);
      const github = createGithub(state, run);
      const context = {
        repo: { owner: 'stranske', repo: 'Trend_Model_Project' },
        payload: {
          workflow_run: {
            id: 5000 + i,
            name: 'CI',
            html_url: `https://ci.example/runs/${i + 1}`,
          },
        },
      };

      // ensure script reads the tuned defaults
      process.env.RATE_LIMIT_MINUTES = '15';
      process.env.STACK_TOKENS_ENABLED = 'false';
      process.env.STACK_TOKEN_MAX_LEN = '160';
      process.env.NEW_ISSUE_COOLDOWN_HOURS = '12';
      process.env.COOLDOWN_SCOPE = 'global';
      process.env.COOLDOWN_RETRY_MS = '0';
      process.env.OCCURRENCE_ESCALATE_THRESHOLD = '3';
      process.env.AUTO_HEAL_INACTIVITY_HOURS = '24';

      await maintPostCi.updateFailureTracker({ github, context, core });
      await maintPostCi.resolveFailureIssuesForRecoveredPR({ github, context, core });
      await maintPostCi.autoHealFailureIssues({ github, context, core });
      await maintPostCi.snapshotFailureIssues({ github, context, core });
      await maintPostCi.applyCiFailureLabel({ github, context, core });
      await maintPostCi.removeCiFailureLabel({ github, context, core });

      if (!state.issue) {
        throw new Error('Tracker did not create or update an issue.');
      }

      const occurrences = extractOccurrences(state.issue.body);
      assert.strictEqual(occurrences, i + 1, `Unexpected occurrence count after ${run.label.toLowerCase()}`);
      const labels = Array.from(state.issue.labels).sort();
      console.log(`${run.label}: occurrences=${occurrences}, labels=${labels.join(', ')}`);
    }
  } finally {
    Date.now = realDateNow;
  }

  assert.strictEqual(state.issueCreationCount, 1, 'Cooldown should prevent spawning duplicate issues.');

  const escalationComments = state.comments.filter((c) => c.body.includes('Escalation: occurrences reached threshold'));
  assert.strictEqual(escalationComments.length, 1, 'Escalation comment should be posted exactly once.');
  assert.ok(state.issue.labels.has('priority: high'), 'Escalation label was not applied at the threshold.');
  assert.ok(state.issue.labels.has('priority: medium'), 'Base priority label should remain alongside escalation label.');

  const createdLabelLogs = state.log.filter((entry) => entry.startsWith('label created: '));
  assert(
    createdLabelLogs.some((entry) => entry.includes('priority: high')),
    'Escalation label should be auto-created when missing.'
  );
  assert.ok(state.availableLabels.has('priority: high'), 'Escalation label should remain available after creation.');

  console.log('Escalation comment posted:', escalationComments[0].body.trim());
  console.log('Simulation completed successfully.');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
