#!/usr/bin/env node
'use strict';

const fs = require('fs');
const { detectKeepalive } = require('../../../.github/scripts/agents_pr_meta_keepalive.js');

async function main() {
  const [, , scenarioPath] = process.argv;
  if (!scenarioPath) {
    throw new Error('Scenario path is required');
  }

  const scenario = JSON.parse(fs.readFileSync(scenarioPath, 'utf8'));

  const info = [];
  const warnings = [];
  const outputs = {};

  const core = {
    info: (message) => info.push(String(message)),
    warning: (message) => warnings.push(String(message)),
    setOutput: (name, value) => {
      outputs[name] = String(value);
    },
  };

  const repo = scenario.repo || { owner: 'stranske', repo: 'Trend_Model_Project' };

  const comment = scenario.comment || {};
  const issueNumber = scenario.issue?.number ?? scenario.pull?.number ?? 0;

  const context = {
    payload: {
      comment: {
        body: comment.body || '',
        user: { login: comment.user?.login || 'stranske-automation-bot' },
      },
      issue: { number: issueNumber },
      repository: scenario.repository || { default_branch: 'phase-2-dev' },
    },
    repo,
  };

  const pull = scenario.pull || {};
  const pullError = scenario.pullError;

  const github = {
    rest: {
      pulls: {
        get: async ({ pull_number }) => {
          if (pullError) {
            throw new Error(pullError);
          }
          const data = {
            number: pull.number ?? pull_number,
            title: pull.title || '',
            body: pull.body || '',
            head: { ref: pull.head?.ref || '' },
            base: { ref: pull.base?.ref || '' },
          };
          return { data };
        },
      },
    },
  };

  const env = {
    ALLOWED_LOGINS:
      scenario.env?.ALLOWED_LOGINS || 'chatgpt-codex-connector,stranske-automation-bot',
    KEEPALIVE_MARKER: scenario.env?.KEEPALIVE_MARKER || '<!-- codex-keepalive-marker -->',
  };

  const result = await detectKeepalive({ core, github, context, env });

  const payload = {
    outputs,
    info,
    warnings,
    result,
  };

  process.stdout.write(JSON.stringify(payload));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
