#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const Module = require('module');

class SummaryRecorder {
  constructor() {
    this.entries = [];
    this.written = false;
  }

  addHeading(text, level = 1) {
    this.entries.push({ type: 'heading', text: String(text), level });
    return this;
  }

  addRaw(text) {
    this.entries.push({ type: 'raw', text: String(text) });
    return this;
  }

  addEOL() {
    this.entries.push({ type: 'eol' });
    return this;
  }

  addDetails(title, items) {
    this.entries.push({ type: 'details', title: String(title), items: Array.from(items).map(String) });
    return this;
  }

  addLink(text, href) {
    this.entries.push({ type: 'link', text: String(text), href: String(href) });
    return this;
  }

  addList(items) {
    this.entries.push({ type: 'list', items: Array.from(items).map(String) });
    return this;
  }

  addTable(rows) {
    const normalised = Array.from(rows).map((row) =>
      Array.isArray(row)
        ? row.map((cell) => (typeof cell === 'object' && cell !== null ? cell.data ?? '' : String(cell)))
        : row
    );
    this.entries.push({ type: 'table', rows: normalised });
    return this;
  }

  async write() {
    this.written = true;
    return this;
  }

  toJSON() {
    return { entries: this.entries, written: this.written };
  }
}

function loadKeepaliveRunner() {
  const targetPath = path.resolve(__dirname, '../../../scripts/keepalive-runner.js');
  const code = fs.readFileSync(targetPath, 'utf8');
  const sandbox = {
    module: { exports: {} },
    exports: {},
    require: Module.createRequire(targetPath),
    __dirname: path.dirname(targetPath),
    __filename: targetPath,
    process,
    console,
    Date,
  };
  vm.createContext(sandbox);
  const wrapper = Module.wrap(code);
  const script = new vm.Script(wrapper, { filename: targetPath });
  const compiled = script.runInContext(sandbox);
  compiled.call(sandbox.exports, sandbox.exports, sandbox.require, sandbox.module, sandbox.__filename, sandbox.__dirname);
  return sandbox.module.exports;
}

function normaliseComment(comment) {
  if (!comment) {
    return null;
  }
  if (typeof comment !== 'object') {
    return null;
  }
  const login = comment.user && typeof comment.user === 'object' ? comment.user.login : undefined;
  return {
    body: comment.body || '',
    created_at: comment.created_at || comment.updated_at || new Date().toISOString(),
    updated_at: comment.updated_at || null,
    user: { login: login || '' },
  };
}

async function runScenario(scenario) {
  const summary = new SummaryRecorder();
  const info = [];
  const warnings = [];
  const notices = [];
  let failedMessage = null;

  const core = {
    summary,
    info: (message) => info.push(String(message)),
    warning: (message) => warnings.push(String(message)),
    notice: (message) => notices.push(String(message)),
    setFailed: (message) => {
      failedMessage = String(message);
    },
  };

  const pulls = Array.from(scenario.pulls || []).map((pull) => ({
    number: pull.number,
    labels: Array.from(pull.labels || []).map((label) =>
      typeof label === 'string' ? { name: label } : label
    ),
  }));

  const commentMap = new Map();
  for (const pull of scenario.pulls || []) {
    const normalised = Array.from(pull.comments || []).map((comment) => normaliseComment(comment)).filter(Boolean);
    commentMap.set(pull.number, normalised);
  }

  const createdComments = [];

  const listPulls = async ({ per_page = 50, page = 1 }) => {
    const start = (page - 1) * per_page;
    const slice = pulls.slice(start, start + per_page);
    return { data: slice };
  };

  const listComments = async ({ issue_number }) => {
    return { data: commentMap.get(issue_number) || [] };
  };

  const createComment = async ({ issue_number, body }) => {
    const entry = { issue_number, body };
    createdComments.push(entry);
    return { data: entry };
  };

  const github = {
    rest: {
      pulls: { list: listPulls },
      issues: {
        listComments,
        createComment,
      },
    },
    paginate: {
      iterator: (method, params) => {
        if (method !== listPulls) {
          throw new Error('Unsupported paginate target');
        }
        const perPage = params.per_page || 50;
        let page = 1;
        return {
          async *[Symbol.asyncIterator]() {
            while (true) {
              const start = (page - 1) * perPage;
              if (start >= pulls.length) {
                break;
              }
              const response = await method({ ...params, page });
              page += 1;
              yield response;
            }
          },
        };
      },
    },
  };

  const context = {
    repo: {
      owner: scenario.repo?.owner || 'owner',
      repo: scenario.repo?.repo || 'repo',
    },
  };

  const originalEnv = {};
  const envOverrides = scenario.env || {};
  for (const [key, value] of Object.entries(envOverrides)) {
    originalEnv[key] = process.env[key];
    process.env[key] = String(value);
  }

  const originalNow = Date.now;
  if (scenario.now) {
    const fixed = new Date(scenario.now).getTime();
    Date.now = () => fixed;
  }

  try {
    const { runKeepalive } = loadKeepaliveRunner();
    await runKeepalive({ core, github, context, env: process.env });
  } finally {
    Date.now = originalNow;
    for (const [key, value] of Object.entries(originalEnv)) {
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
  }

  return {
    summary: summary.toJSON(),
    logs: { info, warnings, notices, failedMessage },
    created_comments: createdComments,
  };
}

async function main() {
  const scenarioPath = process.argv[2];
  if (!scenarioPath) {
    console.error('Usage: node harness.js <scenario.json>');
    process.exit(2);
  }

  let scenario;
  try {
    scenario = JSON.parse(fs.readFileSync(scenarioPath, 'utf8'));
  } catch (error) {
    console.error('Failed to load scenario:', error.message);
    process.exit(2);
  }

  try {
    const result = await runScenario(scenario);
    process.stdout.write(JSON.stringify(result));
  } catch (error) {
    console.error('Harness execution failed:', error.stack || error.message);
    process.exit(1);
  }
}

main();
