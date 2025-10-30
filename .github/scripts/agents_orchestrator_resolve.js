'use strict';

const DEFAULT_READINESS_AGENTS = 'copilot,codex';
const DEFAULT_VERIFY_ISSUE_ASSIGNEES =
  'copilot,chatgpt-codex-connector,stranske-automation-bot';
const DEFAULT_KEEPALIVE_INSTRUCTION = '@codex use the scope, acceptance criteria, and task list so the keepalive workflow continues nudging until everything is complete. Work through the tasks, checking them off only after each acceptance criterion is satisfied, but check during each comment implementation and check off tasks and acceptance criteria that have been satisfied and repost the current version of the initial scope, task list and acceptance criteria each time that any have been newly completed.';
const DEFAULT_OPTIONS_JSON = '{}';
const KEEPALIVE_PAUSE_LABEL = 'keepalive:paused';

const DEFAULTS = {
  enable_readiness: 'false',
  readiness_agents: DEFAULT_READINESS_AGENTS,
  readiness_custom_logins: '',
  require_all: 'false',
  enable_preflight: 'false',
  codex_user: '',
  codex_command_phrase: '',
  enable_verify_issue: 'false',
  verify_issue_number: '',
  verify_issue_valid_assignees: DEFAULT_VERIFY_ISSUE_ASSIGNEES,
  enable_watchdog: 'true',
  enable_keepalive: 'true',
  enable_bootstrap: 'false',
  bootstrap_issues_label: 'agent:codex',
  draft_pr: 'false',
  diagnostic_mode: 'off',
  options_json: DEFAULT_OPTIONS_JSON,
  dry_run: 'false',
  dispatcher_force_issue: '',
  worker_max_parallel: '1',
  conveyor_max_merges: '2'
};

const toString = (value, fallback = '') => {
  if (value === undefined || value === null) {
    return fallback;
  }
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean).join(',');
  }
  return String(value);
};

const toBoolString = (value, fallback) => {
  const candidate = value === undefined ? fallback : value;
  if (typeof candidate === 'boolean') {
    return candidate ? 'true' : 'false';
  }
  if (typeof candidate === 'number') {
    return candidate !== 0 ? 'true' : 'false';
  }
  if (typeof candidate === 'string') {
    const norm = candidate.trim().toLowerCase();
    if (['true', '1', 'yes', 'y', 'on'].includes(norm)) {
      return 'true';
    }
    if (['false', '0', 'no', 'n', 'off', ''].includes(norm)) {
      return 'false';
    }
  }
  return fallback === 'true' || fallback === true ? 'true' : 'false';
};

const toCsv = (value, fallback = '') => {
  if (value === undefined || value === null) {
    return fallback;
  }
  const raw = Array.isArray(value)
    ? value
    : typeof value === 'string'
      ? value.split(',')
      : [];
  const cleaned = raw
    .map((entry) => String(entry).trim())
    .filter(Boolean);
  if (!cleaned.length) {
    return fallback;
  }
  return cleaned.join(',');
};

const nested = (value) => (value && typeof value === 'object' ? value : {});

const toBoundedIntegerString = (value, fallback, bounds = {}) => {
  const { min, max } = bounds;
  const fallbackNumber = Number(fallback);
  let candidate = Number(value);
  if (!Number.isFinite(candidate)) {
    candidate = Number.isFinite(fallbackNumber) ? fallbackNumber : 0;
  }
  if (Number.isFinite(min) && candidate < min) {
    candidate = min;
  }
  if (Number.isFinite(max) && candidate > max) {
    candidate = max;
  }
  if (!Number.isFinite(candidate)) {
    candidate = 0;
  }
  return String(Math.max(0, Math.floor(candidate)));
};

const sanitiseOptions = (core, value) => {
  if (value === undefined || value === null || value === '') {
    return DEFAULT_OPTIONS_JSON;
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) {
      return DEFAULT_OPTIONS_JSON;
    }
    try {
      const parsed = JSON.parse(trimmed);
      return JSON.stringify(parsed);
    } catch (error) {
      core.warning(`options_json is not valid JSON (${error.message}); using default.`);
      return DEFAULT_OPTIONS_JSON;
    }
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch (error) {
      core.warning(`options_json could not be serialised (${error.message}); using default.`);
      return DEFAULT_OPTIONS_JSON;
    }
  }
  return DEFAULT_OPTIONS_JSON;
};

const summarise = (value) => {
  const text = String(value ?? '');
  const limit = 120;
  const separator = ' … ';
  if (text.length <= limit) {
    return text;
  }
  const available = limit - separator.length;
  const headLen = Math.ceil(available / 2);
  const tailLen = Math.floor(available / 2);
  const head = text.slice(0, headLen).trimEnd();
  const tail = text.slice(-tailLen).trimStart();
  return `${head}${separator}${tail}`;
};

async function resolveOrchestratorParams({ github, context, core, env = process.env }) {
  let user = {};
  try {
    const parsed = JSON.parse(env.PARAMS_JSON || '{}');
    if (parsed && typeof parsed === 'object') {
      user = parsed;
    }
  } catch (error) {
    core.warning(`Bad params_json; using defaults. Parse error: ${error.message}`);
  }

  const merged = { ...DEFAULTS, ...user };

  const workflowDryRun = env.WORKFLOW_DRY_RUN;
  if (workflowDryRun !== undefined && workflowDryRun !== null && workflowDryRun !== '') {
    merged.dry_run = workflowDryRun;
  }

  const workflowOptionsJson = env.WORKFLOW_OPTIONS_JSON;
  if (workflowOptionsJson !== undefined && workflowOptionsJson !== null && workflowOptionsJson.trim() !== '') {
    merged.options_json = workflowOptionsJson;
  }

  const workflowKeepaliveEnabled = env.WORKFLOW_KEEPALIVE_ENABLED;
  if (workflowKeepaliveEnabled !== undefined && workflowKeepaliveEnabled !== null && workflowKeepaliveEnabled !== '') {
    merged.enable_keepalive = workflowKeepaliveEnabled;
  }

  const readinessAgents = toCsv(merged.readiness_agents, DEFAULTS.readiness_agents);
  const readinessCustom = toCsv(
    merged.readiness_custom_logins ?? merged.readiness_custom ?? merged.custom_logins,
    DEFAULTS.readiness_custom_logins
  );
  const codexUser = toString(merged.codex_user, DEFAULTS.codex_user);
  const codexCommand = toString(merged.codex_command_phrase, DEFAULTS.codex_command_phrase);
  const verifyIssueNumber = toString(merged.verify_issue_number, DEFAULTS.verify_issue_number).trim();
  const verifyIssueAssignees = toCsv(
    merged.verify_issue_valid_assignees ?? merged.valid_assignees,
    DEFAULT_VERIFY_ISSUE_ASSIGNEES
  );

  const bootstrap = nested(merged.bootstrap);
  const keepalive = nested(merged.keepalive);

  const dryRun = toBoolString(merged.dry_run, DEFAULTS.dry_run);

  const diagnosticModeRaw = toString(merged.diagnostic_mode, DEFAULTS.diagnostic_mode).trim().toLowerCase();
  const diagnosticMode = ['full', 'dry-run'].includes(diagnosticModeRaw) ? diagnosticModeRaw : 'off';

  const enableVerifyIssue = toBoolString(
    merged.enable_verify_issue,
    verifyIssueNumber !== '' ? 'true' : DEFAULTS.enable_verify_issue
  );

  const optionsSource = merged.options_json ?? merged.options ?? DEFAULT_OPTIONS_JSON;
  const sanitisedOptions = sanitiseOptions(core, optionsSource);

  let parsedOptions = {};
  try {
    parsedOptions = JSON.parse(sanitisedOptions);
  } catch (error) {
    core.warning(`options_json could not be parsed (${error.message}); using defaults.`);
  }

  // Inject default keepalive instruction if not already present
  if (!parsedOptions.keepalive_instruction && !parsedOptions.keepalive_instruction_template) {
    parsedOptions.keepalive_instruction = DEFAULT_KEEPALIVE_INSTRUCTION;
  }

  // Re-serialize with injected defaults
  const finalOptionsJson = JSON.stringify(parsedOptions);

  const beltOptions = nested(parsedOptions.belt ?? parsedOptions.codex_belt);
  const dispatcherOptions = nested(beltOptions.dispatcher ?? parsedOptions.dispatcher);
  const workerOptions = nested(beltOptions.worker ?? parsedOptions.worker);
  const conveyorOptions = nested(beltOptions.conveyor ?? parsedOptions.conveyor);

  const dispatcherForceIssue = toString(
    dispatcherOptions.force_issue ?? merged.dispatcher_force_issue,
    DEFAULTS.dispatcher_force_issue
  );

  const workerMaxParallel = toBoundedIntegerString(
    workerOptions.max_parallel ?? workerOptions.parallel ?? merged.worker_max_parallel,
    DEFAULTS.worker_max_parallel,
    { min: 0, max: 5 }
  );

  const conveyorMaxMerges = toBoundedIntegerString(
    conveyorOptions.max_merges ?? conveyorOptions.limit ?? merged.conveyor_max_merges,
    DEFAULTS.conveyor_max_merges,
    { min: 0, max: 5 }
  );

  const keepaliveRequested = toBoolString(
    merged.enable_keepalive ?? keepalive.enabled,
    DEFAULTS.enable_keepalive
  );

  const { owner, repo } = context.repo;
  let keepalivePaused = false;

  if (keepaliveRequested === 'true') {
    try {
      await github.rest.issues.getLabel({ owner, repo, name: KEEPALIVE_PAUSE_LABEL });
      keepalivePaused = true;
      core.info(`keepalive skipped: repository label "${KEEPALIVE_PAUSE_LABEL}" is present.`);
    } catch (error) {
      if (error && error.status === 404) {
        core.info(`Keepalive pause label "${KEEPALIVE_PAUSE_LABEL}" not present; keepalive remains enabled.`);
      } else {
        const message = error instanceof Error ? error.message : String(error);
        core.warning(`Unable to resolve keepalive pause label (${message}); proceeding with keepalive.`);
      }
    }
  } else {
    core.info('Keepalive disabled via configuration; skipping pause label check.');
  }

  const keepaliveEffective = keepalivePaused ? 'false' : keepaliveRequested;

  const outputs = {
    enable_readiness: toBoolString(merged.enable_readiness, DEFAULTS.enable_readiness),
    readiness_agents: readinessAgents,
    readiness_custom_logins: readinessCustom,
    require_all: toBoolString(merged.require_all, DEFAULTS.require_all),
    enable_preflight: toBoolString(merged.enable_preflight, DEFAULTS.enable_preflight),
    codex_user: codexUser,
    codex_command_phrase: codexCommand,
    enable_diagnostic: diagnosticMode === 'off' ? 'false' : 'true',
    diagnostic_attempt_branch: diagnosticMode === 'full' ? 'true' : 'false',
    diagnostic_dry_run: diagnosticMode === 'full' ? 'false' : 'true',
    enable_verify_issue: enableVerifyIssue,
    verify_issue_number: verifyIssueNumber,
    verify_issue_valid_assignees: verifyIssueAssignees,
    enable_watchdog: toBoolString(merged.enable_watchdog, DEFAULTS.enable_watchdog),
    enable_keepalive: keepaliveEffective,
    keepalive_requested: keepaliveRequested,
    keepalive_paused_label: keepalivePaused ? 'true' : 'false',
    keepalive_pause_label: KEEPALIVE_PAUSE_LABEL,
    enable_bootstrap: toBoolString(merged.enable_bootstrap ?? bootstrap.enable, DEFAULTS.enable_bootstrap),
    bootstrap_issues_label: toString(
      merged.bootstrap_issues_label ?? bootstrap.label,
      DEFAULTS.bootstrap_issues_label
    ),
    draft_pr: toBoolString(merged.draft_pr, DEFAULTS.draft_pr),
    dry_run: dryRun,
    options_json: finalOptionsJson,
    dispatcher_force_issue: dispatcherForceIssue,
    worker_max_parallel: workerMaxParallel,
    conveyor_max_merges: conveyorMaxMerges
  };

  const orderedKeys = [
    'enable_readiness',
    'readiness_agents',
    'readiness_custom_logins',
    'require_all',
    'enable_preflight',
    'codex_user',
    'codex_command_phrase',
    'enable_diagnostic',
    'diagnostic_attempt_branch',
    'diagnostic_dry_run',
    'enable_verify_issue',
    'verify_issue_number',
    'verify_issue_valid_assignees',
    'enable_watchdog',
    'enable_keepalive',
    'keepalive_requested',
    'keepalive_paused_label',
    'keepalive_pause_label',
    'enable_bootstrap',
    'bootstrap_issues_label',
    'draft_pr',
    'dry_run',
    'options_json',
    'dispatcher_force_issue',
    'worker_max_parallel',
    'conveyor_max_merges'
  ];

  for (const key of orderedKeys) {
    if (Object.prototype.hasOwnProperty.call(outputs, key)) {
      core.setOutput(key, outputs[key]);
    }
  }

  const summary = core.summary;
  summary.addHeading('Agents orchestrator parameters');
  summary.addTable([
    [{ data: 'Parameter', header: true }, { data: 'Value', header: true }],
    ...orderedKeys.map((key) => [key, summarise(outputs[key])])
  ]);
  if (keepalivePaused) {
    summary.addRaw(`keepalive skipped because the ${KEEPALIVE_PAUSE_LABEL} label is present.`).addEOL();
  } else if (keepaliveRequested !== keepaliveEffective) {
    summary
      .addRaw('keepalive disabled via configuration overrides (input or params).')
      .addEOL();
  }
  await summary.write();

  return { outputs };
}

module.exports = {
  resolveOrchestratorParams,
  __internals: {
    toString,
    toBoolString,
    toCsv,
    nested,
    toBoundedIntegerString,
    sanitiseOptions,
    summarise,
    KEEPALIVE_PAUSE_LABEL,
    DEFAULTS
  }
};
