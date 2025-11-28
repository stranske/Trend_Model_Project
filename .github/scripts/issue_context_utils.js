const {
  extractScopeTasksAcceptanceSections,
  analyzeSectionPresence,
} = require('./issue_scope_parser.js');

const EXPECTED_SECTIONS = ['Scope', 'Tasks', 'Acceptance Criteria'];

function buildIssueContext(issueBody) {
  const body = issueBody || '';
  const scopeBlockWithPlaceholders = extractScopeTasksAcceptanceSections(body, {
    includePlaceholders: true,
  });
  const scopeBlockStrict = extractScopeTasksAcceptanceSections(body, {
    includePlaceholders: false,
  });
  const presence = analyzeSectionPresence(body);
  const missingSections = Array.isArray(presence?.missing) ? presence.missing : [];
  const summaryContentBlock = (scopeBlockStrict || '').trim();
  const summaryNeedsWarning = !summaryContentBlock || missingSections.length > 0;
  const missingDescription = missingSections.length
    ? `Problem detected: ${missingSections.join(', ')} ${missingSections.length === 1 ? 'is' : 'are'} missing or empty in the source issue.`
    : 'Problem detected: The parser could not find any of the canonical headings in the source issue.';
  const warningDetails = summaryNeedsWarning
    ? [
        'Automated Status Summary expects the following sections in the source issue:',
        ...EXPECTED_SECTIONS.map((section) => `- ${section}`),
        '',
        missingDescription,
        '',
        'Please edit the issue to add `## Scope`, `## Tasks`, and `## Acceptance Criteria`, then rerun the agent workflow so keepalive can parse your plan.',
      ]
    : [];
  const warningLines = summaryNeedsWarning ? ['#### ⚠️ Template Warning', '', ...warningDetails] : [];
  const summaryLines = ['<!-- auto-status-summary:start -->', '## Automated Status Summary'];
  if (!summaryNeedsWarning && summaryContentBlock) {
    summaryLines.push(summaryContentBlock);
  } else {
    summaryLines.push('#### ⚠️ Summary Unavailable', '', ...warningDetails);
  }
  summaryLines.push('<!-- auto-status-summary:end -->');

  return {
    scopeBlock: (scopeBlockWithPlaceholders || '').trim(),
    statusSummaryBlock: summaryLines.join('\n'),
    warningLines,
    warningDetails,
    summaryNeedsWarning,
    missingSections,
    summaryContentBlock,
  };
}

module.exports = {
  EXPECTED_SECTIONS,
  buildIssueContext,
};
