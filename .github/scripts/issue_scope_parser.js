'use strict';

const normalizeNewlines = (value) => String(value || '').replace(/\r\n/g, '\n');
const escapeRegExp = (value) => String(value ?? '').replace(/[\\^$.*+?()[\]{}|]/g, '\\$&');

const SECTION_DEFS = [
  { key: 'scope', label: 'Scope', aliases: ['Scope', 'Issue Scope'] },
  { key: 'tasks', label: 'Tasks', aliases: ['Tasks', 'Task List'] },
  {
    key: 'acceptance',
    label: 'Acceptance Criteria',
    aliases: ['Acceptance Criteria', 'Acceptance', 'Acceptance criteria'],
  },
];

function collectSections(source) {
  const normalized = normalizeNewlines(source);
  if (!normalized.trim()) {
    return { segment: '', sections: {} };
  }

  const startMarker = '<!-- auto-status-summary:start -->';
  const endMarker = '<!-- auto-status-summary:end -->';
  const startIndex = normalized.indexOf(startMarker);
  const endIndex = normalized.indexOf(endMarker);

  let segment = normalized;
  if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
    segment = normalized.slice(startIndex + startMarker.length, endIndex);
  }

  const headingLabelPattern = SECTION_DEFS
    .flatMap((section) => section.aliases)
    .map((title) => escapeRegExp(title))
    .join('|');

  // Match headings that may be markdown headers (# H), bold (**H**), or plain text (with optional colon).
  const headingRegex = new RegExp(
    `^\\s*(?:#{1,6}\\s+|\\*\\*)?(${headingLabelPattern})(?:\\*\\*|:)?\\s*$`,
    'gim'
  );

  const aliasLookup = SECTION_DEFS.reduce((acc, section) => {
    section.aliases.forEach((alias) => {
      acc[alias.toLowerCase()] = section;
    });
    return acc;
  }, {});

  const headings = [];
  let match;
  while ((match = headingRegex.exec(segment)) !== null) {
    const title = (match[1] || '').toLowerCase();
    if (!title || !aliasLookup[title]) {
      continue;
    }
    const section = aliasLookup[title];
    headings.push({
      title: section.key,
      label: section.label,
      index: match.index,
      length: match[0].length,
    });
  }

  const extracted = SECTION_DEFS.reduce((acc, section) => {
    acc[section.key] = '';
    return acc;
  }, {});

  if (headings.length === 0) {
    return { segment, sections: extracted };
  }

  for (const section of SECTION_DEFS) {
    const canonicalTitle = section.label;
    const header = headings.find((entry) => entry.title === section.key);
    if (!header) {
      continue; // Skip missing sections instead of failing
    }
    const nextHeader = headings
      .filter((entry) => entry.index > header.index)
      .sort((a, b) => a.index - b.index)[0];
    const contentStart = (() => {
      const start = header.index + header.length;
      if (segment[start] === '\n') {
        return start + 1;
      }
      return start;
    })();
    const contentEnd = nextHeader ? nextHeader.index : segment.length;
    const content = normalizeNewlines(segment.slice(contentStart, contentEnd)).trim();
    extracted[section.key] = content;
  }

  return { segment, sections: extracted };
}

/**
 * Extracts Scope, Tasks/Task List, and Acceptance Criteria sections from issue text.
 *
 * The parser is intentionally tolerant:
 * - Accepts headings written as markdown headers (# Title), bold (**Title**), or plain text
 *   with or without a trailing colon (e.g., "Tasks:").
 * - Searches within auto-status-summary markers when present, falling back to the full body.
 *
 * @param {string} source - The issue body text to parse.
 * @returns {string} Formatted sections with #### headings, or an empty string if none were found.
 */
const extractScopeTasksAcceptanceSections = (source) => {
  const { sections } = collectSections(source);

  const extracted = [];
  for (const section of SECTION_DEFS) {
    const canonicalTitle = section.label;
    const content = sections[section.key];
    if (!content && !extracted.length) {
      // If the first section is empty, continue so the caller can fall back cleanly.
      continue;
    }
    const headerLine = `#### ${canonicalTitle}`;
    extracted.push(content ? `${headerLine}\n${content}` : headerLine);
  }

  return extracted.join('\n\n').trim();
};

const parseScopeTasksAcceptanceSections = (source) => {
  const { sections } = collectSections(source);
  return sections;
};

module.exports = {
  extractScopeTasksAcceptanceSections,
  parseScopeTasksAcceptanceSections,
};
