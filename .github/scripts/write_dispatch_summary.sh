#!/usr/bin/env bash

set -euo pipefail

path_label=${PATH_LABEL:-unknown}
dispatch_value=${DISPATCH:-}
trace_value=${TRACE:-}
pr_value_raw=${PR_NUMBER:-}
fallback_pr=${FALLBACK_PR:-}
comment_id_raw=${COMMENT_ID:-}
comment_fallback=${COMMENT_FALLBACK:-}

if [[ -z "${trace_value}" ]]; then
  trace_value='-'
fi

if [[ -z "${pr_value_raw}" || "${pr_value_raw}" == '0' || "${pr_value_raw}" == 'unknown' ]]; then
  pr_value_raw="${fallback_pr:-}"
fi

if [[ -z "${comment_id_raw}" || "${comment_id_raw}" == '0' || "${comment_id_raw}" == 'unknown' ]]; then
  comment_id_raw="${comment_fallback:-}"
fi

format_pr() {
  local raw="${1:-}"
  if [[ -n "${raw}" && "${raw}" != 'unknown' && "${raw}" != '0' ]]; then
    printf '#%s' "${raw}"
  else
    printf '#?'
  fi
}

pr_value=$(format_pr "${pr_value_raw}")

comment_value=${comment_id_raw:-}
if [[ -z "${comment_value}" ]]; then
  comment_value='<none>'
fi

dispatch_normalised=$(printf '%s' "${dispatch_value}" | tr '[:upper:]' '[:lower:]')
if [[ "${dispatch_normalised}" == 'true' ]]; then
  ok_value='true'
else
  ok_value='false'
fi

summary_line="DISPATCH: ok=${ok_value} path=${path_label} pr=${pr_value} comment_id=${comment_value} trace=${trace_value}"

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  printf '%s\n' "${summary_line}" >>"${GITHUB_STEP_SUMMARY}"
fi

printf '%s\n' "${summary_line}"