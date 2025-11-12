#!/usr/bin/env bash

set -euo pipefail

path_label=${PATH_LABEL:-unknown}
dispatch=${DISPATCH:-}
reason=${REASON:-}
trace_value=${TRACE:-}
pr_value_raw=${PR_NUMBER:-}
fallback_pr=${FALLBACK_PR:-}
activation_id=${ACTIVATION_ID:-}
activation_fallback=${ACTIVATION_FALLBACK:-}
concurrency_success=${CONCURRENCY_SUCCESS:-granted}
concurrency_held=${CONCURRENCY_HELD:-held}

if [[ -z "${trace_value}" ]]; then
  trace_value='-'
fi

if [[ -z "${pr_value_raw}" || "${pr_value_raw}" == '0' || "${pr_value_raw}" == 'unknown' ]]; then
  pr_value_raw="${fallback_pr:-}"
fi

activation_value="${activation_id:-}"
if [[ -z "${activation_value}" ]]; then
  activation_value="${activation_fallback:-}"
fi
if [[ -z "${activation_value}" ]]; then
  activation_value='<none>'
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

summary_reason='no-activation'
if [[ -n "${reason}" ]]; then
  summary_reason="${reason}"
fi

case "${summary_reason}" in
  duplicate-keepalive)
    summary_reason='lock-held'
    ;;
  missing-pr-number)
    summary_reason='no-linked-pr'
    ;;
esac

if [[ "${dispatch}" == 'true' ]]; then
  concurrency_label="${concurrency_success}"
  summary_line="DISPATCH: ok=true path=${path_label} pr=${pr_value} activation=${activation_value} concurrency=${concurrency_label} trace=${trace_value}"
else
  case "${summary_reason}" in
    lock-held)
      concurrency_label="${concurrency_held}"
      summary_line="DISPATCH: ok=false reason=lock-held path=${path_label} pr=${pr_value} activation=${activation_value} concurrency=${concurrency_label} trace=${trace_value}"
      ;;
    no-linked-pr)
      summary_line="DISPATCH: ok=false reason=no-linked-pr path=${path_label} activation=<none> trace=${trace_value}"
      ;;
    *)
      summary_line="DISPATCH: ok=false reason=${summary_reason} path=${path_label} pr=${pr_value} activation=<none> trace=${trace_value}"
      ;;
  esac
fi

if [[ -z "${summary_line:-}" ]]; then
  summary_line="DISPATCH: ok=false reason=summary-error path=${path_label} trace=${trace_value}"
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  printf '%s\n' "${summary_line}" >>"${GITHUB_STEP_SUMMARY}"
fi

printf '%s\n' "${summary_line}"