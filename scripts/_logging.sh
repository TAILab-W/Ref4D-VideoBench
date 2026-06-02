#!/usr/bin/env bash

ref4d_timestamp() {
  date +%Y%m%d-%H%M%S
}

ref4d_run_logged() {
  local label="$1"
  local log_file="$2"
  shift 2

  mkdir -p "$(dirname "$log_file")"

  if [[ "${REF4D_VERBOSE:-0}" == "1" ]]; then
    echo "[$label] streaming output to terminal"
    "$@"
    return $?
  fi

  echo "[$label] log: $log_file"
  if "$@" >"$log_file" 2>&1; then
    echo "[$label] done"
  else
    local rc=$?
    echo "[$label] failed with exit code $rc. Log tail:" >&2
    tail -n "${REF4D_LOG_TAIL_LINES:-80}" "$log_file" >&2 || true
    return "$rc"
  fi
}
