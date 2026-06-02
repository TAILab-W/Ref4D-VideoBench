#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

DIMS="${DIMS:-semantic,event,motion,world}"
MODELS_ARG="${MODELS:-}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
USE_CONDA_ENVS="${USE_CONDA_ENVS:-0}"
SEMANTIC_ENV="${SEMANTIC_ENV:-ref4d_semantic_world}"
EVENT_ENV="${EVENT_ENV:-ref4d_event}"
MOTION_ENV="${MOTION_ENV:-ref4d_motion}"
SEMANTIC_GPUS="${SEMANTIC_GPUS:-${GPUS:-0}}"
MOTION_WORKERS="${MOTION_WORKERS:-${WORKERS:-3}}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_4d_eval.sh [options]

Options:
  --dims semantic,event,motion,world   Dimensions to run (default: all four).
  --models modelA,modelB               Generated-video model directories to evaluate.
  OUTPUT_SUFFIX=name                   Write summaries under outputs/<name>_{semantic,event,motion,world}/.
  SEMANTIC_GPUS=0                      GPU list for semantic evidence extraction. Use auto for multi-GPU.
  MOTION_WORKERS=6                     Worker count for motion only.
  SEMANTIC_STEPS=score                 Override semantic STEPS without affecting event.
  EVENT_STEPS=match,metrics            Override event STEPS without affecting semantic.
  USE_CONDA_ENVS=1                     Run each dimension in its recommended conda env.
  -h, --help                           Show this help.

Environment overrides are forwarded to the underlying single-dimension scripts.
Conda env names can be overridden with SEMANTIC_ENV, EVENT_ENV, and MOTION_ENV.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dims)
      DIMS="$2"
      shift 2
      ;;
    --models)
      MODELS_ARG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[run_4d_eval] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

has_dim() {
  local dims=",${DIMS,,},"
  local dim="$1"
  [[ "$dims" == *",${dim},"* ]]
}

run_dim_script() {
  local env_name="$1"
  local steps_value="$2"
  local script_path="$3"
  shift 3
  local env_args=("$@")
  if [[ "$USE_CONDA_ENVS" == "1" ]]; then
    cmd=(conda run --no-capture-output -n "$env_name")
    if [[ -n "$steps_value" || ${#env_args[@]} -gt 0 ]]; then
      cmd+=(env)
      if [[ -n "$steps_value" ]]; then
        cmd+=("STEPS=$steps_value")
      fi
      cmd+=("${env_args[@]}")
    fi
    cmd+=(bash "$script_path")
    "${cmd[@]}"
  else
    if [[ -n "$steps_value" || ${#env_args[@]} -gt 0 ]]; then
      if [[ -n "$steps_value" ]]; then
        env_args+=("STEPS=$steps_value")
      fi
      env "${env_args[@]}" bash "$script_path"
    else
      bash "$script_path"
    fi
  fi
}

if [[ -n "$MODELS_ARG" ]]; then
  export INCLUDE_MODELS="${INCLUDE_MODELS:-$MODELS_ARG}"
  export MODELS="${MODELS:-$MODELS_ARG}"
fi

if [[ -n "$OUTPUT_SUFFIX" ]]; then
  export GEN_OUT_ROOT="${GEN_OUT_ROOT:-outputs/${OUTPUT_SUFFIX}_semantic/cache/evidence_gen}"
  export SCORES_OUT_DIR="${SCORES_OUT_DIR:-outputs/${OUTPUT_SUFFIX}_semantic/scores}"
  export CACHE_ROOT="${CACHE_ROOT:-outputs/${OUTPUT_SUFFIX}_event/cache}"
  export SCORES_ROOT="${SCORES_ROOT:-outputs/${OUTPUT_SUFFIX}_event/scores}"
  export OUT="${OUT:-outputs/${OUTPUT_SUFFIX}_motion/scores/motion_scores_summary.csv}"
  export OUT_DIR="${OUT_DIR:-outputs/${OUTPUT_SUFFIX}_world}"
fi

if [[ -n "${REF_VIDEO_ROOT:-}" && -z "${REF_VIDEO_DIR:-}" ]]; then
  export REF_VIDEO_DIR="$REF_VIDEO_ROOT"
fi

if [[ -n "${META_PATH:-}" && -z "${REF_OUT_DIR:-}" ]]; then
  meta_dir="$(dirname "$META_PATH")"
  if [[ -d "${meta_dir}/semantic_evidence" ]]; then
    export REF_OUT_DIR="${meta_dir}/semantic_evidence"
  fi
fi

if [[ -n "${META_PATH:-}" && -z "${CFG:-}" ]]; then
  meta_dir="$(dirname "$META_PATH")"
  custom_cfg="outputs/motion/custom_motion_ref4d.yaml"
  if [[ -f "$custom_cfg" && -d "${meta_dir}/motion_ref" ]]; then
    export CFG="$custom_cfg"
  fi
fi

ran_dims=()

if has_dim semantic; then
  echo "[run_4d_eval] semantic"
  run_dim_script "$SEMANTIC_ENV" "${SEMANTIC_STEPS:-}" scripts/run_semantic_eval.sh "GPUS=$SEMANTIC_GPUS"
  ran_dims+=("semantic")
fi

if has_dim event; then
  echo "[run_4d_eval] event"
  run_dim_script "$EVENT_ENV" "${EVENT_STEPS:-}" scripts/run_event_eval.sh
  ran_dims+=("event")
fi

if has_dim motion; then
  echo "[run_4d_eval] motion"
  run_dim_script "$MOTION_ENV" "" scripts/run_motion_eval.sh "WORKERS=$MOTION_WORKERS"
  ran_dims+=("motion")
fi

if has_dim world; then
  echo "[run_4d_eval] world"
  run_dim_script "$SEMANTIC_ENV" "" scripts/run_world_eval.sh
  ran_dims+=("world")
fi

if (( ${#ran_dims[@]} == 0 )); then
  echo "[run_4d_eval] No supported dimensions selected. Use --dims semantic,event,motion,world." >&2
  exit 2
fi
