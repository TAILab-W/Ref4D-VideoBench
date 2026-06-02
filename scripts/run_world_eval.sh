#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source scripts/_logging.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0}"
MODELS="${MODELS:-}"
BANK_DIR="${BANK_DIR:-data/metadata/world_qa}"
GEN_VIDEO_ROOT="${GEN_VIDEO_ROOT:-data/genvideo}"
OUT_DIR="${OUT_DIR:-${WORLD_OUT_DIR:-outputs/world}}"
LOCAL_PATH="${LOCAL_PATH:-checkpoints/minicpm-v-4_5}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
FPS="${FPS:-3}"
CAP_FRAMES="${CAP_FRAMES:-300}"
RESIZE_SHORT="${RESIZE_SHORT:-448}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
VERBOSE="${VERBOSE:-0}"

if [[ -z "$MODELS" ]]; then
  if [[ -d "$GEN_VIDEO_ROOT" ]]; then
    discovered_models=()
    while IFS= read -r -d '' model_dir; do
      discovered_models+=("$(basename "$model_dir")")
    done < <(find "$GEN_VIDEO_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -print0 | sort -z)
    if (( ${#discovered_models[@]} > 0 )); then
      MODELS="$(IFS=,; echo "${discovered_models[*]}")"
    fi
  fi
fi

if [[ -z "$MODELS" ]]; then
  echo "[world] MODELS is empty and no model directories were found under $GEN_VIDEO_ROOT" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
runner_args=()
if [[ "$VERBOSE" == "1" ]]; then
  runner_args+=(--verbose)
fi

for model in "${MODEL_LIST[@]}"; do
  model="$(echo "$model" | xargs)"
  [[ -z "$model" ]] && continue
  video_dir="$GEN_VIDEO_ROOT/$model"
  if [[ ! -d "$video_dir" ]]; then
    echo "[world] Skip missing video dir: $video_dir" >&2
    continue
  fi
  log_file="${WORLD_LOG_FILE:-${OUT_DIR%/}/logs/world_eval_${model}_$(ref4d_timestamp).log}"
  echo "[world] Evaluating model=$model ..."
  ref4d_run_logged world "$log_file" "$PYTHON_BIN" -m ref4d_eval.world.runner \
    --bank-dir "$BANK_DIR" \
    --video-dir "$video_dir" \
    --out-dir "$OUT_DIR" \
    --local-path "$LOCAL_PATH" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --fps "$FPS" \
    --cap-frames "$CAP_FRAMES" \
    --resize-short "$RESIZE_SHORT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    "${runner_args[@]}"
  echo "[world] Done model=$model"
done

echo "[world] All models finished. Summary: $OUT_DIR/scores/world_scores_summary.csv"
