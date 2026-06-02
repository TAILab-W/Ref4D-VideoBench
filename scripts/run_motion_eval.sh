#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source scripts/_logging.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="${CFG:-ref4d_eval/motion/configs/motion_ref4d.yaml}"
OUT="${OUT:-outputs/motion/scores/motion_scores_summary.csv}"
WORKERS="${WORKERS:-3}"
MODELS="${MODELS:-}"
GEN_VIDEO_ROOT="${GEN_VIDEO_ROOT:-data/genvideo}"

mkdir -p "$(dirname "$OUT")"

args=(
  --cfg "$CFG"
  --base "$repo_root"
  --out "$OUT"
  --workers "$WORKERS"
  --gen-video-root "$GEN_VIDEO_ROOT"
)

if [[ "${FORCE:-0}" == "1" ]]; then
  args+=(--force)
fi
if [[ -n "$MODELS" ]]; then
  args+=(--models "$MODELS")
fi

motion_base="$(dirname "$(dirname "$OUT")")"
log_file="${MOTION_LOG_FILE:-${motion_base}/logs/motion_eval_$(ref4d_timestamp).log}"

ref4d_run_logged motion "$log_file" "$PYTHON_BIN" -m ref4d_eval.motion.run_batch_motion "${args[@]}" "$@"
echo "[motion] summary: $OUT"
