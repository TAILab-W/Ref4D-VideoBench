#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source scripts/_logging.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-batch}"
TOPICS="${TOPICS:-}"
MODELS="${MODELS:-}"
SAMPLE_ID="${SAMPLE_ID:-}"
MODEL="${MODEL:-}"
TOPIC="${TOPIC:-}"
STEPS="${STEPS:-detect,vlm,embed,merge,match,metrics}"
CFG_DEFAULT="${CFG_DEFAULT:-ref4d_eval/event/configs/default.yaml}"
CFG_VLM="${CFG_VLM:-ref4d_eval/event/configs/model_vlm.yaml}"
CFG_EMBED="${CFG_EMBED:-ref4d_eval/event/configs/model_embed.yaml}"
CFG_SHOT="${CFG_SHOT:-ref4d_eval/event/configs/model_shot.yaml}"
CFG_GEBD="${CFG_GEBD:-ref4d_eval/event/configs/model_gebd.yaml}"
META_PATH="${META_PATH:-data/metadata/ref4d_meta.jsonl}"
REF_VIDEO_ROOT="${REF_VIDEO_ROOT:-data/refvideo}"
GEN_VIDEO_ROOT="${GEN_VIDEO_ROOT:-data/genvideo}"
REF_EVENT_ROOT="${REF_EVENT_ROOT:-data/metadata/event_evidence}"
CACHE_ROOT="${CACHE_ROOT:-outputs/event/cache}"
SCORES_ROOT="${SCORES_ROOT:-outputs/event/scores}"

common=(
  --steps "$STEPS"
  --cfg-default "$CFG_DEFAULT"
  --cfg-vlm "$CFG_VLM"
  --cfg-embed "$CFG_EMBED"
  --cfg-shot "$CFG_SHOT"
  --cfg-gebd "$CFG_GEBD"
  --meta-path "$META_PATH"
  --ref-video-root "$REF_VIDEO_ROOT"
  --gen-video-root "$GEN_VIDEO_ROOT"
  --ref-event-root "$REF_EVENT_ROOT"
  --cache-root "$CACHE_ROOT"
  --scores-root "$SCORES_ROOT"
)
if [[ "${FORCE:-0}" == "1" ]]; then
  common+=(--force)
fi

event_base="${SCORES_ROOT%/}"
event_base="${event_base%/scores}"
log_file="${EVENT_LOG_FILE:-${event_base}/logs/event_eval_$(ref4d_timestamp).log}"

if [[ "$MODE" == "run" ]]; then
  if [[ -z "$SAMPLE_ID" || -z "$MODEL" ]]; then
    echo "[run_event_eval] MODE=run requires SAMPLE_ID and MODEL; TOPIC is optional." >&2
    exit 2
  fi
  ref4d_run_logged event "$log_file" "$PYTHON_BIN" -m ref4d_eval.event.src.cli.main run \
    --topic "$TOPIC" \
    --sample-id "$SAMPLE_ID" \
    --model "$MODEL" \
    "${common[@]}" "$@"
else
  batch_args=(batch "${common[@]}")
  if [[ -n "$TOPICS" ]]; then
    batch_args+=(--topics "$TOPICS")
  fi
  if [[ -n "$MODELS" ]]; then
    batch_args+=(--models "$MODELS")
  elif ! find "$GEN_VIDEO_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -print -quit 2>/dev/null | grep -q .; then
    echo "[run_event_eval] No MODELS provided and no model directories found under $GEN_VIDEO_ROOT/<model>/." >&2
    echo "Place generated videos under $GEN_VIDEO_ROOT/<model>/<sample_id>.mp4 or set MODELS=modelA,modelB." >&2
    exit 2
  fi
  ref4d_run_logged event "$log_file" "$PYTHON_BIN" -m ref4d_eval.event.src.cli.main "${batch_args[@]}" "$@"
fi

echo "[event] summary: ${SCORES_ROOT%/}/event_scores_summary.csv"
