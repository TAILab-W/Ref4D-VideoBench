#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

for arg in "$@"; do
  if [[ "$arg" == *=* ]]; then
    export "$arg"
  fi
done

RUN_SEMANTIC="${RUN_SEMANTIC:-1}"
RUN_MOTION="${RUN_MOTION:-1}"
RUN_EVENT="${RUN_EVENT:-1}"
RUN_WORLD="${RUN_WORLD:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"

SEMANTIC_STEPS="${SEMANTIC_STEPS:-${STEPS:-both}}"
EVENT_STEPS="${EVENT_STEPS:-${STEPS:-detect,vlm,embed,merge,match,metrics}}"

missing=()

has_step() {
  local steps=",$1,"
  local step="$2"
  [[ "${steps,,}" == *",${step,,},"* ]]
}

needs_any_step() {
  local steps="$1"
  shift
  local step
  for step in "$@"; do
    if has_step "$steps" "$step"; then
      return 0
    fi
  done
  return 1
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    missing+=("$label: $path")
  fi
}

require_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    missing+=("$label: $path")
  fi
}

require_nonempty_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "$path" ]] || ! find "$path" -type f ! -name '.gitkeep' -print -quit 2>/dev/null | grep -q .; then
    missing+=("$label: $path")
  fi
}

preflight_semantic() {
  if needs_any_step "$SEMANTIC_STEPS" extract both; then
    require_dir "semantic generated video root" "${GEN_VIDEO_ROOT:-data/genvideo}"
    require_nonempty_dir "semantic MiniCPM checkpoint" "${MODEL_LOCAL_PATH:-checkpoints/minicpm-v-4_5}"
  fi
  if needs_any_step "$SEMANTIC_STEPS" score both; then
    require_nonempty_dir "semantic reference evidence cache" "${REF_OUT_DIR:-data/metadata/semantic_evidence}"
    if ! needs_any_step "$SEMANTIC_STEPS" extract both; then
      require_dir "semantic generated evidence root" "${GEN_OUT_ROOT:-outputs/semantic/cache/evidence_gen}"
    fi
  fi
}

preflight_motion() {
  require_dir "motion generated video root" "${GEN_VIDEO_ROOT:-data/genvideo}"
  require_file "motion GroundingDINO config" "${GDINO_CFG:-third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
  require_file "motion GroundingDINO checkpoint" "${GDINO_CKPT:-checkpoints/groundingdino/groundingdino_swint_ogc.pth}"
  require_dir "motion SAM2 third_party tree" "${SAM2_REPO_DIR:-third_party/sam2}"
  require_file "motion SAM2 checkpoint" "${SAM2_CKPT:-checkpoints/sam2/sam2.1_hiera_large.pt}"
  require_file "motion TAPIR torch implementation" "${TAPIR_TORCH_PY:-third_party/tapir/tapnet/torch/tapir_model.py}"
  require_file "motion TAPIR checkpoint" "${TAPIR_CKPT:-checkpoints/tapnet_checkpoints/bootstapir_checkpoint_v2.pt}"
}

preflight_event() {
  if [[ -z "${MODELS:-}" ]] && ! find "${GEN_VIDEO_ROOT:-data/genvideo}" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -print -quit 2>/dev/null | grep -q .; then
    missing+=("event models: set MODELS or place videos under ${GEN_VIDEO_ROOT:-data/genvideo}/<model>/")
  fi
  if needs_any_step "$EVENT_STEPS" detect; then
    require_file "event DDM checkpoint" "${DDM_CKPT:-checkpoints/ddmnet/checkpoint.pth.tar}"
    require_file "event TransNetV2 weights" "${TRANSNETV2_WEIGHTS:-checkpoints/transnetv2/transnetv2-pytorch-weights.pth}"
  fi
  if needs_any_step "$EVENT_STEPS" vlm; then
    require_nonempty_dir "event VideoLLaMA3 checkpoint" "${VIDEOLLAMA3_DIR:-checkpoints/videollama3-7b}"
  fi
  if needs_any_step "$EVENT_STEPS" embed; then
    require_nonempty_dir "event E5 checkpoint" "${E5_DIR:-checkpoints/e5-large-v2}"
  fi
}

preflight_world() {
  if [[ -z "${MODELS:-}" ]] && ! find "${GEN_VIDEO_ROOT:-data/genvideo}" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -print -quit 2>/dev/null | grep -q .; then
    missing+=("world models: set MODELS or place videos under ${GEN_VIDEO_ROOT:-data/genvideo}/<model>/")
  fi
  require_nonempty_dir "world question bank" "${BANK_DIR:-data/metadata/world_qa}"
  require_nonempty_dir "world MiniCPM checkpoint" "${LOCAL_PATH:-checkpoints/minicpm-v-4_5}"
}

if [[ "$RUN_SEMANTIC" == "1" ]]; then
  preflight_semantic
fi
if [[ "$RUN_MOTION" == "1" ]]; then
  preflight_motion
fi
if [[ "$RUN_EVENT" == "1" ]]; then
  preflight_event
fi
if [[ "$RUN_WORLD" == "1" ]]; then
  preflight_world
fi

if (( ${#missing[@]} > 0 )); then
  echo "[run_all] Preflight failed; missing required resources:" >&2
  printf '  [MISSING] %s\n' "${missing[@]}" >&2
  echo "[run_all] Suggested preparation scripts:" >&2
  echo "  bash scripts/download_semantic_world_models.sh" >&2
  echo "  bash scripts/download_event_models.sh" >&2
  echo "  bash scripts/download_motion_models.sh" >&2
  exit 2
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "[run_all] Preflight OK."
  exit 0
fi

if [[ "$RUN_SEMANTIC" == "1" ]]; then
  echo "[run_all] semantic"
  STEPS="$SEMANTIC_STEPS" bash scripts/run_semantic_eval.sh
fi

if [[ "$RUN_MOTION" == "1" ]]; then
  echo "[run_all] motion"
  bash scripts/run_motion_eval.sh
fi

if [[ "$RUN_EVENT" == "1" ]]; then
  echo "[run_all] event"
  STEPS="$EVENT_STEPS" bash scripts/run_event_eval.sh
fi

if [[ "$RUN_WORLD" == "1" ]]; then
  echo "[run_all] world"
  bash scripts/run_world_eval.sh
fi
