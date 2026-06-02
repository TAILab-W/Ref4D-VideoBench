#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source scripts/_logging.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
EVI_EXTRACT_PY="${EVI_EXTRACT_PY:-ref4d_eval/semantic/evidence_extract/evi_extract.py}"
BATCH_SCORING_PY="${BATCH_SCORING_PY:-ref4d_eval/semantic/softalign/batch_scoring.py}"
SOFTALIGN_YAML="${SOFTALIGN_YAML:-ref4d_eval/semantic/softalign/softalign.yaml}"
REF_VIDEO_DIR="${REF_VIDEO_DIR:-data/refvideo}"
GEN_VIDEO_ROOT="${GEN_VIDEO_ROOT:-data/genvideo}"
REF_OUT_DIR="${REF_OUT_DIR:-data/metadata/semantic_evidence}"
GEN_OUT_ROOT="${GEN_OUT_ROOT:-outputs/semantic/cache/evidence_gen}"
SCORES_OUT_DIR="${SCORES_OUT_DIR:-outputs/semantic/scores}"
MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH:-checkpoints/minicpm-v-4_5}"
GPUS="${GPUS:-auto}"
STEPS="${STEPS:-both}"
MODEL_FILTER="${INCLUDE_MODELS:-${MODELS:-}}"

args=(
  --evi-extract-py "$EVI_EXTRACT_PY"
  --batch-scoring-py "$BATCH_SCORING_PY"
  --softalign-yaml "$SOFTALIGN_YAML"
  --gen-video-root "$GEN_VIDEO_ROOT"
  --ref-out-dir "$REF_OUT_DIR"
  --gen-out-root "$GEN_OUT_ROOT"
  --model-local-path "$MODEL_LOCAL_PATH"
  --gpus "$GPUS"
  --steps "$STEPS"
  --scores-out-dir "$SCORES_OUT_DIR"
)

if [[ -d "$REF_VIDEO_DIR" ]]; then
  args+=(--ref-video-dir "$REF_VIDEO_DIR")
fi
if [[ -n "$MODEL_FILTER" ]]; then
  args+=(--include-models "$MODEL_FILTER")
fi
if [[ -n "${EXCLUDE_MODELS:-}" ]]; then
  args+=(--exclude-models "$EXCLUDE_MODELS")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  args+=(--force)
fi
if [[ "${VERIFY:-0}" == "1" ]]; then
  args+=(--verify)
fi
if [[ "${LIVE:-0}" == "1" ]]; then
  args+=(--live)
fi
if [[ "${SERIALIZE_OUTPUT:-0}" == "1" ]]; then
  args+=(--serialize-output)
fi
if [[ "${QUIET:-0}" == "1" ]]; then
  args+=(--quiet)
fi

semantic_base="${SCORES_OUT_DIR%/}"
semantic_base="${semantic_base%/scores}"
log_file="${SEMANTIC_LOG_FILE:-${semantic_base}/logs/semantic_eval_$(ref4d_timestamp).log}"

if [[ "${LIVE:-0}" == "1" && -z "${REF4D_VERBOSE:-}" ]]; then
  export REF4D_VERBOSE=1
fi

ref4d_run_logged semantic "$log_file" "$PYTHON_BIN" -m ref4d_eval.semantic.semantics_evi_score_dist "${args[@]}" "$@"
echo "[semantic] summary: ${SCORES_OUT_DIR%/}/semantic_scores_summary.csv"
