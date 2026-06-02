#!/usr/bin/env bash
set -euo pipefail

# One-click prompt build pipeline:
# Ref videos (data/refvideo) + evidence (data/metadata/semantic_event_evidence)
# -> merged prompts (data/metadata/ref4d_prompts.jsonl)
#
# You can override defaults via env vars:
#   PYTHON_BIN, VIDEO_BASE_DIR, EVIDENCE_BASE_DIR, OUTPUT_JSONL, MODEL_PATH,
#   METADATA_JSONL, SOURCE_INDEX, PROGRESS_FILE
#
# Extra CLI arguments are forwarded to:
#   ref4d_build/prompt/batch_video_prompt_generator.py
# Example:
#   bash scripts/build_prompts_from_refvideo.sh --theme animals_and_ecology architecture --dry-run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
VIDEO_BASE_DIR="${VIDEO_BASE_DIR:-${REPO_ROOT}/data/refvideo}"
EVIDENCE_BASE_DIR="${EVIDENCE_BASE_DIR:-${REPO_ROOT}/data/metadata/semantic_event_evidence}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${REPO_ROOT}/data/metadata/ref4d_prompts.jsonl}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/minicpm-v-4_5}"
PROGRESS_FILE="${PROGRESS_FILE:-${REPO_ROOT}/data/metadata/prompt_progress.json}"
METADATA_JSONL="${METADATA_JSONL:-${REPO_ROOT}/data/metadata/ref4d_prompts.jsonl}"
SOURCE_INDEX="${SOURCE_INDEX:-${REPO_ROOT}/data/metadata/ref4d_videobench_reference_sources.csv}"

BATCH_SCRIPT="${REPO_ROOT}/ref4d_build/prompt/batch_video_prompt_generator.py"

if [[ ! -f "${BATCH_SCRIPT}" ]]; then
  echo "[ERROR] Batch script not found: ${BATCH_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${VIDEO_BASE_DIR}" ]]; then
  echo "[ERROR] Video directory not found: ${VIDEO_BASE_DIR}" >&2
  exit 1
fi

if [[ ! -d "${EVIDENCE_BASE_DIR}" ]]; then
  echo "[ERROR] Evidence directory not found: ${EVIDENCE_BASE_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_JSONL}")"
mkdir -p "$(dirname "${PROGRESS_FILE}")"

echo "============================================================"
echo "Build Prompts From RefVideo"
echo "============================================================"
echo "[INFO] Python:       ${PYTHON_BIN}"
echo "[INFO] Video dir:    ${VIDEO_BASE_DIR}"
echo "[INFO] Evidence dir: ${EVIDENCE_BASE_DIR}"
echo "[INFO] Output JSONL: ${OUTPUT_JSONL}"
echo "[INFO] Model path:   ${MODEL_PATH}"
echo "[INFO] Metadata:     ${METADATA_JSONL}"
echo "[INFO] Source index: ${SOURCE_INDEX}"
echo "[INFO] Progress:     ${PROGRESS_FILE}"
echo "============================================================"

"${PYTHON_BIN}" "${BATCH_SCRIPT}" \
  --video-base-dir "${VIDEO_BASE_DIR}" \
  --json-base-dir "${EVIDENCE_BASE_DIR}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --model-path "${MODEL_PATH}" \
  --metadata-jsonl "${METADATA_JSONL}" \
  --source-index "${SOURCE_INDEX}" \
  --progress-file "${PROGRESS_FILE}" \
  --resume \
  --skip-confirmation \
  "$@"
