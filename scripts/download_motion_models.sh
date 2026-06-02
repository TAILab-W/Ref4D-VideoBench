#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${USE_NETWORK_TURBO:-0}" == "1" && -f /etc/network_turbo ]]; then
  # Optional acceleration for Hugging Face/GitHub access on AutoDL-like hosts.
  # Users can also source this file manually before running the script.
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
AUTO_CLONE="${AUTO_CLONE:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

GROUNDINGDINO_REPO_URL="${GROUNDINGDINO_REPO_URL:-https://github.com/IDEA-Research/GroundingDINO.git}"
SAM2_REPO_URL="${SAM2_REPO_URL:-https://github.com/facebookresearch/sam2.git}"
TAPIR_REPO_URL="${TAPIR_REPO_URL:-https://github.com/google-deepmind/tapnet.git}"

GDINO_CFG="${GDINO_CFG:-third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GDINO_CKPT="${GDINO_CKPT:-checkpoints/groundingdino/groundingdino_swint_ogc.pth}"
GDINO_CKPT_URL="${GDINO_CKPT_URL:-}"

SAM2_REPO_DIR="${SAM2_REPO_DIR:-third_party/sam2}"
SAM2_CKPT="${SAM2_CKPT:-checkpoints/sam2/sam2.1_hiera_large.pt}"
SAM2_CKPT_URL="${SAM2_CKPT_URL:-}"

TAPIR_TORCH_PY="${TAPIR_TORCH_PY:-third_party/tapir/tapnet/torch/tapir_model.py}"
TAPIR_CKPT="${TAPIR_CKPT:-checkpoints/tapnet_checkpoints/bootstapir_checkpoint_v2.pt}"
TAPIR_CKPT_URL="${TAPIR_CKPT_URL:-}"

BERT_REPO_ID="${BERT_REPO_ID:-bert-base-uncased}"
BERT_DIR="${BERT_DIR:-checkpoints/bert-base-uncased}"

clone_if_missing() {
  local url="$1"
  local dst="$2"
  if [[ -d "$dst/.git" ]]; then
    echo "[download_motion_models] skip existing repo: $dst"
    return
  fi
  if [[ -d "$dst" ]] && find "$dst" -mindepth 1 ! -name '.gitkeep' -print -quit | grep -q .; then
    echo "[download_motion_models] skip existing directory: $dst"
    return
  fi
  if [[ -d "$dst" ]]; then
    find "$dst" -mindepth 1 -maxdepth 1 -name '.gitkeep' -delete
    rmdir "$dst" 2>/dev/null || true
  fi
  echo "[download_motion_models] git clone $url -> $dst"
  git clone --depth 1 "$url" "$dst"
}

download_if_requested() {
  local url="$1"
  local dst="$2"
  if [[ -f "$dst" && -s "$dst" ]]; then
    echo "[download_motion_models] skip existing file: $dst"
    return
  fi
  if [[ -z "$url" ]]; then
    return
  fi
  "$PYTHON_BIN" - "$url" "$dst" <<'PY'
import sys
from pathlib import Path
from urllib.request import urlretrieve

url = sys.argv[1]
dst = Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)
print(f"[download_motion_models] {url} -> {dst}")
urlretrieve(url, dst)
if not dst.is_file() or dst.stat().st_size <= 0:
    raise RuntimeError(f"downloaded file is empty: {dst}")
head = dst.read_bytes()[:512].lower()
if b"<html" in head or b"<!doctype html" in head:
    dst.unlink(missing_ok=True)
    raise RuntimeError(
        f"downloaded file looks like an HTML page, not a checkpoint: {dst}. "
        "Use a direct-download URL or download it manually."
    )
PY
}

download_hf_snapshot() {
  local repo_id="$1"
  local dst="$2"
  "$PYTHON_BIN" - "$repo_id" "$dst" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
print(f"[download_motion_models] hf snapshot {repo_id} -> {dst}")
snapshot_download(
    repo_id=repo_id,
    local_dir=str(dst),
    local_dir_use_symlinks=False,
    allow_patterns=[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "model.safetensors",
        "pytorch_model.bin",
    ],
)
PY
}

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  echo "[download_motion_models] LOCAL_FILES_ONLY=1; checking local files only."
else
  if [[ "$AUTO_CLONE" == "1" ]]; then
    clone_if_missing "$GROUNDINGDINO_REPO_URL" "third_party/GroundingDINO"
    clone_if_missing "$SAM2_REPO_URL" "$SAM2_REPO_DIR"
    clone_if_missing "$TAPIR_REPO_URL" "third_party/tapir"
  else
    echo "[download_motion_models] AUTO_CLONE=0; repository cloning is disabled."
  fi
  download_if_requested "$GDINO_CKPT_URL" "$GDINO_CKPT"
  download_if_requested "$SAM2_CKPT_URL" "$SAM2_CKPT"
  download_if_requested "$TAPIR_CKPT_URL" "$TAPIR_CKPT"
  if [[ ! -f "$BERT_DIR/vocab.txt" || ! -f "$BERT_DIR/config.json" ]] || \
     [[ ! -f "$BERT_DIR/model.safetensors" && ! -f "$BERT_DIR/pytorch_model.bin" ]]; then
    download_hf_snapshot "$BERT_REPO_ID" "$BERT_DIR"
  else
    echo "[download_motion_models] skip existing BERT text encoder: $BERT_DIR"
  fi
fi

missing=()
check_file() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" && -s "$path" ]] && ! head -c 512 "$path" | grep -Eiq '<html|<!doctype html'; then
    echo "[OK] $label: $path"
  else
    echo "[MISSING] $label: $path"
    missing+=("$path")
  fi
}

check_torch_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" || ! -s "$path" ]] || head -c 512 "$path" | grep -Eiq '<html|<!doctype html'; then
    echo "[MISSING] $label: $path"
    missing+=("$path")
    return
  fi
  if "$PYTHON_BIN" - "$path" <<'PY' >/dev/null 2>&1
import sys
import torch
torch.load(sys.argv[1], map_location="cpu")
PY
  then
    echo "[OK] $label: $path"
  else
    echo "[MISSING] $label is not a loadable torch checkpoint: $path"
    missing+=("$path")
  fi
}

check_dir() {
  local label="$1"
  local path="$2"
  if [[ -d "$path" ]]; then
    echo "[OK] $label: $path"
  else
    echo "[MISSING] $label: $path"
    missing+=("$path")
  fi
}

check_bert_dir() {
  local label="$1"
  local path="$2"
  if [[ -f "$path/vocab.txt" && -f "$path/config.json" ]] && \
     [[ -f "$path/model.safetensors" || -f "$path/pytorch_model.bin" ]]; then
    echo "[OK] $label: $path"
  else
    echo "[MISSING] $label: $path"
    missing+=("$path")
  fi
}

check_file "motion GroundingDINO config" "$GDINO_CFG"
check_torch_file "motion GroundingDINO checkpoint" "$GDINO_CKPT"
check_dir "motion SAM2 repository" "$SAM2_REPO_DIR"
check_torch_file "motion SAM2 checkpoint" "$SAM2_CKPT"
check_file "motion TAPIR torch implementation" "$TAPIR_TORCH_PY"
check_torch_file "motion TAPIR checkpoint" "$TAPIR_CKPT"
check_bert_dir "motion GroundingDINO BERT text encoder" "$BERT_DIR"

if (( ${#missing[@]} > 0 )); then
  echo "[download_motion_models] Required motion assets are missing:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo "[download_motion_models] Use AUTO_CLONE=1 for code repos and set GDINO_CKPT_URL/SAM2_CKPT_URL/TAPIR_CKPT_URL when you have stable weight URLs." >&2
  exit 2
fi
