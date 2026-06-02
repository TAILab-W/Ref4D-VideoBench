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
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
E5_REPO_ID="${E5_REPO_ID:-intfloat/e5-large-v2}"
E5_DIR="${E5_DIR:-checkpoints/e5-large-v2}"
MINICPM_REPO_ID="${MINICPM_REPO_ID:-openbmb/MiniCPM-V-4_5}"
MINICPM_DIR="${MINICPM_DIR:-checkpoints/minicpm-v-4_5}"
WORLD_REPO_IDS="${WORLD_REPO_IDS:-}"

export LOCAL_FILES_ONLY E5_REPO_ID E5_DIR MINICPM_REPO_ID MINICPM_DIR WORLD_REPO_IDS

"$PYTHON_BIN" - <<'PY'
import os
import json
import sys
from pathlib import Path


def has_payload(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() and p.name != ".gitkeep" for p in path.rglob("*"))


def has_hf_model_payload(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").is_file()
    has_tokenizer = (
        any(path.glob("tokenizer*"))
        or (path / "vocab.txt").is_file()
        or any(path.glob("*.model"))
    )
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
            shards = sorted(set(weight_map.values()))
        except Exception:
            shards = []
        has_model = bool(shards) and all(valid_safetensors(path / shard) for shard in shards)
    else:
        has_model = any(valid_safetensors(p) for p in path.glob("*.safetensors"))
    if not has_model:
        has_model = any(p.is_file() and p.stat().st_size > 1024 * 1024 for p in path.glob("pytorch_model*.bin"))
    return has_config and has_model and has_tokenizer


def valid_safetensors(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 1024:
        return False
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        return bool(keys)
    except Exception:
        return False


def status(label: str, path: Path) -> bool:
    ok = has_hf_model_payload(path)
    print(f"[{'OK' if ok else 'MISSING'}] {label}: {path}")
    return ok


def snapshot(repo_id: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download_semantic_world_models] {repo_id} -> {local_dir}")
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
        "allow_patterns": [
            "config.json",
            "configuration*.py",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "image_processing*.py",
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab*",
            "merges.txt",
            "sentencepiece*",
            "*.model",
            "chat_template.json",
            "model.safetensors",
            "model-*.safetensors",
            "*.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model-*.bin",
            "pytorch_model.bin.index.json",
            "*.py",
        ],
    }
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    snapshot_download(**kwargs)


local_only = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
targets = [
    ("semantic E5", os.environ["E5_REPO_ID"], Path(os.environ["E5_DIR"])),
    ("semantic MiniCPM", os.environ["MINICPM_REPO_ID"], Path(os.environ["MINICPM_DIR"])),
]

world_repos = [x.strip() for x in os.environ.get("WORLD_REPO_IDS", "").split(",") if x.strip()]
for item in world_repos:
    if "=" in item:
        repo_id, rel_dir = item.split("=", 1)
        local_dir = Path(rel_dir)
    else:
        repo_id = item
        local_dir = Path("checkpoints/world") / repo_id.rsplit("/", 1)[-1]
    targets.append((f"optional world {repo_id}", repo_id, local_dir))

if local_only:
    print("[download_semantic_world_models] LOCAL_FILES_ONLY=1; checking local files only.")
else:
    for label, repo_id, local_dir in targets:
        if has_hf_model_payload(local_dir):
            print(f"[download_semantic_world_models] skip existing {label}: {local_dir}")
            continue
        snapshot(repo_id, local_dir)

missing = []
for label, _repo_id, local_dir in targets:
    ok = status(label, local_dir)
    if not ok and not label.startswith("optional world"):
        missing.append(str(local_dir))

if world_repos:
    print("[download_semantic_world_models] Additional WORLD_REPO_IDS are optional; MiniCPM-V-4.5 (openbmb/MiniCPM-V-4_5) is the default world evaluator model.")

if missing:
    print("[download_semantic_world_models] Required semantic model assets are missing:", file=sys.stderr)
    for path in missing:
        print(f"  {path}", file=sys.stderr)
    sys.exit(2)
PY
