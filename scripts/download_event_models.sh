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
VIDEOLLAMA3_REPO_ID="${VIDEOLLAMA3_REPO_ID:-}"
VIDEOLLAMA3_DIR="${VIDEOLLAMA3_DIR:-checkpoints/videollama3-7b}"
DDM_CKPT_URL="${DDM_CKPT_URL:-}"
DDM_CKPT="${DDM_CKPT:-checkpoints/ddmnet/checkpoint.pth.tar}"
TRANSNETV2_WEIGHTS_URL="${TRANSNETV2_WEIGHTS_URL:-}"
TRANSNETV2_WEIGHTS="${TRANSNETV2_WEIGHTS:-checkpoints/transnetv2/transnetv2-pytorch-weights.pth}"

export LOCAL_FILES_ONLY E5_REPO_ID E5_DIR VIDEOLLAMA3_REPO_ID VIDEOLLAMA3_DIR
export DDM_CKPT_URL DDM_CKPT TRANSNETV2_WEIGHTS_URL TRANSNETV2_WEIGHTS

"$PYTHON_BIN" - <<'PY'
import os
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve


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


def ok_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    head = path.read_bytes()[:512].lower()
    return b"<html" not in head and b"<!doctype html" not in head


def status(label: str, path: Path, is_dir: bool = False) -> bool:
    ok = has_hf_model_payload(path) if is_dir else ok_file(path)
    print(f"[{'OK' if ok else 'MISSING'}] {label}: {path}")
    return ok


def snapshot(repo_id: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download_event_models] {repo_id} -> {local_dir}")
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


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download_event_models] {url} -> {dst}")
    urlretrieve(url, dst)
    if not ok_file(dst):
        head = dst.read_bytes()[:512].lower() if dst.exists() else b""
        if b"<html" in head or b"<!doctype html" in head:
            dst.unlink(missing_ok=True)
            raise RuntimeError(
                f"downloaded file looks like an HTML page, not a checkpoint: {dst}. "
                "Use a direct-download URL or download it manually."
            )
        raise RuntimeError(
            f"downloaded file is empty or invalid: {dst}"
        )


local_only = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
e5_dir = Path(os.environ["E5_DIR"])
vlm_dir = Path(os.environ["VIDEOLLAMA3_DIR"])
ddm = Path(os.environ["DDM_CKPT"])
transnet = Path(os.environ["TRANSNETV2_WEIGHTS"])

if local_only:
    print("[download_event_models] LOCAL_FILES_ONLY=1; checking local files only.")
else:
    if not has_hf_model_payload(e5_dir):
        snapshot(os.environ["E5_REPO_ID"], e5_dir)
    else:
        print(f"[download_event_models] skip existing E5: {e5_dir}")

    repo_id = os.environ.get("VIDEOLLAMA3_REPO_ID", "").strip()
    if repo_id:
        if not has_hf_model_payload(vlm_dir):
            snapshot(repo_id, vlm_dir)
        else:
            print(f"[download_event_models] skip existing VideoLLaMA3: {vlm_dir}")
    else:
        print("[download_event_models] VIDEOLLAMA3_REPO_ID not set; skipping VideoLLaMA3 download.")

    if os.environ.get("DDM_CKPT_URL", "").strip() and not ok_file(ddm):
        download_file(os.environ["DDM_CKPT_URL"], ddm)
    elif not ok_file(ddm):
        print("[download_event_models] DDM_CKPT_URL not set; cannot auto-download DDM checkpoint.")

    if os.environ.get("TRANSNETV2_WEIGHTS_URL", "").strip() and not ok_file(transnet):
        download_file(os.environ["TRANSNETV2_WEIGHTS_URL"], transnet)
    elif not ok_file(transnet):
        print("[download_event_models] TRANSNETV2_WEIGHTS_URL not set; cannot auto-download TransNetV2 weights.")

missing = []
if not status("event E5", e5_dir, is_dir=True):
    missing.append(str(e5_dir))
if not status("event VideoLLaMA3", vlm_dir, is_dir=True):
    missing.append(str(vlm_dir))
if not status("event DDM checkpoint", ddm):
    missing.append(str(ddm))
if not status("event TransNetV2 weights", transnet):
    missing.append(str(transnet))

if missing:
    print("[download_event_models] Required event assets are missing:", file=sys.stderr)
    for path in missing:
        print(f"  {path}", file=sys.stderr)
    print("[download_event_models] Set VIDEOLLAMA3_REPO_ID, DDM_CKPT_URL, and TRANSNETV2_WEIGHTS_URL when automatic download is possible.", file=sys.stderr)
    sys.exit(2)
PY
