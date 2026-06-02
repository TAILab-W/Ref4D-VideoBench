

"""
E5 encoder for event-side text embeddings.

I/O:
- Input : outputs/event/cache/vlm/{ref|gen}/{sample_id}.vlm.json
          Each item must contain valid normalized event fields:
          {"id","s","e","text", ...}
- Output: outputs/event/cache/embeds/{ref|gen}/{sample_id}.emb.json
          Expanded list. Each item (one micro-event):
            {
              "id": "e0001#1",
              "parent_id": "e0001",
              "s_abs","e_abs","s","e",
              "text": "one clause",
              "emb": [float, ...],   # L2-normalized when normalize=True
              "norm": true
            }
- Optional split export (embed.microevent.split_export = true):
  outputs/event/cache/embeds/{ref|gen}/{sample_id}.parts/e0001__1.emb.json

Notes:
- This module encodes event texts for downstream semantic similarity in event matching.
- Empty segment text or empty micro-event splits are treated as schema errors and raise.
"""

from __future__ import annotations
import os
import math
import re
import argparse
import logging
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F

from ..common.io import read_json, write_json, read_yaml, ensure_dir

LOGGER = logging.getLogger("event_eval.embed.e5")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]






def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "fp32").lower()
    if s in ("fp16", "float16"): return torch.float16
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    return torch.float32

def _expand_placeholders(s: str, cfg: Dict[str, Any]) -> str:
    data_root = cfg.get("paths", {}).get("data_root", "outputs/event/cache")
    return s.replace("${paths.data_root}", str(data_root))

def _infer_side(path: str) -> str:
    parts = [p.lower() for p in Path(path).parts]
    for anchor in ("vlm", "events"):
        if anchor in parts:
            idx = parts.index(anchor)
            if idx + 1 < len(parts) and parts[idx + 1] in ("ref", "gen"):
                return parts[idx + 1]
    return "ref"

def _infer_sample_id(vlm_path: str) -> str:
    name = Path(vlm_path).name
    if name.endswith(".vlm.json"):
        return name[:-len(".vlm.json")]
    if name.endswith(".events.json"):
        return name[:-len(".events.json")]
    return Path(vlm_path).stem

def _split_microevents(
    text: str,
    delims: Optional[List[str]] = None,
    sep: Optional[str] = None,
    strip_punct: bool = True,
    max_parts: int = 8,
) -> List[str]:
    if text is None:
        return []
    t0 = text.strip()
    if not t0:
        return []

    if delims and isinstance(delims, list) and any(delims):
        _delims = [str(d) for d in delims if d is not None and str(d) != ""]
    elif sep:
        _delims = [str(sep)]
    else:
        _delims = ["|", "。", "；", ";", "."]

    pattern = "|".join([re.escape(d) for d in _delims])
    parts = re.split(pattern, t0)

    out: List[str] = []
    for t in parts:
        t = t.strip()
        if strip_punct:
            t = t.strip(" .;、，。！？!?:：-—|")
        if t:
            out.append(t)
        if max_parts and len(out) >= max_parts:
            break
    return out

def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)

def _device_from_cfg(device: str) -> str:
    device = (device or "auto").strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' but CUDA is not available.")
        return "cuda"
    raise ValueError(f"Unsupported device: {device!r}. Expected one of: auto, cpu, cuda.")

def _resolve_repo_relative_path(path_str: str) -> str:
    p = Path(os.path.expandvars(os.path.expanduser(str(path_str))))
    if p.is_absolute():
        return str(p)
    candidate = (_PROJECT_ROOT / p).resolve()
    if candidate.exists():
        return str(candidate)
    return str(p)

def _validate_vlm_record(seg: Dict[str, Any], idx: int, vlm_json_path: str) -> None:
    raw_id = seg.get("id") or seg.get("eid") or seg.get("event_id")
    seg_id = "" if raw_id is None else str(raw_id)
    if not seg_id:
        raise ValueError(f"VLM item[{idx}] missing non-empty id in {vlm_json_path}")

    if "s" not in seg or "e" not in seg:
        raise ValueError(f"VLM item[{idx}] missing normalized s/e in {vlm_json_path}")
    try:
        s = float(seg["s"])
        e = float(seg["e"])
    except Exception as exc:
        raise ValueError(f"VLM item[{idx}] has non-numeric s/e in {vlm_json_path}") from exc
    if not (math.isfinite(s) and math.isfinite(e)):
        raise ValueError(f"VLM item[{idx}] has non-finite s/e in {vlm_json_path}")
    tol = 1e-6
    if s < -tol or e < -tol or s > 1.0 + tol or e > 1.0 + tol or e < s - tol:
        raise ValueError(f"VLM item[{idx}] has invalid normalized interval s={s}, e={e} in {vlm_json_path}")

    text = seg.get("text")
    if not isinstance(text, str):
        raise ValueError(f"VLM item[{idx}] text must be a string in {vlm_json_path}")
    if not text.strip():
        raise ValueError(f"VLM item[{idx}] has empty text in {vlm_json_path}")






def load_e5(cfg: Dict[str, Any]):
    from transformers import AutoTokenizer, AutoModel  

    ecfg = cfg.get("embed", {}) or {}
    model_name = _resolve_repo_relative_path(ecfg.get("model", "checkpoints/e5-large-v2"))
    device = _device_from_cfg(ecfg.get("device", "auto"))
    dtype = _dtype_from_str(ecfg.get("dtype", "fp32"))
    local_files_only = bool(ecfg.get("local_files_only", True))

    if device == "cpu" and dtype is not torch.float32:
        LOGGER.info(f"[E5] Forcing dtype=float32 on CPU (was {dtype}).")
        dtype = torch.float32

    LOGGER.info(
        f"Loading E5 model: {model_name} "
        f"(device={device}, dtype={dtype}, local_files_only={local_files_only})"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, local_files_only=local_files_only)
    model.to(device)
    model.eval()
    return tokenizer, model, device, dtype






@torch.inference_mode()
def encode_texts(texts: List[str],
                 tokenizer,
                 model,
                 device: str,
                 dtype: torch.dtype,
                 batch_size: int,
                 max_length: int,
                 text_prefix: str,
                 normalize: bool) -> torch.Tensor:
    all_out: List[torch.Tensor] = []
    n = len(texts)
    for st in range(0, n, batch_size):
        ed = min(n, st + batch_size)
        batch = [f"{text_prefix}{t}" for t in texts[st:ed]]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        last_hidden = out.last_hidden_state
        pooled = _mean_pool(last_hidden, enc["attention_mask"])
        if normalize:
            pooled = _normalize(pooled)
        all_out.append(pooled.detach().to("cpu"))
    return torch.cat(all_out, dim=0) if all_out else torch.empty(0, model.config.hidden_size)






def pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.empty(a.size(0), b.size(0))
    a_n = _normalize(a) if not torch.allclose(a.norm(dim=1), torch.ones(a.size(0)), atol=1e-3) else a
    b_n = _normalize(b) if not torch.allclose(b.norm(dim=1), torch.ones(b.size(0)), atol=1e-3) else b
    return a_n @ b_n.t()

def pairwise_sim_sem(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (pairwise_cosine(a, b) + 1.0) / 2.0






def _derive_out_paths(cfg: Dict[str, Any], side: str, sample_id: str) -> Tuple[Path, Optional[Path]]:
    ecfg = cfg.get("embed", {}) or {}
    exp = ecfg.get("export", {}) or {}

    out_dir = _expand_placeholders(exp.get("out_dir", "${paths.data_root}/embeds"), cfg)
    fname = exp.get("fname_ref" if side == "ref" else "fname_gen", "{sample_id}.emb.json")
    out_path = Path(out_dir) / fname.format(sample_id=sample_id)
    ensure_dir(out_path.parent)

    parts_dir_pat = exp.get("parts_dir", None)
    parts_dir = None
    if parts_dir_pat:
        parts_rel = parts_dir_pat.format(side=side, sample_id=sample_id)
        parts_dir = Path(out_dir) / parts_rel
        ensure_dir(parts_dir)
    return out_path, parts_dir

def run(vlm_json_path: str, out_json_path: Optional[str], cfg_path: str) -> Dict[str, Any]:
    cfg = read_yaml(cfg_path)
    ecfg = cfg.get("embed", {}) or {}

    tokenizer, model, device, dtype = load_e5(cfg)

    bs = int(ecfg.get("batch_size", 64))
    max_len = int(ecfg.get("max_length", 128))
    text_prefix = str(ecfg.get("text_prefix", "query: "))
    norm = bool(ecfg.get("normalize", True))

    mcfg = ecfg.get("microevent", {}) or {}
    delims_cfg = mcfg.get("delims", None)
    delims = [str(d) for d in delims_cfg] if isinstance(delims_cfg, list) else None
    sep = str(mcfg.get("sep", "")).strip() or None
    strip_punct = bool(mcfg.get("strip_punct", True))
    max_parts = int(mcfg.get("max_parts", 8))
    split_export = bool(mcfg.get("split_export", True))

    vlm = read_json(vlm_json_path)
    if not isinstance(vlm, list):
        raise ValueError(f"vlm json must be list: {vlm_json_path}")

    side = _infer_side(vlm_json_path)
    sample_id = _infer_sample_id(vlm_json_path)

    if out_json_path:
        out_path = Path(out_json_path)
        ensure_dir(out_path.parent)
        parts_dir = None
    else:
        out_path, parts_dir = _derive_out_paths(cfg, side, sample_id)
        if not split_export:
            parts_dir = None

    expanded: List[Dict[str, Any]] = []
    for idx_seg, seg in enumerate(vlm):
        if not isinstance(seg, dict):
            raise ValueError(f"VLM item[{idx_seg}] must be a dict in {vlm_json_path}")
        _validate_vlm_record(seg, idx_seg, vlm_json_path)

        base = {k: seg[k] for k in seg.keys() if k not in ("text",)}
        parent_id = str(seg.get("id") or seg.get("eid") or seg.get("event_id"))
        texts = _split_microevents(
            seg.get("text", ""),
            delims=delims,
            sep=sep,
            strip_punct=strip_punct,
            max_parts=max_parts,
        )
        if not texts:
            raise ValueError(f"VLM item[{idx_seg}] has no non-empty micro-events after split in {vlm_json_path}")
        for idx, t in enumerate(texts, start=1):
            rec = dict(base)
            rec["parent_id"] = parent_id
            rec["id"] = f"{parent_id}#{idx}"
            rec["text"] = t
            expanded.append(rec)

    texts = [r["text"] for r in expanded]
    embs = encode_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        dtype=dtype,
        batch_size=bs,
        max_length=max_len,
        text_prefix=text_prefix,
        normalize=norm,
    )
    dim = embs.size(-1)

    for rec, vec in zip(expanded, embs):
        rec["emb"] = vec.tolist()
        rec["norm"] = bool(norm)
    write_json(expanded, out_path, indent=2)
    LOGGER.info(f"Wrote aggregate embeddings: {out_path} (items={len(expanded)}, dim={dim})")

    if parts_dir is not None:
        for rec in expanded:
            fname = f"{rec['id'].replace('#','__')}.emb.json"
            write_json([rec], parts_dir / fname, indent=2)
        LOGGER.info(f"Wrote per-micro-event files under: {parts_dir}")

    return {"n_items": len(expanded), "dim": dim, "out": str(out_path), "parts_dir": str(parts_dir) if parts_dir else None}






def parse_args():
    ap = argparse.ArgumentParser(description="Encode event VLM outputs into E5 embeddings for event matching")
    ap.add_argument(
        "--vlm",
        type=str,
        required=True,
        help="Path to VLM json, e.g., outputs/event/cache/vlm/ref/<sample_id>.vlm.json",
    )
    ap.add_argument("--config", type=str, required=True, help="Path to event model_embed.yaml")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional override for aggregate output path (pipeline mode usually passes outputs/event/cache/embeds/...)",
    )
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        vlm_json_path=args.vlm,
        out_json_path=args.out,
        cfg_path=args.config,
    )
