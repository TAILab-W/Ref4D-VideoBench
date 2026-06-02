

"""
Common I/O utilities for the event_eval project:
- Safe (atomic) JSON/YAML read & write
- Directory helpers
- Config deep-merge
- Deterministic random seed setup
- Pair ID slugify
"""

from __future__ import annotations
import os
import io
import json
import yaml
import tempfile
import shutil
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Mapping

try:
    import numpy as np  
except Exception:  
    np = None  

LOGGER = logging.getLogger("event_eval.common.io")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter(
        "[%(levelname)s] %(name)s: %(message)s"
    )
    h.setFormatter(fmt)
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)






def ensure_dir(path: os.PathLike[str] | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _atomic_write_text(text: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(out_path.parent)) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(out_path)


def _atomic_write_bytes(data: bytes, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(out_path.parent)) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(out_path)






def read_json(path: os.PathLike[str] | str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {p}\n{e}") from e


def write_json(obj: Any,
               path: os.PathLike[str] | str,
               indent: int = 2,
               sort_keys: bool = False) -> None:
    p = Path(path)
    text = json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
    _atomic_write_text(text, p)


def read_yaml(path: os.PathLike[str] | str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML must be a mapping: {p}")
    return data


def write_yaml(obj: Mapping[str, Any],
               path: os.PathLike[str] | str) -> None:
    p = Path(path)
    text = yaml.safe_dump(dict(obj), allow_unicode=True, sort_keys=False)
    _atomic_write_text(text, p)






def deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    def _merge(a: Dict[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
        return out

    return _merge(base, override)






def set_random_seed(seed: int,
                    deterministic_torch: bool = True,
                    quiet: bool = False) -> None:
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        if not quiet:
            LOGGER.warning("numpy not available; skipping numpy seed")

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except Exception:
        if not quiet:
            LOGGER.warning("torch not available; skipping torch seed")






def expand_path(path: os.PathLike[str] | str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def slugify(value: str, max_len: int = 128) -> str:
    out_chars = []
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    for ch in value:
        out_chars.append(ch if ch in keep else "-")
    slug = "".join(out_chars).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug


def copy_tree(src: os.PathLike[str] | str, dst: os.PathLike[str] | str, overwrite: bool = True) -> None:
    src_p, dst_p = Path(src), Path(dst)
    if not src_p.exists():
        raise FileNotFoundError(f"copy_tree source not found: {src_p}")
    if dst_p.exists() and overwrite:
        shutil.rmtree(dst_p)
    shutil.copytree(src_p, dst_p)
