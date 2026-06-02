
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .config import PreprocConfig
from .types import Entity

__all__ = [
    "normalize_text",
    "normalize_name",
    "normalize_key",
    "normalize_value",
    "normalize_attr_map",
    "coerce_entities_from_raw",
]

_ALLOW_NONALPHA_KEYS = {"number-or-id", "printed-text", "brand-or-logo"}

def _strip_decorations(s: str) -> str:
    
    return s.strip(" \t\r\n'\"`.,;:!?()[]{}<>")

def _canonical_hyphen(s: str) -> str:
    
    s = s.replace("_", "-")
    s = re.sub(r"\s*-\s*", "-", s)         
    s = re.sub(r"-{2,}", "-", s)           
    s = re.sub(r"\s+", " ", s)
    return s

def _normalize_unicode(s: str) -> str:
    
    s = unicodedata.normalize("NFC", s)
    
    s = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", s)
    return s

def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    strip: bool = True,
    canonical_hyphen: bool = True,
) -> str:
    if not isinstance(text, str):
        return ""
    s = _normalize_unicode(text)
    if lowercase:
        s = s.lower()
    if strip:
        s = _strip_decorations(s)
    if canonical_hyphen:
        s = _canonical_hyphen(s)
    
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _has_alpha(s: str) -> bool:
    return bool(re.search(r"[a-z]", s))

def _passes_length_gate(token: str, drop_short_token_len: int) -> bool:
    
    alpha_len = len(re.sub(r"[^a-z]", "", token))
    if alpha_len <= drop_short_token_len and len(token) <= drop_short_token_len + 1:
        return False
    return True

def normalize_name(name: str, cfg: PreprocConfig) -> str:
    return normalize_text(
        name or "",
        lowercase=cfg.lowercase,
        strip=cfg.strip,
        canonical_hyphen=cfg.canonical_hyphen,
    )

def normalize_key(key: str, cfg: PreprocConfig) -> str:
    return normalize_text(
        key or "",
        lowercase=cfg.lowercase,
        strip=cfg.strip,
        canonical_hyphen=cfg.canonical_hyphen,
    )

def normalize_value(key: str, val: str, cfg: PreprocConfig) -> Optional[str]:
    s = normalize_text(
        val or "",
        lowercase=cfg.lowercase,
        strip=cfg.strip,
        canonical_hyphen=cfg.canonical_hyphen,
    )
    if not s:
        return None

    key_norm = normalize_key(key, cfg)
    allow_nonalpha = key_norm in _ALLOW_NONALPHA_KEYS
    if cfg.drop_nonalpha and (not allow_nonalpha) and (not _has_alpha(s)):
        return None

    if (not allow_nonalpha) and (not _passes_length_gate(s, cfg.drop_short_token_len)):
        return None

    return s

def _dedup_stable(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def normalize_attr_map(attrs: Dict[str, List[str]], cfg: PreprocConfig) -> Dict[str, List[str]]:
    if not isinstance(attrs, dict):
        return {}

    out: Dict[str, List[str]] = {}
    for raw_k, raw_vals in attrs.items():
        k = normalize_key(str(raw_k), cfg)
        if not k:
            continue
        
        if k == "signature":
            continue

        vals: List[str] = []
        for v in (raw_vals or []):
            nv = normalize_value(k, v, cfg)
            if nv:
                vals.append(nv)

        vals = _dedup_stable(vals)
        if vals:
            out[k] = vals

    return out

def coerce_entities_from_raw(
    raw_doc: Dict,
    cfg: PreprocConfig,
    *,
    expect_path: Tuple[str, ...] = ("fine", "entities"),
) -> List[Entity]:
    obj = raw_doc or {}
    for k in expect_path:
        obj = obj.get(k, {}) if isinstance(obj, dict) else {}
    entities_raw = obj if isinstance(obj, list) else []

    out: List[Entity] = []
    for e in entities_raw:
        if not isinstance(e, dict):
            continue
        name = normalize_name(e.get("name", "") or "", cfg)
        if not name:
            continue
        attrs = e.get("attributes", {}) or {}
        attrs_norm = normalize_attr_map(attrs, cfg)
        out.append(Entity(id=str(e.get("id", "") or ""), name=name, attrs=attrs_norm))
    return out
