
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import Config
from .types import Entity, EntityRepr, Embedding
from .encoder import TextEncoder

__all__ = [
    "build_fragments_for_entity",
    "encode_entity_repr",
    "encode_entity_reprs",
]

def _get_key_weight(key: str, cfg: Config) -> float:
    kw = getattr(cfg, "repr", None)
    if kw and isinstance(getattr(kw, "key_weight", None), dict):
        return float(kw.key_weight.get(key, 1.0))
    return 1.0

def _include_name_in_set(cfg: Config) -> bool:
    r = getattr(cfg, "repr", None)
    
    return bool(getattr(r, "include_name_in_set", False))

def _name_in_set_weight(cfg: Config) -> float:
    r = getattr(cfg, "repr", None)
    return float(getattr(r, "name_weight_in_set", 1.0))

def _name_purpose(cfg: Config) -> str:
    r = getattr(cfg, "repr", None)
    return str(getattr(r, "name_channel_purpose", "query"))  

def _set_purpose(cfg: Config) -> str:
    r = getattr(cfg, "repr", None)
    return str(getattr(r, "set_channel_purpose", "passage"))  

@dataclass
class _Fragments:
    name_text: str
    frag_texts: List[str]
    frag_weights: List[float]

def build_fragments_for_entity(ent: Entity, cfg: Config) -> _Fragments:
    name_text = ent.name or ""

    frag_texts: List[str] = []
    frag_weights: List[float] = []

    for k, values in (ent.attrs or {}).items():
        if not values:
            continue
        w_k = _get_key_weight(k, cfg)
        for v in values:
            
            frag_texts.append(f"{k}: {v}")
            frag_weights.append(float(w_k))

    if _include_name_in_set(cfg) and name_text:
        frag_texts.append(name_text)
        frag_weights.append(_name_in_set_weight(cfg))

    if not frag_texts and name_text:
        frag_texts.append(name_text)
        frag_weights.append(1.0)

    return _Fragments(name_text=name_text, frag_texts=frag_texts, frag_weights=frag_weights)

def _weighted_average(vectors: List[np.ndarray], weights: List[float]) -> np.ndarray:
    if not vectors:
        return np.zeros((0,), dtype=np.float32)
    W = np.asarray(weights, dtype=np.float32)
    W = np.clip(W, 0.0, np.finfo(np.float32).max)
    if float(W.sum()) <= 0.0:
        W = np.ones_like(W)
    W = W / (W.sum() + 1e-12)
    M = np.stack(vectors, axis=0)  
    pooled = (M * W[:, None]).sum(axis=0)  
    
    norm = np.linalg.norm(pooled) + 1e-12
    pooled = (pooled / norm).astype(np.float32, copy=False)
    return pooled

def encode_entity_repr(
    ent: Entity,
    encoder: TextEncoder,
    cfg: Config,
    *,
    max_length: int = 512,
) -> EntityRepr:
    frags = build_fragments_for_entity(ent, cfg)

    name_embs = encoder.embed_texts([frags.name_text], purpose=_name_purpose(cfg), max_length=max_length)
    name_vec = name_embs[0].vec if isinstance(name_embs[0], Embedding) else np.asarray(name_embs[0], dtype=np.float32)

    piece_embs = encoder.embed_texts(frags.frag_texts, purpose=_set_purpose(cfg), max_length=max_length)
    piece_vecs = [e.vec if isinstance(e, Embedding) else np.asarray(e, dtype=np.float32) for e in piece_embs]
    set_vec = _weighted_average(piece_vecs, frags.frag_weights)

    return EntityRepr(
        entity=ent,
        name_text=frags.name_text,
        frag_texts=frags.frag_texts,
        frag_weights=frags.frag_weights,
        name_vec=name_vec,
        set_vec=set_vec,
    )

def encode_entity_reprs(
    ents: List[Entity],
    encoder: TextEncoder,
    cfg: Config,
    *,
    max_length: int = 512,
) -> List[EntityRepr]:
    out: List[EntityRepr] = []

    name_texts: List[str] = []
    set_texts_all: List[str] = []
    set_slices: List[Tuple[int, int]] = []  
    set_weights_all: List[float] = []
    fragments_cache: List[_Fragments] = []

    for ent in ents:
        fr = build_fragments_for_entity(ent, cfg)
        fragments_cache.append(fr)
        name_texts.append(fr.name_text)

        s = len(set_texts_all)
        set_texts_all.extend(fr.frag_texts)
        set_weights_all.extend(fr.frag_weights)
        e = len(set_texts_all)
        set_slices.append((s, e))

    name_embs = encoder.embed_texts(name_texts, purpose=_name_purpose(cfg), max_length=max_length)
    name_vecs = [e.vec if isinstance(e, Embedding) else np.asarray(e, dtype=np.float32) for e in name_embs]

    set_piece_embs = encoder.embed_texts(set_texts_all, purpose=_set_purpose(cfg), max_length=max_length)
    set_piece_vecs = [e.vec if isinstance(e, Embedding) else np.asarray(e, dtype=np.float32) for e in set_piece_embs]

    for ent, fr, name_v, (s, e) in zip(ents, fragments_cache, name_vecs, set_slices):
        set_vec = _weighted_average(set_piece_vecs[s:e], set_weights_all[s:e])
        out.append(
            EntityRepr(
                entity=ent,
                name_text=fr.name_text,
                frag_texts=fr.frag_texts,
                frag_weights=fr.frag_weights,
                name_vec=name_v,
                set_vec=set_vec,
            )
        )

    return out
