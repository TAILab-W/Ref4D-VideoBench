
"""
Build candidate edge masks and similarity matrices for bipartite matching.

Inputs (merged, per-side):
  - outputs/event/cache/events_merged/{ref|gen}/<id>.newevents.json
      canonical merged event evidence; each event must provide valid normalized
      timeline fields s/e in [0,1]. Optional s_abs/e_abs may also be present.
  - outputs/event/cache/embeds/{ref|gen}/<id>.emb.merged.json
      id-aligned embeddings; each item: {"id","emb":[...]}.

Outputs:
  - outputs/event/cache/match/<pair_id>/gate_masks.npz
      - ref_ids: list[str]  (length Nr)
      - gen_ids: list[str]  (length Ng)
      - sim_sem: float32 [Nr,Ng] in [0,1]
      - r_tiou : float32 [Nr,Ng] in [0,1]
      - gate   : bool    [Nr,Ng] where (sim_sem >= s0) & (r_tiou >= u0)
      - s0, u0, delta: float32
      - meta: json string (portable paths, shapes, and optional abs-duration stats)

Notes:
  - Merged events and merged embeddings must be complete, unique, and id-aligned.
  - This module does not recover normalized s/e from absolute timestamps.
    The merged event evidence is required to already contain valid normalized s/e.
"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple
from pathlib import Path
import argparse
import logging
import numpy as np

from ..common.io import read_yaml, ensure_dir

LOGGER = logging.getLogger("event_eval.match.gating")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


def _load_events(path: str) -> List[Dict[str, Any]]:
    dat = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(dat, dict) and "events" in dat:
        evs = dat["events"]
    else:
        evs = dat
    if not isinstance(evs, list):
        raise ValueError(f"Invalid merged event evidence in {path}: expected list or dict with key 'events'.")

    out = []
    for idx, d in enumerate(evs):
        if not isinstance(d, dict):
            raise ValueError(f"Invalid merged event evidence in {path}: events[{idx}] must be a dict.")
        e = dict(d)
        e["id"] = e.get("id") or e.get("eid") or e.get("event_id")

        has_norm = ("s" in d) and ("e" in d)
        e["_has_norm"] = bool(has_norm)
        if has_norm:
            e["s"] = float(d["s"])
            e["e"] = float(d["e"])

        has_abs = ("s_abs" in d) and ("e_abs" in d)
        e["_has_abs"] = bool(has_abs)
        if has_abs:
            e["s_abs"] = float(d["s_abs"])
            e["e_abs"] = float(d["e_abs"])
        out.append(e)

    def _sort_key(x: Dict[str, Any]) -> Tuple[float, float]:
        if x.get("_has_abs", False):
            return (float(x["s_abs"]), float(x["e_abs"]))
        if x.get("_has_norm", False):
            return (float(x["s"]), float(x["e"]))
        return (float("inf"), float("inf"))

    out.sort(key=_sort_key)
    return out


def _validate_events_have_valid_norm(events: List[Dict[str, Any]], path: str) -> None:
    errs: List[str] = []
    seen_ids = set()
    tol = 1e-6
    for idx, e in enumerate(events):
        eid = e.get("id")
        if not isinstance(eid, str) or not eid:
            errs.append(f"event[{idx}] missing id")
            continue
        if eid in seen_ids:
            errs.append(f"event[{idx}] id={eid}: duplicate id")
            continue
        seen_ids.add(eid)

        if not e.get("_has_norm", False):
            errs.append(f"event[{idx}] id={eid}: missing normalized s/e")
            continue
        s = e.get("s")
        t = e.get("e")
        if not np.isfinite(s) or not np.isfinite(t):
            errs.append(f"event[{idx}] id={eid}: non-finite s/e ({s}, {t})")
            continue
        if s < -tol or t > 1.0 + tol or t < s - tol:
            errs.append(f"event[{idx}] id={eid}: invalid normalized interval s={s}, e={t}")

        if e.get("_has_abs", False):
            s_abs = e.get("s_abs")
            e_abs = e.get("e_abs")
            if not np.isfinite(s_abs) or not np.isfinite(e_abs):
                errs.append(f"event[{idx}] id={eid}: non-finite s_abs/e_abs ({s_abs}, {e_abs})")
            elif e_abs < s_abs:
                errs.append(f"event[{idx}] id={eid}: invalid absolute interval s_abs={s_abs}, e_abs={e_abs}")

    if errs:
        joined = "\n  - ".join(errs[:20])
        more = "" if len(errs) <= 20 else f"\n  ... and {len(errs) - 20} more"
        raise ValueError(f"Invalid merged event evidence in {path}:\n  - {joined}{more}")


def _load_embeds(path: str) -> Dict[str, List[float]]:
    dat = json.loads(Path(path).read_text(encoding="utf-8"))
    m: Dict[str, List[float]] = {}
    if isinstance(dat, dict) and all(isinstance(k, str) for k in dat.keys()):
        for k, v in dat.items():
            if k in m:
                raise ValueError(f"Invalid merged embeddings in {path}: duplicate id '{k}'.")
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError(f"Invalid merged embeddings in {path}: id '{k}' has empty embedding.")
            vec = [float(x) for x in v]
            if not np.all(np.isfinite(np.asarray(vec, dtype=np.float64))):
                raise ValueError(f"Invalid merged embeddings in {path}: id '{k}' has non-finite values.")
            m[k] = _l2norm(vec)
        return m
    if isinstance(dat, list):
        for idx, it in enumerate(dat):
            if not isinstance(it, dict):
                raise ValueError(f"Invalid merged embeddings in {path}: items[{idx}] must be a dict.")
            eid = it.get("id") or it.get("eid") or it.get("event_id")
            emb = it.get("emb") or it.get("embedding") or it.get("vec")
            if not isinstance(eid, str) or not eid:
                raise ValueError(f"Invalid merged embeddings in {path}: items[{idx}] missing id.")
            if eid in m:
                raise ValueError(f"Invalid merged embeddings in {path}: duplicate id '{eid}'.")
            if not isinstance(emb, list) or len(emb) == 0:
                raise ValueError(f"Invalid merged embeddings in {path}: id '{eid}' has empty embedding.")
            vec = [float(x) for x in emb]
            if not np.all(np.isfinite(np.asarray(vec, dtype=np.float64))):
                raise ValueError(f"Invalid merged embeddings in {path}: id '{eid}' has non-finite values.")
            m[eid] = _l2norm(vec)
        return m
    raise ValueError(f"Invalid merged embeddings in {path}: expected list or dict.")


def _l2norm(vec: List[float]) -> List[float]:
    s = float(np.linalg.norm(np.asarray(vec, dtype=np.float64)))
    if s <= 0:
        return []
    return [float(x / s) for x in vec]


def _video_abs_duration(events: List[Dict[str, Any]]) -> float | None:
    if not events or not all(("s_abs" in e and "e_abs" in e) for e in events):
        return None
    s0 = min(e["s_abs"] for e in events)
    e1 = max(e["e_abs"] for e in events)
    return float(max(1e-9, e1 - s0))


def _pairwise_cos_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    C = A @ B.T
    return C.astype(np.float32)


def _r_tiou_delta(a: Tuple[float, float], b: Tuple[float, float], delta: float) -> float:
    s1, e1 = a
    s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(0.0, max(e1, e2) - min(s1, s2))
    return float((inter + delta) / (union + delta)) if union >= 0.0 else 0.0


def _matrix_r_tiou(ref_events, gen_events, delta: float) -> np.ndarray:
    Nr, Ng = len(ref_events), len(gen_events)
    out = np.zeros((Nr, Ng), dtype=np.float32)
    for i, re in enumerate(ref_events):
        a = (float(re["s"]), float(re["e"]))
        for j, ge in enumerate(gen_events):
            b = (float(ge["s"]), float(ge["e"]))
            out[i, j] = _r_tiou_delta(a, b, delta)
    return out


def _portable_path(path: str) -> str:
    p = Path(path)
    parts = p.parts
    for anchor in ("outputs", "data", "ref4d_eval", "docs", "checkpoints", "third_party"):
        if anchor in parts:
            idx = parts.index(anchor)
            return Path(*parts[idx:]).as_posix()
    return p.name


def _validate_event_embed_alignment(event_ids: List[str], emb_map: Dict[str, List[float]], emb_path: str, side: str) -> None:
    event_set = set(event_ids)
    emb_set = set(emb_map.keys())
    missing = sorted(event_set - emb_set)
    extra = sorted(emb_set - event_set)
    errs: List[str] = []
    if missing:
        errs.append(f"{side}: missing embeddings for ids {missing[:10]}")
    if extra:
        errs.append(f"{side}: extra embedding ids not present in events {extra[:10]}")
    if errs:
        joined = "; ".join(errs)
        raise ValueError(f"Invalid merged event/embedding alignment ({emb_path}): {joined}")

    dims = {len(v) for v in emb_map.values()}
    if len(dims) != 1:
        raise ValueError(f"Invalid merged embeddings in {emb_path}: inconsistent embedding dimensions {sorted(dims)}.")
    dim = next(iter(dims)) if dims else 0
    if dim <= 0:
        raise ValueError(f"Invalid merged embeddings in {emb_path}: empty embedding dimension.")


def save_gate_npz(ref_events_path: str,
                  ref_embeds_path: str,
                  gen_events_path: str,
                  gen_embeds_path: str,
                  cfg_path: str,
                  out_npz_path: str) -> Dict[str, Any]:
    cfg = read_yaml(cfg_path)

    def _getf(*keys, default=None):
        for k in keys:
            v = cfg
            for kk in k.split("."):
                if isinstance(v, dict) and kk in v:
                    v = v[kk]
                else:
                    v = None
                    break
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return float(default) if default is not None else None

    s0 = _getf("gating.s0", default=0.8)
    u0 = _getf("gating.u0", default=0.5)
    semantic_floor = _getf("gating.semantic_floor", default=0.0)
    delta = _getf("gating.rtiou.delta", default=1e-6)
    if semantic_floor < 0.0 or semantic_floor >= 1.0:
        raise ValueError(f"gating.semantic_floor must be in [0, 1), got {semantic_floor}")

    ref_events = _load_events(ref_events_path)
    gen_events = _load_events(gen_events_path)
    _validate_events_have_valid_norm(ref_events, ref_events_path)
    _validate_events_have_valid_norm(gen_events, gen_events_path)

    dur_ref = _video_abs_duration(ref_events)
    dur_gen = _video_abs_duration(gen_events)

    ref_ids = [e["id"] for e in ref_events]
    gen_ids = [e["id"] for e in gen_events]

    ref_emb_map = _load_embeds(ref_embeds_path)
    gen_emb_map = _load_embeds(gen_embeds_path)
    _validate_event_embed_alignment(ref_ids, ref_emb_map, ref_embeds_path, "ref")
    _validate_event_embed_alignment(gen_ids, gen_emb_map, gen_embeds_path, "gen")

    ref_dim = len(next(iter(ref_emb_map.values()))) if ref_emb_map else 0
    gen_dim = len(next(iter(gen_emb_map.values()))) if gen_emb_map else 0
    if ref_dim != gen_dim:
        raise ValueError(
            f"Embedding dimension mismatch: ref={ref_dim} ({ref_embeds_path}) vs gen={gen_dim} ({gen_embeds_path})."
        )
    D = ref_dim

    Nr, Ng = len(ref_ids), len(gen_ids)
    A = np.zeros((Nr, D), dtype=np.float32)
    B = np.zeros((Ng, D), dtype=np.float32)
    for i, eid in enumerate(ref_ids):
        A[i, :] = np.asarray(ref_emb_map[eid], dtype=np.float32)
    for j, eid in enumerate(gen_ids):
        B[j, :] = np.asarray(gen_emb_map[eid], dtype=np.float32)

    cos = _pairwise_cos_sim(A, B)
    raw_sim_sem = (cos + 1.0) * 0.5
    sim_sem = np.clip((raw_sim_sem - semantic_floor) / (1.0 - semantic_floor), 0.0, 1.0)
    r_tiou = _matrix_r_tiou(ref_events, gen_events, delta)
    gate = (sim_sem >= s0) & (r_tiou >= u0)

    ensure_dir(Path(out_npz_path).parent)
    meta = {
        "ref_events": _portable_path(ref_events_path),
        "gen_events": _portable_path(gen_events_path),
        "ref_embeds": _portable_path(ref_embeds_path),
        "gen_embeds": _portable_path(gen_embeds_path),
        "cfg": _portable_path(cfg_path),
        "Nr": int(Nr), "Ng": int(Ng),
        "semantic_floor": float(semantic_floor),
        "delta": float(delta),
        "dur_ref_sec": float(dur_ref) if dur_ref is not None else None,
        "dur_gen_sec": float(dur_gen) if dur_gen is not None else None
    }
    np.savez_compressed(
        out_npz_path,
        ref_ids=np.array(ref_ids, dtype=object),
        gen_ids=np.array(gen_ids, dtype=object),
        sim_sem=sim_sem.astype(np.float32),
        r_tiou=r_tiou.astype(np.float32),
        gate=gate.astype(np.bool_),
        s0=np.float32(s0),
        u0=np.float32(u0),
        semantic_floor=np.float32(semantic_floor),
        delta=np.float32(delta),
        meta=json.dumps(meta)
    )
    LOGGER.info(f"Wrote gate masks: {out_npz_path} (Nr={Nr}, Ng={Ng}, s0={s0}, u0={u0}, semantic_floor={semantic_floor}, delta={delta})")
    return {"nr": Nr, "ng": Ng, "out": str(out_npz_path), "s0": float(s0), "u0": float(u0), "semantic_floor": float(semantic_floor), "delta": float(delta)}


def parse_args():
    ap = argparse.ArgumentParser(description="Build gating masks and similarity matrices for event matching.")
    ap.add_argument("--ref-events", type=str, required=True)
    ap.add_argument("--ref-embeds", type=str, required=True)
    ap.add_argument("--gen-events", type=str, required=True)
    ap.add_argument("--gen-embeds", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="outputs/event/cache/match/<pair_id>/gate_masks.npz")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_gate_npz(
        ref_events_path=args.ref_events,
        ref_embeds_path=args.ref_embeds,
        gen_events_path=args.gen_events,
        gen_embeds_path=args.gen_embeds,
        cfg_path=args.config,
        out_npz_path=args.out
    )
