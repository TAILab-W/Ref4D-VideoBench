"""
Inputs:
  - pairs.json
  - ref_events_json / gen_events_json:
      explicitly passed merged event evidence files; spans are read from
      normalized s/e and must already be valid in [0,1].
"""

from __future__ import annotations
import json
import argparse
import logging
from typing import Any, Dict, List, Tuple
from pathlib import Path
import numpy as np

from ..common.io import read_json, read_yaml

LOGGER = logging.getLogger("event_eval.metrics.ERel")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)




def _load_span_map(events_path: str) -> Dict[str, Tuple[float, float]]:
    data = read_json(events_path)
    if isinstance(data, dict) and "events" in data:
        data = data["events"]
    spans: Dict[str, Tuple[float, float]] = {}
    for idx, it in enumerate(data):
        eid = str(it.get("id") or it.get("eid") or it.get("event_id") or "")
        if not eid:
            raise ValueError(f"Event[{idx}] missing id in {events_path}")
        if eid in spans:
            raise ValueError(f"Duplicate event id '{eid}' in {events_path}")
        if "s" not in it or "e" not in it:
            raise ValueError(f"Event[{idx}] id={eid} missing normalized s/e in {events_path}")
        s = float(it["s"])
        e = float(it["e"])
        spans[eid] = (s, e)
    return spans


def _validate_span_map(spans: Dict[str, Tuple[float, float]], path: str) -> None:
    errs: List[str] = []
    tol = 1e-6
    for eid, (s, e) in spans.items():
        if not np.isfinite(s) or not np.isfinite(e):
            errs.append(f"id={eid}: non-finite s/e ({s}, {e})")
            continue
        if s < -tol or e > 1.0 + tol or e < s - tol:
            errs.append(f"id={eid}: invalid normalized interval s={s}, e={e}")
    if errs:
        joined = "\n  - ".join(errs[:20])
        more = "" if len(errs) <= 20 else f"\n  ... and {len(errs) - 20} more"
        raise ValueError(f"Invalid merged event evidence in {path}:\n  - {joined}{more}")




def _eq(x: float, y: float, eps: float) -> bool:
    return abs(x - y) <= eps


def _lt(x: float, y: float, eps: float) -> bool:
    return x < (y - eps)


def _gt(x: float, y: float, eps: float) -> bool:
    return x > (y + eps)


_ALLEN_LABELS: List[str] = ["E", "S", "Si", "F", "Fi", "D", "Di", "M", "Mi", "B", "Bi", "O", "Oi"]
_L2I: Dict[str, int] = {r: i for i, r in enumerate(_ALLEN_LABELS)}


def _allen_relation(a: Tuple[float, float], b: Tuple[float, float], eps: float) -> str:
    sa, ea = a
    sb, eb = b
    
    if _eq(sa, sb, eps) and _eq(ea, eb, eps):
        return "E"
    
    if _eq(sa, sb, eps) and _lt(ea, eb, eps):
        return "S"
    if _eq(sa, sb, eps) and _gt(ea, eb, eps):
        return "Si"
    
    if _eq(ea, eb, eps) and _gt(sa, sb, eps):
        return "F"
    if _eq(ea, eb, eps) and _lt(sa, sb, eps):
        return "Fi"
    
    if _lt(sb, sa, eps) and _lt(ea, eb, eps):
        return "D"
    if _lt(sa, sb, eps) and _lt(eb, ea, eps):
        return "Di"
    
    if _eq(ea, sb, eps):
        return "M"
    if _eq(eb, sa, eps):
        return "Mi"
    
    if _lt(ea, sb, eps):
        return "B"
    if _gt(sa, eb, eps):
        return "Bi"
    
    if _lt(sa, sb, eps) and _lt(sb, ea, eps) and _lt(ea, eb, eps):
        return "O"
    if _lt(sb, sa, eps) and _lt(sa, eb, eps) and _lt(eb, ea, eps):
        return "Oi"
    raise ValueError(
        f"Allen relation is undefined for intervals a=({sa}, {ea}), b=({sb}, {eb}) with eps={eps}."
    )




def _affinity_from_cfg(cfg: Dict[str, Any]) -> np.ndarray:
    ERel_cfg = (cfg.get("ERel") or {})
    aff_cfg = ERel_cfg.get("affinity", None)

    A = np.zeros((13, 13), dtype=float)
    np.fill_diagonal(A, 1.0)  

    if aff_cfg in (None, False, 0):
        return A  

    if isinstance(aff_cfg, str) and aff_cfg.lower() == "default_v1":
        def setw(a, b, w):
            ia, ib = _L2I.get(a), _L2I.get(b)
            if ia is None or ib is None:
                return
            A[ia, ib] = max(A[ia, ib], w)
            A[ib, ia] = max(A[ib, ia], w)

        for lab in ["S", "Si", "F", "Fi"]:
            setw("E", lab, 0.70)

        setw("M", "B", 0.80)
        setw("Mi", "Bi", 0.80)

        setw("S", "D", 0.75)
        setw("Si", "Di", 0.75)
        setw("F", "D", 0.75)
        setw("Fi", "Di", 0.75)

        setw("O", "D", 0.60)
        setw("Oi", "Di", 0.60)

        return A

    
    if isinstance(aff_cfg, dict):
        pairs = aff_cfg.get("pairs", [])
        for it in pairs:
            if isinstance(it, (list, tuple)) and len(it) == 3:
                a, b, w = str(it[0]), str(it[1]), float(it[2])
                ia, ib = _L2I.get(a), _L2I.get(b)
                if ia is None or ib is None:
                    continue
                w = float(np.clip(w, 0.0, 1.0))
                A[ia, ib] = max(A[ia, ib], w)
                A[ib, ia] = max(A[ib, ia], w)
        return A

    
    return A


def _affinity_score(A: np.ndarray, r1: str, r2: str) -> float:
    i, j = _L2I.get(r1), _L2I.get(r2)
    if i is None or j is None:
        return 0.0
    return float(A[i, j])


def _portable_path(path: str) -> str:
    p = Path(path)
    parts = p.parts
    for anchor in ("outputs", "data", "ref4d_eval", "docs", "checkpoints", "third_party"):
        if anchor in parts:
            idx = parts.index(anchor)
            return Path(*parts[idx:]).as_posix()
    return p.name




def compute_ERel(pairs_json_path: str, ref_events_json: str, gen_events_json: str, cfg_path: str) -> Dict[str, Any]:
    pairs = read_json(pairs_json_path)
    M = pairs.get("M", [])
    if not isinstance(M, list) or len(M) < 2:
        
        return {"score": 0.0, "valid": False, "details": {"n_pairs": len(M), "omitted": True}}

    cfg = read_yaml(cfg_path)
    eps = float(((cfg.get("ERel") or {}).get("allen_eps") or {}).get("eq", 1e-3))
    A = _affinity_from_cfg(cfg)  

    ref_spans = _load_span_map(ref_events_json)
    gen_spans = _load_span_map(gen_events_json)
    _validate_span_map(ref_spans, ref_events_json)
    _validate_span_map(gen_spans, gen_events_json)

    
    def _start_or_raise(eid: str) -> float:
        if eid not in ref_spans:
            raise KeyError(f"Matched reference event id '{eid}' missing from {ref_events_json}")
        return ref_spans[eid][0]

    M_sorted = sorted(M, key=lambda x: _start_or_raise(str(x[0])))

    ref_ids = [str(x[0]) for x in M_sorted]
    gen_ids = [str(x[1]) for x in M_sorted]
    Nr = len(M_sorted)

    
    total = 0           
    exact = 0           
    aff_sum = 0.0       

    for a in range(Nr):
        for b in range(a + 1, Nr):
            ri, rj = ref_ids[a], ref_ids[b]
            gi, gj = gen_ids[a], gen_ids[b]
            if ri not in ref_spans:
                raise KeyError(f"Matched reference event id '{ri}' missing from {ref_events_json}")
            if rj not in ref_spans:
                raise KeyError(f"Matched reference event id '{rj}' missing from {ref_events_json}")
            if gi not in gen_spans:
                raise KeyError(f"Matched generated event id '{gi}' missing from {gen_events_json}")
            if gj not in gen_spans:
                raise KeyError(f"Matched generated event id '{gj}' missing from {gen_events_json}")
            R_ref = _allen_relation(ref_spans[ri], ref_spans[rj], eps)
            R_gen = _allen_relation(gen_spans[gi], gen_spans[gj], eps)
            total += 1
            if R_ref == R_gen:
                exact += 1
            aff_sum += _affinity_score(A, R_ref, R_gen)

    if total == 0:
        return {
            "score": 0.0,
            "valid": False,
            "details": {
                "n_pairs": Nr,
                "checked": 0,
                "eps": eps,
                "omitted": True,
                "ref_events": _portable_path(ref_events_json),
                "gen_events": _portable_path(gen_events_json),
                "affinity": "enabled" if (A.sum() > 13.0) else "strict_identity",
            }
        }

    
    S = aff_sum / float(total)
    return {
        "score": float(S),
        "valid": True,
        "details": {
            "n_pairs": Nr,
            "checked": total,
            "exact_equal": exact,
            "eps": eps,
            "ref_events": _portable_path(ref_events_json),
            "gen_events": _portable_path(gen_events_json),
            "affinity": "enabled" if (A.sum() > 13.0) else "strict_identity",
        }
    }


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute ERel score from merged event evidence with valid normalized s/e (Allen 13 relations)."
    )
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument("--ref_events", type=str, required=True)
    ap.add_argument("--gen_events", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = compute_ERel(args.pairs, args.ref_events, args.gen_events, args.config)
    print(json.dumps(out, ensure_ascii=False, indent=2))
