
"""
EGA: Event-Graph Alignment (local quality on matched pairs)

Inputs:
  - pairs.json: outputs/event/cache/match/<pair_id>/pairs.json
      {
        "M": [
          ["<ref_id>", "<gen_id>", {"sim_sem": float, "r_tIoU": float, "q": float}],
          ...
        ],
        "meta": {..., "w1":..., "w2":...}
      }
  - ref_events_json: canonical merged reference event evidence
      (must provide valid normalized s/e in events_merged/*.newevents.json)


Returns:
  { "score": float in [0,1], "valid": bool, "details": {...} }
"""

from __future__ import annotations
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..common.io import read_json, read_yaml

LOGGER = logging.getLogger("event_eval.metrics.ega")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


def _portable_path(p: str) -> str:
    try:
        path = Path(str(p))
        parts = path.parts
        for anchor in ("outputs", "data", "ref4d_eval", "docs", "checkpoints", "third_party"):
            if anchor in parts:
                idx = parts.index(anchor)
                return Path(*parts[idx:]).as_posix()
        return path.name
    except Exception:
        return str(p)


def _load_ref_span_map(ref_events_json: str) -> Dict[str, Tuple[float, float]]:
    data = read_json(ref_events_json)
    if isinstance(data, dict) and "events" in data:
        data = data["events"]
    d: Dict[str, Tuple[float, float]] = {}
    for idx, it in enumerate(data):
        rid_raw = it.get("id") or it.get("eid") or it.get("event_id")
        rid = str(rid_raw) if rid_raw is not None else ""
        if not rid:
            raise ValueError(f"Event[{idx}] missing id in reference events: {ref_events_json}")
        if rid in d:
            raise ValueError(f"Duplicate event id '{rid}' in reference events: {ref_events_json}")
        if "s" not in it or "e" not in it:
            raise ValueError(f"Event[{idx}] missing normalized s/e for id={rid} in {ref_events_json}")
        s = float(it["s"])
        e = float(it["e"])
        d[rid] = (s, e)
    return d


def _validate_span_map(span_map: Dict[str, Tuple[float, float]], events_path: str) -> None:
    tol = 1e-6
    for eid, (s, e) in span_map.items():
        if not (np.isfinite(s) and np.isfinite(e)):
            raise ValueError(f"Non-finite normalized span for id={eid} in {events_path}: s={s}, e={e}")
        if s < -tol or e > 1.0 + tol or e < s - tol:
            raise ValueError(f"Invalid normalized span for id={eid} in {events_path}: s={s}, e={e}")


def _weights_from_pairs_meta(cfg: Dict[str, Any], pairs_meta: Dict[str, Any]) -> Tuple[float, float, float]:
    if not isinstance(pairs_meta, dict):
        raise ValueError("pairs.json meta must be a dict and contain w1/w2.")
    if "w1" not in pairs_meta or "w2" not in pairs_meta:
        raise ValueError("pairs.json meta missing w1/w2; cannot compute EGA consistently with matching.")
    w1 = float(pairs_meta["w1"])
    w2 = float(pairs_meta["w2"])
    if not (np.isfinite(w1) and np.isfinite(w2)):
        raise ValueError(f"Non-finite w1/w2 in pairs meta: w1={w1}, w2={w2}")
    if w1 < 0 or w2 < 0:
        raise ValueError(f"pairs meta w1/w2 must be non-negative, got w1={w1}, w2={w2}")
    if abs((w1 + w2) - 1.0) > 1e-6:
        raise ValueError(f"pairs meta requires w1+w2=1, got w1={w1}, w2={w2}")

    ega = cfg.get("ega") or {}
    if "w1" in ega or "w2" in ega:
        try:
            cfg_w1 = float(ega.get("w1", w1))
            cfg_w2 = float(ega.get("w2", w2))
            if abs(cfg_w1 - w1) > 1e-6 or abs(cfg_w2 - w2) > 1e-6:
                raise ValueError(
                    f"ega.w1/w2 in config disagree with pairs meta: cfg=({cfg_w1},{cfg_w2}) vs pairs=({w1},{w2})"
                )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Invalid ega.w1/w2 in config.") from exc

    rho = float(ega.get("rho", 0.5))
    if not np.isfinite(rho) or rho < 0:
        raise ValueError(f"ega.rho must be finite and >= 0, got rho={rho}")
    return w1, w2, rho


def compute_ega(pairs_json_path: str, ref_events_json: str, cfg_path: str) -> Dict[str, Any]:
    pairs = read_json(pairs_json_path)
    M = pairs.get("M", [])
    meta = pairs.get("meta", {})
    if not isinstance(M, list):
        raise ValueError(f"{pairs_json_path}: key 'M' must be a list")

    if len(M) == 0:
        return {
            "score": 0.0,
            "valid": True,
            "details": {
                "n_pairs": 0,
                "used_pairs": 0,
                "omitted": False,
                "reason": "no_matched_pairs",
                "pairs": _portable_path(pairs_json_path),
                "ref_events": _portable_path(ref_events_json),
            },
        }

    cfg = read_yaml(cfg_path)
    w1, w2, rho = _weights_from_pairs_meta(cfg, meta)

    span_map = _load_ref_span_map(ref_events_json)
    _validate_span_map(span_map, ref_events_json)
    weights: List[float] = []
    scores: List[float] = []

    for trip in M:
        ref_id, gen_id, d = trip
        if ref_id not in span_map:
            raise ValueError(f"Matched ref_id={ref_id} not found in reference events: {ref_events_json}")
        s0, e0 = span_map[ref_id]
        dur = max(0.0, e0 - s0)
        w = dur ** rho
        if w <= 0:
            continue

        if isinstance(d, dict) and ("q" in d):
            q = float(d["q"])
        else:
            s = float(d.get("sim_sem", 0.0))
            u = float(d.get("r_tIoU", 0.0))
            q = w1 * s + w2 * u
        if not np.isfinite(q) or q < -1e-6 or q > 1.0 + 1e-6:
            raise ValueError(f"Invalid q for matched pair ({ref_id}, {gen_id}): q={q}")
        q = float(min(1.0, max(0.0, q)))

        weights.append(w)
        scores.append(q)

    if not weights:
        return {"score": 0.0, "valid": True, "details": {"n_pairs": len(M), "used_pairs": 0}}

    weights_np = np.asarray(weights, dtype=np.float64)
    scores_np = np.asarray(scores, dtype=np.float64)
    S = float((weights_np * scores_np).sum() / max(1e-12, weights_np.sum()))
    return {
        "score": S,
        "valid": True,
        "details": {
            "n_pairs": len(M),
            "used_pairs": int(len(weights)),
            "w1": w1,
            "w2": w2,
            "rho": rho,
            "ref_events": _portable_path(ref_events_json),
            "pairs": _portable_path(pairs_json_path),
        }
    }


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute EGA from pairs.json and canonical merged reference event evidence."
    )
    ap.add_argument("--pairs", type=str, required=True, help="outputs/event/cache/match/<pair_id>/pairs.json")
    ap.add_argument(
        "--ref_events",
        type=str,
        required=True,
        help="canonical merged reference event evidence with valid normalized s/e",
    )
    ap.add_argument("--config", type=str, required=True, help="event config yaml (rho and optional consistency check)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out = compute_ega(args.pairs, args.ref_events, args.config)
    print(json.dumps(out, ensure_ascii=False, indent=2))