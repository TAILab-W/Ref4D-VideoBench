
"""
ECR: Event Coverage & Redundancy

Inputs:
  - ref_events_json: canonical merged reference event evidence,
      typically outputs/event/cache/events_merged/ref/<sample_id>.newevents.json
  - gen_events_json: canonical merged generated event evidence,
      typically outputs/event/cache/events_merged/gen/<pair_id>.newevents.json
  - pairs.json: one-to-one event matching result,
      typically outputs/event/cache/match/<pair_id>/pairs.json
"""
from __future__ import annotations
import json
import argparse
import logging
import math
from typing import Any, Dict, Set

from ..common.io import read_json

LOGGER = logging.getLogger("event_eval.metrics.ecr")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


def _validate_norm_event_node(it: Any, path: str, idx: int) -> str:
    if not isinstance(it, dict):
        raise ValueError(f"{path}: event[{idx}] must be a dict, got {type(it).__name__}")
    eid = it.get("id") or it.get("eid") or it.get("event_id")
    if not isinstance(eid, str) or not eid.strip():
        raise ValueError(f"{path}: event[{idx}] missing non-empty string id")
    if "s" not in it or "e" not in it:
        raise ValueError(f"{path}: event[{idx}] missing normalized s/e")
    try:
        s = float(it["s"])
        e = float(it["e"])
    except Exception as exc:
        raise ValueError(f"{path}: event[{idx}] has non-numeric normalized s/e") from exc
    if not (math.isfinite(s) and math.isfinite(e)):
        raise ValueError(f"{path}: event[{idx}] has non-finite normalized s/e")
    tol = 1e-6
    if s < -tol or e < -tol or s > 1.0 + tol or e > 1.0 + tol or e < s - tol:
        raise ValueError(f"{path}: event[{idx}] has invalid normalized interval s={s}, e={e}")
    return eid.strip()



def _event_id_set(path: str) -> Set[str]:
    data = read_json(path)
    if isinstance(data, dict) and "events" in data:
        data = data["events"]
    elif isinstance(data, list):
        pass
    else:
        raise ValueError(f"{path}: expected a list or dict with key 'events'")
    if not isinstance(data, list):
        raise ValueError(f"{path}: 'events' must be a list")
    seen: Set[str] = set()
    for idx, it in enumerate(data):
        eid = _validate_norm_event_node(it, path, idx)
        if eid in seen:
            raise ValueError(f"{path}: duplicate event id '{eid}'")
        seen.add(eid)
    return seen



def _count_event_nodes(path: str) -> int:
    return len(_event_id_set(path))



def _matched_event_sets(pairs_json_path: str) -> tuple[Set[str], Set[str], int]:
    pairs = read_json(pairs_json_path)
    if not isinstance(pairs, dict):
        raise ValueError(f"{pairs_json_path}: pairs json must be a dict")
    M = pairs.get("M", None)
    if not isinstance(M, list):
        raise ValueError(f"{pairs_json_path}: key 'M' must be a list")

    matched_ref: Set[str] = set()
    matched_gen: Set[str] = set()
    m = 0
    for idx, trip in enumerate(M):
        if not isinstance(trip, (list, tuple)) or len(trip) < 2:
            raise ValueError(f"{pairs_json_path}: M[{idx}] must be a list/tuple with at least 2 items")
        ref_id = trip[0]
        gen_id = trip[1]
        if not isinstance(ref_id, str) or not ref_id.strip():
            raise ValueError(f"{pairs_json_path}: M[{idx}] has invalid ref_id")
        if not isinstance(gen_id, str) or not gen_id.strip():
            raise ValueError(f"{pairs_json_path}: M[{idx}] has invalid gen_id")
        ref_id = ref_id.strip()
        gen_id = gen_id.strip()
        if ref_id in matched_ref:
            raise ValueError(f"{pairs_json_path}: duplicate matched ref_id '{ref_id}' violates one-to-one constraint")
        if gen_id in matched_gen:
            raise ValueError(f"{pairs_json_path}: duplicate matched gen_id '{gen_id}' violates one-to-one constraint")
        matched_ref.add(ref_id)
        matched_gen.add(gen_id)
        m += 1
    return matched_ref, matched_gen, m



def compute_ecr(ref_events_json: str, gen_events_json: str, pairs_json_path: str, eps: float = 1e-12) -> Dict[str, Any]:
    ref_event_ids = _event_id_set(ref_events_json)
    gen_event_ids = _event_id_set(gen_events_json)
    n_ref = len(ref_event_ids)
    n_gen = len(gen_event_ids)
    if n_ref == 0 or n_gen == 0:
        return {
            "score": 0.0,
            "valid": False,
            "details": {
                "n_ref": n_ref,
                "n_gen": n_gen,
                "m": 0,
                "matched_ref": 0,
                "matched_gen": 0,
                "unmatched_gen": 0,
                "C_ref": 0.0,
                "H_gen": 0.0,
                "kept_gen": 0.0,
                "eps": float(eps),
            },
        }

    matched_ref_ids, matched_gen_ids, m = _matched_event_sets(pairs_json_path)

    extra_ref = matched_ref_ids - ref_event_ids
    extra_gen = matched_gen_ids - gen_event_ids
    if extra_ref:
        bad = ", ".join(sorted(extra_ref))
        raise ValueError(f"{pairs_json_path}: matched ref ids not found in {ref_events_json}: {bad}")
    if extra_gen:
        bad = ", ".join(sorted(extra_gen))
        raise ValueError(f"{pairs_json_path}: matched gen ids not found in {gen_events_json}: {bad}")

    matched_ref = len(matched_ref_ids)
    matched_gen = len(matched_gen_ids)
    unmatched_gen = max(0, n_gen - matched_gen)

    c_ref = matched_ref / n_ref
    h_gen = unmatched_gen / n_gen
    kept_gen = 1.0 - h_gen
    score = (2.0 * c_ref * kept_gen) / (c_ref + kept_gen + float(eps))

    return {
        "score": score,
        "valid": True,
        "details": {
            "n_ref": n_ref,
            "n_gen": n_gen,
            "m": m,
            "matched_ref": matched_ref,
            "matched_gen": matched_gen,
            "unmatched_gen": unmatched_gen,
            "C_ref": c_ref,
            "H_gen": h_gen,
            "kept_gen": kept_gen,
            "eps": float(eps),
        }
    }



def parse_args():
    ap = argparse.ArgumentParser(description="Compute ECR score from canonical merged event evidence and pairs.json")
    ap.add_argument("--ref_events", type=str, required=True, help="Canonical merged reference event evidence JSON")
    ap.add_argument("--gen_events", type=str, required=True, help="Canonical merged generated event evidence JSON")
    ap.add_argument("--pairs", type=str, required=True, help="One-to-one event matching result JSON")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = compute_ecr(args.ref_events, args.gen_events, args.pairs)
    print(json.dumps(out, ensure_ascii=False, indent=2))
