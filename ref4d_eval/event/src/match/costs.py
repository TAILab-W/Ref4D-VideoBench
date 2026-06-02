

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

from ..common.io import read_yaml, ensure_dir

BIG_COST = 1e6


def _portable_path(path: str) -> str:
    p = Path(path)
    parts = p.parts
    for anchor in ("outputs", "data", "ref4d_eval", "docs", "checkpoints", "third_party"):
        if anchor in parts:
            idx = parts.index(anchor)
            return Path(*parts[idx:]).as_posix()
    return p.name


def _load_gate(gate_npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(gate_npz_path, allow_pickle=True)
    sim = d["sim_sem"].astype(np.float64)
    rt = d["r_tiou"].astype(np.float64)
    gate = d["gate"].astype(bool)
    ref_ids = d["ref_ids"]
    gen_ids = d["gen_ids"]
    return sim, rt, gate, ref_ids, gen_ids


def _weights_from_cfg(cfg_path: str) -> Dict[str, float]:
    cfg = read_yaml(cfg_path)
    ega = (cfg.get("ega") or {}) if isinstance(cfg, dict) else {}
    matching = (cfg.get("matching") or {}) if isinstance(cfg, dict) else {}
    w1 = float(ega.get("w1", 0.8))
    w2 = float(ega.get("w2", 0.2))
    null_ref = float(matching.get("null_ref", 1.001))
    null_gen = float(matching.get("null_gen", 1.001))
    if not np.isfinite(w1) or not np.isfinite(w2):
        raise ValueError(f"ega.w1 / ega.w2 must be finite, got w1={w1}, w2={w2}")
    if w1 < 0 or w2 < 0:
        raise ValueError(f"ega.w1 / ega.w2 must be non-negative, got w1={w1}, w2={w2}")
    if abs((w1 + w2) - 1.0) > 1e-6:
        raise ValueError(f"ega.w1 + ega.w2 must equal 1, got w1={w1}, w2={w2}, sum={w1 + w2}")
    if not np.isfinite(null_ref) or not np.isfinite(null_gen):
        raise ValueError(f"matching.null_ref / matching.null_gen must be finite, got null_ref={null_ref}, null_gen={null_gen}")
    if null_ref < 0 or null_gen < 0:
        raise ValueError(f"matching.null_ref / matching.null_gen must be non-negative, got null_ref={null_ref}, null_gen={null_gen}")
    return {"w1": w1, "w2": w2, "null_ref": null_ref, "null_gen": null_gen}


def _build_square_cost(sim: np.ndarray, rt: np.ndarray, gate: np.ndarray, w1: float, w2: float, null_ref: float, null_gen: float) -> Tuple[np.ndarray, int, int]:
    Nr, Ng = sim.shape
    base = 1.0 - (w1 * sim + w2 * rt)
    cost = np.where(gate, base, BIG_COST)
    N = Nr + Ng
    C = np.full((N, N), BIG_COST, dtype=np.float64)
    C[:Nr, :Ng] = cost
    for i in range(Nr):
        C[i, Ng + i] = null_ref
    for j in range(Ng):
        C[Nr + j, j] = null_gen
    C[Nr:, Ng:] = 0.0
    return C, Nr, Ng


def build_and_save(gate_npz_path: str, cfg_path: str, out_npz_path: str) -> Dict[str, Any]:
    sim, rt, gate, ref_ids, gen_ids = _load_gate(gate_npz_path)
    W = _weights_from_cfg(cfg_path)
    C, Nr, Ng = _build_square_cost(sim, rt, gate, W["w1"], W["w2"], W["null_ref"], W["null_gen"])
    meta = {
        "Nr": int(Nr), "Ng": int(Ng), "Npad": int(C.shape[0]),
        "w1": W["w1"], "w2": W["w2"],
        "null_ref": W["null_ref"], "null_gen": W["null_gen"],
        "sources": {"gate_npz": _portable_path(gate_npz_path), "cfg": _portable_path(cfg_path)}
    }
    ensure_dir(Path(out_npz_path).parent)
    np.savez_compressed(
        out_npz_path,
        C=C, Nr=np.int32(Nr), Ng=np.int32(Ng),
        w1=np.float32(W["w1"]), w2=np.float32(W["w2"]),
        null_ref=np.float32(W["null_ref"]), null_gen=np.float32(W["null_gen"]), big=np.float32(BIG_COST),
        ref_ids=ref_ids, gen_ids=gen_ids,
        meta=json.dumps(meta, ensure_ascii=False)
    )
    print(f"[costs] Wrote cost matrix: {out_npz_path} (shape={C.shape}, Nr={Nr}, Ng={Ng}, w1={W['w1']}, w2={W['w2']}, null_ref={W['null_ref']}, null_gen={W['null_gen']})")
    return {"nr": int(Nr), "ng": int(Ng), "out": str(out_npz_path), "w1": W["w1"], "w2": W["w2"], "null_ref": W["null_ref"], "null_gen": W["null_gen"]}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build the Hungarian cost matrix from outputs/event/cache/match/<pair_id>/gate_masks.npz."
    )
    ap.add_argument(
        "--gate",
        type=str,
        required=True,
        help="path to outputs/event/cache/match/<pair_id>/gate_masks.npz",
    )
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to the event default config (for ega.w1 / ega.w2)",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="path to outputs/event/cache/match/<pair_id>/cost_matrix.npz",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_and_save(args.gate, args.config, args.out)
