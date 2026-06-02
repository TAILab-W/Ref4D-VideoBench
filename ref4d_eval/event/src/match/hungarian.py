from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


try:
    from scipy.optimize import linear_sum_assignment  
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _hungarian_numpy(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    C = C.copy().astype(float)
    n = C.shape[0]
    C -= C.min(axis=1, keepdims=True)
    C -= C.min(axis=0, keepdims=True)

    STAR = -1
    PRIME = -2
    marks = np.zeros_like(C, dtype=int)
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(n, dtype=bool)

    
    for i in range(n):
        js = np.where((C[i] == 0) & (~col_covered))[0]
        if js.size > 0:
            marks[i, js[0]] = STAR
            col_covered[js[0]] = True
    col_covered[:] = False

    def cover_star_cols():
        col_covered[:] = np.any(marks == STAR, axis=0)

    def find_zero():
        for i in range(n):
            if row_covered[i]:
                continue
            for j in range(n):
                if not col_covered[j] and C[i, j] == 0:
                    return i, j
        return None, None

    def star_in_row(r):
        js = np.where(marks[r] == STAR)[0]
        return js[0] if js.size > 0 else None

    def star_in_col(c):
        is_ = np.where(marks[:, c] == STAR)[0]
        return is_[0] if is_.size > 0 else None

    def prime_in_row(r):
        js = np.where(marks[r] == PRIME)[0]
        return js[0] if js.size > 0 else None

    def augment(path):
        for (r, c) in path:
            if marks[r, c] == STAR:
                marks[r, c] = 0
            else:
                marks[r, c] = STAR

    def clear_primes():
        marks[marks == PRIME] = 0

    cover_star_cols()
    while col_covered.sum() < n:
        while True:
            r, c = find_zero()
            if r is None:
                ur = ~row_covered
                uc = ~col_covered
                m = np.min(C[np.ix_(ur, uc)])
                C[ur, :] -= m
                C[:, col_covered] += m
            else:
                marks[r, c] = PRIME
                s = star_in_row(r)
                if s is None:
                    path = [(r, c)]
                    cc = c
                    rr = star_in_col(cc)
                    while rr is not None:
                        path.append((rr, cc))
                        cc = prime_in_row(rr)
                        path.append((rr, cc))
                        rr = star_in_col(cc)
                    augment(path)
                    clear_primes()
                    row_covered[:] = False
                    col_covered[:] = False
                    cover_star_cols()
                    break
                else:
                    row_covered[r] = True
                    col_covered[s] = False

    row_ind = np.arange(n)
    col_ind = np.zeros(n, dtype=int)
    for i in range(n):
        j = np.where(marks[i] == STAR)[0]
        col_ind[i] = j[0]
    return row_ind, col_ind


def _portable_path(path: str) -> str:
    p = Path(path)
    parts = list(p.parts)
    for anchor in ("outputs", "data", "checkpoints", "third_party", "docs", "scripts", "envs"):
        if anchor in parts:
            idx = parts.index(anchor)
            return Path(*parts[idx:]).as_posix()
    return p.name or str(path)


def _validate_weights(w1: float, w2: float, tol: float = 1e-6) -> Tuple[float, float]:
    if not (np.isfinite(w1) and np.isfinite(w2)):
        raise ValueError(f"Invalid matching weights: w1={w1}, w2={w2} must be finite.")
    if w1 < 0 or w2 < 0:
        raise ValueError(f"Invalid matching weights: w1={w1}, w2={w2} must be non-negative.")
    if abs((w1 + w2) - 1.0) > tol:
        raise ValueError(f"Invalid matching weights: require w1 + w2 = 1, got w1={w1}, w2={w2}.")
    return float(w1), float(w2)


def _read_weights_from_cost_npz(Cdat) -> Tuple[float, float]:
    if "w1" in Cdat and "w2" in Cdat:
        try:
            return _validate_weights(float(Cdat["w1"]), float(Cdat["w2"]))
        except Exception as exc:
            raise ValueError("Invalid w1/w2 stored in cost NPZ direct keys.") from exc

    if "meta" in Cdat:
        try:
            meta_raw = Cdat["meta"]
            if isinstance(meta_raw, np.ndarray):
                meta_raw = meta_raw.item()
            meta = json.loads(str(meta_raw))
            if "w1" in meta and "w2" in meta:
                return _validate_weights(float(meta["w1"]), float(meta["w2"]))
        except Exception as exc:
            raise ValueError("Failed to parse strict matching weights from cost NPZ metadata.") from exc

    raise ValueError("Missing strict matching weights in cost NPZ: expected direct keys w1/w2 or metadata fields.")


def _validate_cost_gate_contract(
    C: np.ndarray,
    Nr: int,
    Ng: int,
    sim: np.ndarray,
    rt: np.ndarray,
    gate: np.ndarray,
    ref_ids: list[str],
    gen_ids: list[str],
    Cdat,
) -> None:
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Cost matrix must be square, got shape={C.shape}.")
    if C.shape[0] != Nr + Ng:
        raise ValueError(f"Cost matrix must use Nr+Ng augmentation, got Nr={Nr}, Ng={Ng}, shape={C.shape}.")
    if sim.shape != rt.shape or sim.shape != gate.shape:
        raise ValueError(
            f"Gate tensors must share the same shape, got sim={sim.shape}, rt={rt.shape}, gate={gate.shape}."
        )
    if sim.ndim != 2:
        raise ValueError(f"Gate tensors must be 2D, got sim.ndim={sim.ndim}.")
    if sim.shape[0] < Nr or sim.shape[1] < Ng:
        raise ValueError(
            f"Gate tensor shape {sim.shape} is smaller than required Nr={Nr}, Ng={Ng}."
        )
    if len(ref_ids) != sim.shape[0]:
        raise ValueError(f"ref_ids length {len(ref_ids)} != sim rows {sim.shape[0]}.")
    if len(gen_ids) != sim.shape[1]:
        raise ValueError(f"gen_ids length {len(gen_ids)} != sim cols {sim.shape[1]}.")

    if "ref_ids" in Cdat:
        cost_ref_ids = [str(x) for x in Cdat["ref_ids"].tolist()]
        if cost_ref_ids != ref_ids:
            raise ValueError("ref_ids mismatch between cost NPZ and gate NPZ.")
    if "gen_ids" in Cdat:
        cost_gen_ids = [str(x) for x in Cdat["gen_ids"].tolist()]
        if cost_gen_ids != gen_ids:
            raise ValueError("gen_ids mismatch between cost NPZ and gate NPZ.")


def solve_and_save(cost_npz: str, gate_npz: str, out_json: str):
    Cdat = np.load(cost_npz, allow_pickle=True)
    Gdat = np.load(gate_npz, allow_pickle=True)

    C = Cdat["C"].astype(float)
    Nr = int(Cdat["Nr"])
    Ng = int(Cdat["Ng"])
    null_ref = float(Cdat["null_ref"]) if "null_ref" in Cdat else None
    null_gen = float(Cdat["null_gen"]) if "null_gen" in Cdat else None

    w1, w2 = _read_weights_from_cost_npz(Cdat)

    ref_ids = [str(x) for x in Gdat["ref_ids"].tolist()]
    gen_ids = [str(x) for x in Gdat["gen_ids"].tolist()]
    sim = Gdat["sim_sem"].astype(float)    
    rt = Gdat["r_tiou"].astype(float)      
    gate = Gdat["gate"].astype(bool)       

    _validate_cost_gate_contract(C, Nr, Ng, sim, rt, gate, ref_ids, gen_ids, Cdat)

    if _HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(C)
    else:
        row_ind, col_ind = _hungarian_numpy(C)

    M = []
    for r, c in zip(row_ind, col_ind):
        if r < Nr and c < Ng and gate[r, c]:
            s = float(sim[r, c])
            u = float(rt[r, c])
            q = float(w1 * s + w2 * u)
            M.append([ref_ids[r], gen_ids[c], {"sim_sem": s, "r_tIoU": u, "q": q}])

    meta = {
        "Nr": Nr,
        "Ng": Ng,
        "Npad": int(C.shape[0]),
        "w1": float(w1),
        "w2": float(w2),
        "null_ref": null_ref,
        "null_gen": null_gen,
        "sources": {
            "cost_npz": _portable_path(cost_npz),
            "gate_npz": _portable_path(gate_npz),
        },
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"M": M, "meta": meta}, f, ensure_ascii=False, indent=2)
    print(f"[hungarian] wrote: {out_json} (|M|={len(M)}, w1={w1}, w2={w2})")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Solve one-to-one event matching from cost_matrix.npz and gate_masks.npz."
    )
    ap.add_argument(
        "--cost",
        required=True,
        help="outputs/event/cache/match/<pair_id>/cost_matrix.npz",
    )
    ap.add_argument(
        "--gate",
        required=True,
        help="outputs/event/cache/match/<pair_id>/gate_masks.npz",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="outputs/event/cache/match/<pair_id>/pairs.json",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    solve_and_save(args.cost, args.gate, args.out)
