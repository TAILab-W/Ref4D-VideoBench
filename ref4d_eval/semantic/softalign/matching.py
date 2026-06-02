
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import Config
from .types import MatchPair, MatchResult

__all__ = [
    "hungarian_match",
    "greedy_match",
    "compute_matching",
]

def _min_match_score(cfg: Config) -> float:
    m = getattr(cfg, "matching", None)
    return float(getattr(m, "min_score", 0.30))

def _unmatch_cost_from_gate(gate: float) -> float:
    gate = float(max(0.0, min(0.999999, gate)))
    return 1.0 - gate

def _build_padded_cost(sim: np.ndarray, gate: float) -> Tuple[np.ndarray, int, int]:
    R, G = sim.shape
    N = R + G
    if N == 0:
        return np.zeros((0, 0), dtype=np.float32), R, G

    M = np.full((N, N), fill_value=1e6, dtype=np.float32)

    if R > 0 and G > 0:
        M[:R, :G] = (1.0 - sim).astype(np.float32, copy=False)

    c_unmatch = _unmatch_cost_from_gate(gate)

    for i in range(R):
        M[i, G + i] = c_unmatch

    for j in range(G):
        M[R + j, j] = c_unmatch

    if R > 0 and G > 0:
        M[R:, G:] = 0.0

    return M, R, G

def _hungarian_min_cost(cost: np.ndarray):
    try:
        from scipy.optimize import linear_sum_assignment  
    except Exception as e:
        raise ImportError("scipy.optimize.linear_sum_assignment is required to use Hungarian matching") from e
    row_ind, col_ind = linear_sum_assignment(cost)
    total = float(cost[row_ind, col_ind].sum())
    return row_ind, col_ind, total

def _greedy_maximum(sim: np.ndarray, gate: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    R, G = sim.shape
    used_r = np.zeros(R, dtype=bool)
    used_g = np.zeros(G, dtype=bool)
    pairs: List[Tuple[int, int]] = []
    while True:
        best = gate
        bi = -1
        bj = -1
        for i in range(R):
            if used_r[i]:
                continue
            for j in range(G):
                if used_g[j]:
                    continue
                s = sim[i, j]
                if s > best:
                    best = s
                    bi, bj = i, j
        if bi < 0 or bj < 0:
            break
        pairs.append((bi, bj))
        used_r[bi] = True
        used_g[bj] = True
    ref_un = [i for i in range(R) if not used_r[i]]
    gen_un = [j for j in range(G) if not used_g[j]]
    return pairs, ref_un, gen_un

def hungarian_match(sim: np.ndarray, gate: float) -> MatchResult:
    sim = np.asarray(sim, dtype=np.float32)
    R, G = sim.shape
    if R == 0 and G == 0:
        return MatchResult(pairs=[], method="hungarian")

    C, R0, G0 = _build_padded_cost(sim, gate)

    rows, cols, _ = _hungarian_min_cost(C)

    pairs: List[MatchPair] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r < R0 and c < G0:
            s = float(sim[r, c])
            if s > gate:
                pairs.append(MatchPair(r_idx=r, g_idx=c, score=s))

    return MatchResult(
        pairs=sorted(pairs, key=lambda x: (-x.score, x.r_idx, x.g_idx)),
        method="hungarian",
    )

def greedy_match(sim: np.ndarray, gate: float) -> MatchResult:
    pairs_greedy, _, _ = _greedy_maximum(np.asarray(sim, dtype=np.float32), gate)
    pairs = [MatchPair(r_idx=i, g_idx=j, score=float(sim[i, j])) for (i, j) in pairs_greedy]
    return MatchResult(pairs=sorted(pairs, key=lambda x: (-x.score, x.r_idx, x.g_idx)), method="greedy")

def compute_matching(sim: np.ndarray, cfg: Config) -> MatchResult:
    gate = _min_match_score(cfg)
    return hungarian_match(sim, gate)
