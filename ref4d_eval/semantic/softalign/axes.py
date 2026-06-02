from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np

from .config import Config
from .types import (
    EntityRepr,
    MatchPair,
    MatchResult,
    CatCovPairDetail,
    AICCoverageByKey,
    AICMisbindItem,
    AICPairDetail,
    HalExtraCategory,
    HalExtraAttr,
)
from .similarity import pairwise_similarity, value_pairwise_similarity
from .matching import compute_matching

_EPS = 1e-8

def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den + _EPS)

def _key_weight(cfg: Config, key: str) -> float:
    rp = getattr(cfg, "repr", None)
    if rp and isinstance(getattr(rp, "key_weight", None), dict):
        try:
            return float(rp.key_weight.get(key, 1.0))
        except Exception:
            return 1.0
    return 1.0

def _extract_kv_triplets(ent: EntityRepr) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    attrs = getattr(ent.entity, "attrs", {}) or {}
    for k, vs in attrs.items():
        if not vs:
            continue
        for v in vs:
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                out.append((k, v))
    return out

def _build_ref_bank_by_key(R: List[EntityRepr]) -> Dict[str, List[Tuple[int, str]]]:
    bank: Dict[str, List[Tuple[int, str]]] = {}
    for i, e in enumerate(R):
        for k, v in _extract_kv_triplets(e):
            bank.setdefault(k, []).append((i, v))
    return bank

class SoftCatCovCalculator:

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute(
        self,
        R: List[EntityRepr],
        G: List[EntityRepr],
        *,
        sim_matrix: Optional[np.ndarray] = None,
        matching: Optional[MatchResult] = None,
    ) -> Tuple[float, List[CatCovPairDetail]]:
        if sim_matrix is None:
            sim_matrix = pairwise_similarity(R, G, self.cfg)  
        if matching is None:
            matching = compute_matching(sim_matrix, self.cfg)

        Rn = len(R)
        cov_per_r = [0.0] * Rn
        matched_g_by_r = {p.r_idx: p.g_idx for p in matching.pairs}

        details: List[CatCovPairDetail] = []
        for ri in range(Rn):
            if ri in matched_g_by_r:
                gi = matched_g_by_r[ri]
                s = float(sim_matrix[ri, gi])
                cov_per_r[ri] = s
                details.append(CatCovPairDetail(r_idx=ri, g_idx=gi, sim=s, note="matched"))
            else:
                details.append(CatCovPairDetail(r_idx=ri, g_idx=-1, sim=0.0, note="unmatched"))

        score = (sum(cov_per_r) / max(1, Rn)) if Rn > 0 else 0.0
        return score, details

class SoftAICCalculator:

    def __init__(self, cfg: Config, encoder):
        self.cfg = cfg
        self.encoder = encoder

    def _coverage_for_pair(self, r: EntityRepr, g: EntityRepr) -> Tuple[float, float, List[AICCoverageByKey]]:
        UR = _extract_kv_triplets(r)
        if not UR:
            return 0.0, 0.0, []

        G_dict: Dict[str, List[str]] = {}
        for k, v in _extract_kv_triplets(g):
            G_dict.setdefault(k, []).append(v)

        numer = 0.0
        denom = 0.0
        by_key_details: List[AICCoverageByKey] = []

        keys = sorted({k for (k, _) in UR})
        for k in keys:
            w = _key_weight(self.cfg, k)
            vals_r = [v for (kk, v) in UR if kk == k]
            vals_g = G_dict.get(k, [])
            if vals_g:
                S = value_pairwise_similarity(vals_r, vals_g, self.encoder, self.cfg, purpose="passage", max_length=32)
                s_hit = float(S.max(axis=1).mean()) if S.size > 0 else 0.0  
            else:
                s_hit = 0.0
            weighted_hit = w * s_hit
            numer += weighted_hit
            denom += w
            by_key_details.append(
                AICCoverageByKey(
                    key=k,
                    weighted_hit=weighted_hit,
                    weighted_total=w,
                    score=_safe_div(weighted_hit, w),
                )
            )
        return numer, max(denom, _EPS), by_key_details

    def _misbind_for_pair(
        self,
        r_idx: int,
        g_idx: int,
        g: EntityRepr,
        ref_bank: Dict[str, List[Tuple[int, str]]],
        R: List[EntityRepr],
    ) -> Tuple[float, float, List[AICMisbindItem]]:
        UG = _extract_kv_triplets(g)
        if not UG:
            return 0.0, _EPS, []

        r_bank: Dict[str, List[str]] = {}
        for k, v in _extract_kv_triplets(R[r_idx]):
            r_bank.setdefault(k, []).append(v)

        items: List[AICMisbindItem] = []
        numer = 0.0
        denom = 0.0

        by_key: Dict[str, List[str]] = {}
        for k, vprime in UG:
            by_key.setdefault(k, []).append(vprime)

        for k, vprimes in by_key.items():
            ref_all = ref_bank.get(k, [])
            if not ref_all:
                
                continue

            vals_all = [val for (_idx, val) in ref_all]
            vals_r = r_bank.get(k, [])

            Sstar = value_pairwise_similarity(vprimes, vals_all, self.encoder, self.cfg, purpose="passage", max_length=32)
            s_star_best = Sstar.max(axis=1) if Sstar.size > 0 else np.zeros((len(vprimes),), dtype=np.float32)

            if vals_r:
                Sr = value_pairwise_similarity(vprimes, vals_r, self.encoder, self.cfg, purpose="passage", max_length=32)
                s_r_best = Sr.max(axis=1)
            else:
                s_r_best = np.zeros((len(vprimes),), dtype=np.float32)

            if Sstar.size > 0:
                best_pos = Sstar.argmax(axis=1)
            else:
                best_pos = np.zeros((len(vprimes),), dtype=np.int64)

            w = _key_weight(self.cfg, k)
            deltas = np.maximum(0.0, s_star_best - s_r_best)
            numer += float((w * deltas).sum())
            denom += float((w * s_star_best).sum())

            for idx, vpr in enumerate(vprimes):
                best_r_idx = int(ref_all[int(best_pos[idx])][0]) if len(ref_all) > 0 else None
                items.append(
                    AICMisbindItem(
                        g_idx=g_idx,
                        key=k,
                        value=vpr,
                        s_star=float(s_star_best[idx]),
                        s_ref=float(s_r_best[idx]),
                        delta=float(deltas[idx]),
                        weight=float(w),
                        best_r_idx=best_r_idx,
                    )
                )
        return numer, max(denom, _EPS), items

    def compute(
        self,
        R: List[EntityRepr],
        G: List[EntityRepr],
        *,
        sim_matrix: Optional[np.ndarray] = None,
        matching: Optional[MatchResult] = None,
    ) -> Tuple[float, List[AICPairDetail]]:
        if sim_matrix is None:
            sim_matrix = pairwise_similarity(R, G, self.cfg)
        if matching is None:
            matching = compute_matching(sim_matrix, self.cfg)

        if not R:
            return 0.0, []

        ref_bank = _build_ref_bank_by_key(R)

        cov_sum = 0.0
        cov_den = 0.0
        mis_sum = 0.0
        mis_den = 0.0

        results: List[AICPairDetail] = []

        for p in matching.pairs:
            r_idx, g_idx = p.r_idx, p.g_idx
            numer_cov, denom_cov, cov_by_key = self._coverage_for_pair(R[r_idx], G[g_idx])
            numer_mis, denom_mis, mis_items = self._misbind_for_pair(r_idx, g_idx, G[g_idx], ref_bank, R)

            cov_sum += numer_cov
            cov_den += denom_cov
            mis_sum += numer_mis
            mis_den += denom_mis

            cov_val = _safe_div(numer_cov, denom_cov)
            mis_val = min(max(_safe_div(numer_mis, denom_mis), 0.0), 1.0)

            results.append(
                AICPairDetail(
                    r_idx=r_idx,
                    g_idx=g_idx,
                    coverage=cov_val,
                    misbind=mis_val,
                    coverage_by_key=cov_by_key,
                    misbind_items=mis_items,
                )
            )

        coverage = _safe_div(cov_sum, cov_den) if cov_den > 0 else 0.0
        misbind = min(max(_safe_div(mis_sum, mis_den) if mis_den > 0 else 0.0, 0.0), 1.0)
        score = float(coverage * (1.0 - misbind))
        return score, results

class HallucinationPenaltyCalculator:

    def __init__(self, cfg: Config, encoder):
        self.cfg = cfg
        self.encoder = encoder

    @staticmethod
    def _group_attr_values_by_key(ent: EntityRepr) -> Dict[str, List[str]]:
        by_key: Dict[str, List[str]] = {}
        attrs = getattr(ent.entity, "attrs", {}) or {}
        for k, vs in attrs.items():
            if not isinstance(k, str) or not k.strip():
                continue
            if not vs:
                continue
            keep = [v for v in vs if isinstance(v, str) and v.strip()]
            if keep:
                by_key.setdefault(k, []).extend(keep)
        return by_key

    @staticmethod
    def _max_weight_pairs(sim: np.ndarray) -> List[Tuple[int, int]]:
        sim = np.asarray(sim, dtype=np.float32)
        if sim.ndim != 2 or sim.size == 0:
            return []

        nr, ng = sim.shape
        if nr == 0 or ng == 0:
            return []

        try:
            from scipy.optimize import linear_sum_assignment  
        except Exception as e:
            raise ImportError("scipy.optimize.linear_sum_assignment is required for Hall local attribute Hungarian matching") from e

        row_ind, col_ind = linear_sum_assignment(1.0 - sim)
        return [(int(i), int(j)) for i, j in zip(row_ind.tolist(), col_ind.tolist())]

    def _residual_values_for_key(
        self,
        vals_r: List[str],
        vals_g: List[str],
    ) -> List[Tuple[int, str]]:
        if not vals_g:
            return []
        if not vals_r:
            return [(j, v) for j, v in enumerate(vals_g)]

        S = value_pairwise_similarity(
            vals_r,
            vals_g,
            self.encoder,
            self.cfg,
            purpose="passage",
            max_length=32,
        )
        pairs = self._max_weight_pairs(S)
        matched_g = {gj for _ri, gj in pairs}
        return [(j, v) for j, v in enumerate(vals_g) if j not in matched_g]

    def _extra_cat_penalty(
        self,
        sim_matrix: np.ndarray,
        pairs: List[MatchPair],
        R_size: int,
        G_size: int,
    ) -> Tuple[float, List[HalExtraCategory]]:
        matched_g = {p.g_idx for p in pairs}
        extras: List[HalExtraCategory] = []
        numer = 0.0

        for j in range(G_size):
            if j in matched_g:
                continue
            s_best = float(sim_matrix[:, j].max()) if R_size > 0 else 0.0
            pen = 1.0 - s_best
            numer += pen
            best_r = int(sim_matrix[:, j].argmax()) if R_size > 0 else None
            extras.append(
                HalExtraCategory(
                    g_idx=j,
                    best_r_idx=best_r if R_size > 0 else None,
                    w_max=s_best,
                    penalty=pen,
                )
            )
        return numer, extras

    def _extra_attr_penalty(
        self,
        R: List[EntityRepr],
        G: List[EntityRepr],
        pairs: List[MatchPair],
        ref_bank: Dict[str, List[Tuple[int, str]]],
    ) -> Tuple[float, float, List[HalExtraAttr]]:
        numer = 0.0
        denom = 0.0
        items: List[HalExtraAttr] = []

        matched_r_by_g = {p.g_idx: p.r_idx for p in pairs}

        for g_idx, g in enumerate(G):
            g_by_key = self._group_attr_values_by_key(g)
            if not g_by_key:
                continue

            r_idx = matched_r_by_g.get(g_idx, None)
            r_by_key = self._group_attr_values_by_key(R[r_idx]) if r_idx is not None else {}

            for k, vals_g in g_by_key.items():
                w = _key_weight(self.cfg, k)

                denom += w * len(vals_g)

                if r_idx is None:
                    residuals = [(j, v) for j, v in enumerate(vals_g)]
                else:
                    vals_r = r_by_key.get(k, [])
                    residuals = self._residual_values_for_key(vals_r, vals_g)

                if not residuals:
                    continue

                ref_all = ref_bank.get(k, [])
                if not ref_all:
                    for _j, vpr in residuals:
                        items.append(
                            HalExtraAttr(
                                r_idx=r_idx,
                                g_idx=g_idx,
                                key=k,
                                value=vpr,
                                s_star=0.0,
                                penalty=1.0,
                                weight=float(w),
                                best_r_idx=None,
                            )
                        )
                    numer += w * len(residuals)
                    continue

                vals_all = [val for (_ref_i, val) in ref_all]
                residual_vals = [v for (_j, v) in residuals]
                Sstar = value_pairwise_similarity(
                    residual_vals,
                    vals_all,
                    self.encoder,
                    self.cfg,
                    purpose="passage",
                    max_length=32,
                )

                if Sstar.size > 0:
                    s_star_best = Sstar.max(axis=1)
                    best_pos = Sstar.argmax(axis=1)
                else:
                    s_star_best = np.zeros((len(residual_vals),), dtype=np.float32)
                    best_pos = np.zeros((len(residual_vals),), dtype=np.int64)

                numer += float((w * (1.0 - s_star_best)).sum())
                for idx, (_j, vpr) in enumerate(residuals):
                    best_r_idx = int(ref_all[int(best_pos[idx])][0]) if len(ref_all) > 0 else None
                    items.append(
                        HalExtraAttr(
                            r_idx=r_idx,
                            g_idx=g_idx,
                            key=k,
                            value=vpr,
                            s_star=float(s_star_best[idx]),
                            penalty=float(1.0 - s_star_best[idx]),
                            weight=float(w),
                            best_r_idx=best_r_idx,
                        )
                    )

        return numer, max(denom, _EPS), items

    def compute(
        self,
        R: List[EntityRepr],
        G: List[EntityRepr],
        *,
        sim_matrix: Optional[np.ndarray] = None,
        matching: Optional[MatchResult] = None,
    ) -> Tuple[float, List[HalExtraCategory], List[HalExtraAttr]]:
        if sim_matrix is None:
            sim_matrix = pairwise_similarity(R, G, self.cfg)
        if matching is None:
            matching = compute_matching(sim_matrix, self.cfg)

        ref_bank = _build_ref_bank_by_key(R)

        extra_cat_numer, extra_cat_items = self._extra_cat_penalty(
            sim_matrix, matching.pairs, len(R), len(G)
        )

        extra_attr_numer, extra_attr_denom, extra_attr_items = self._extra_attr_penalty(
            R, G, matching.pairs, ref_bank
        )

        hall_numer = extra_cat_numer + extra_attr_numer
        hall_denom = float(len(G)) + extra_attr_denom
        hall_rate = min(max(_safe_div(hall_numer, hall_denom) if hall_denom > 0 else 0.0, 0.0), 1.0)
        score = float(1.0 - hall_rate)
        return score, extra_cat_items, extra_attr_items
