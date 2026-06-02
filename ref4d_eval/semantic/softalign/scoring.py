from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import axes as _axes
from .config import Config
from .types import (
    Entity,
    EntityRepr,
    ScoreReport,
    MatchResult,
    AxisScores,
    AxisDetails,
)
from .preprocessing import coerce_entities_from_raw
from .repr_builder import encode_entity_reprs
from .similarity import pairwise_similarity
from .matching import compute_matching

@dataclass
class LearnedSemanticAggregator:
    w_catcov: float
    w_aic: float
    w_hal: float
    bias: float = 0.0

def _validate_non_empty(doc: Optional[Dict]) -> Tuple[bool, str]:
    if not isinstance(doc, dict) or not doc:
        return False, "doc is empty or not a JSON object"
    fine = doc.get("fine")
    if isinstance(fine, dict):
        e_list = fine.get("entities", [])
        if isinstance(e_list, list) and len(e_list) > 0:
            return True, ""
    return False, "no entities found in fine.entities"

def _calc_unmatched(Rn: int, Gn: int, m: MatchResult) -> Tuple[List[int], List[int]]:
    matched_r = {p.r_idx for p in m.pairs}
    matched_g = {p.g_idx for p in m.pairs}
    ref_un = [i for i in range(Rn) if i not in matched_r]
    gen_un = [j for j in range(Gn) if j not in matched_g]
    return ref_un, gen_un

def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _semantic_aggregator_from_cfg(cfg: Config) -> LearnedSemanticAggregator:
    cand = getattr(cfg, "semantic_aggregator", None)
    if cand is None:
        raise ValueError(
            "Missing learned semantic aggregator in config. "
            "Expected cfg.semantic_aggregator with format {weights:{catcov,aic,hal}, bias}."
        )

    weights = _obj_get(cand, "weights", None)
    if not isinstance(weights, dict):
        raise ValueError(
            "Invalid learned semantic aggregator format in config. "
            "Expected cfg.semantic_aggregator with format {weights:{catcov,aic,hal}, bias}."
        )

    try:
        return LearnedSemanticAggregator(
            w_catcov=float(weights["catcov"]),
            w_aic=float(weights["aic"]),
            w_hal=float(weights["hal"]),
            bias=float(_obj_get(cand, "bias", 0.0)),
        )
    except Exception as e:
        raise ValueError(
            "Invalid learned semantic aggregator format in config. "
            "Expected cfg.semantic_aggregator with format {weights:{catcov,aic,hal}, bias}."
        ) from e

def _stdnorm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))

def _cdf_to_100(x: float) -> float:
    return 100.0 * _stdnorm_cdf(x)

class SampleScorer:

    def __init__(self, cfg: Config, encoder):
        self.cfg = cfg
        self.encoder = encoder
        self._catcov = _axes.SoftCatCovCalculator(cfg)
        self._aic = _axes.SoftAICCalculator(cfg, encoder)
        hal_cls = getattr(_axes, "HallucinationPenaltyCalculator", None)
        if hal_cls is None:
            raise ImportError("scoring.py requires HallucinationPenaltyCalculator in softalign.axes")
        self._hal = hal_cls(cfg, encoder)
        self._semantic_agg = _semantic_aggregator_from_cfg(cfg)

    @staticmethod
    def _load_json(path: str | Path) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def score_pair(
        self,
        ref_doc_or_path: Dict | str | Path,
        gen_doc_or_path: Dict | str | Path,
        sample_id: Optional[str] = None,
    ) -> ScoreReport:
        
        ref_doc = ref_doc_or_path if isinstance(ref_doc_or_path, dict) else self._load_json(ref_doc_or_path)
        gen_doc = gen_doc_or_path if isinstance(gen_doc_or_path, dict) else self._load_json(gen_doc_or_path)

        ok_r, why_r = _validate_non_empty(ref_doc)
        ok_g, why_g = _validate_non_empty(gen_doc)

        if not ok_r or not ok_g:
            info = {
                "skipped": True,
                "skip_reason": f"ref invalid: {why_r}" if not ok_r else (f"gen invalid: {why_g}" if not ok_g else ""),
            }
            nan = float("nan")
            return ScoreReport(
                sample_id=sample_id or "",
                axis=AxisScores(catcov=nan, aic=nan, hal=nan),
                s_base=nan,
                details=AxisDetails(),
                sizes={"R": 0, "G": 0, "matched": 0},
                info=info,
            )

        preproc_cfg = getattr(self.cfg, "preproc", None)
        R_entities: List[Entity] = coerce_entities_from_raw(ref_doc, preproc_cfg)
        G_entities: List[Entity] = coerce_entities_from_raw(gen_doc, preproc_cfg)

        R: List[EntityRepr] = encode_entity_reprs(R_entities, self.encoder, self.cfg)
        G: List[EntityRepr] = encode_entity_reprs(G_entities, self.encoder, self.cfg)

        if len(R) == 0 or len(G) == 0:
            info = {
                "skipped": True,
                "skip_reason": "no valid entities after preprocessing",
            }
            nan = float("nan")
            return ScoreReport(
                sample_id=sample_id or "",
                axis=AxisScores(catcov=nan, aic=nan, hal=nan),
                s_base=nan,
                details=AxisDetails(),
                sizes={"R": len(R), "G": len(G), "matched": 0},
                info=info,
            )

        sim = pairwise_similarity(R, G, self.cfg)
        match: MatchResult = compute_matching(sim, self.cfg)
        ref_un, gen_un = _calc_unmatched(len(R), len(G), match)

        s_catcov, catcov_details = self._catcov.compute(R, G, sim_matrix=sim, matching=match)
        s_aic, aic_pair_details = self._aic.compute(R, G, sim_matrix=sim, matching=match)
        s_hal, hal_extra_cats, hal_extra_attrs = self._hal.compute(R, G, sim_matrix=sim, matching=match)

        agg = self._semantic_agg
        semantic_raw = float(
            agg.w_catcov * s_catcov +
            agg.w_aic * s_aic +
            agg.w_hal * s_hal +
            agg.bias
        )
        semantic_cdf_0_100 = float(_cdf_to_100(semantic_raw))

        details = AxisDetails(
            catcov_pairs=catcov_details,
            aic_pairs=aic_pair_details,
            hal_extra_categories=hal_extra_cats,
            hal_extra_attrs=hal_extra_attrs,
            misc={
                "matching": {
                    "algo": getattr(match, "method", "hungarian"),
                    "num_pairs": len(match.pairs),
                    "ref_unmatched": ref_un,
                    "gen_unmatched": gen_un,
                    "min_score": float(getattr(getattr(self.cfg, "matching", None), "min_score", 0.30)),
                },
                "semantic_aggregator": {
                    "w_catcov": agg.w_catcov,
                    "w_aic": agg.w_aic,
                    "w_hal": agg.w_hal,
                    "bias": agg.bias,
                },
            },
        )

        return ScoreReport(
            sample_id=sample_id or "",
            axis=AxisScores(catcov=s_catcov, aic=s_aic, hal=s_hal),
            s_base=semantic_raw,
            details=details,
            sizes={"R": len(R), "G": len(G), "matched": len(match.pairs)},
            info={
                "sim": {
                    "tau0": float(getattr(getattr(self.cfg, "sim", None), "tau0", 0.30)),
                    "combine": "weighted",
                    "w_name": float(getattr(getattr(self.cfg, "sim", None), "w_name", 0.5)),
                    "w_set": float(getattr(getattr(self.cfg, "sim", None), "w_set", 0.5)),
                },
                "semantic_final": {
                    "raw": semantic_raw,
                    "cdf_0_100": semantic_cdf_0_100,
                },
            },
        )

    def score_pair_from_files(
        self,
        ref_json_path: str | Path,
        gen_json_path: str | Path,
        sample_id: Optional[str] = None,
    ) -> ScoreReport:
        return self.score_pair(ref_json_path, gen_json_path, sample_id=sample_id)
