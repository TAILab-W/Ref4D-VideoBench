

from __future__ import annotations
import argparse
import logging
import math
from typing import Any, Dict, List, Tuple
from pathlib import Path

from ..common.io import read_yaml, ensure_dir, write_json
from ..metrics.ega import compute_ega
from ..metrics.erel import compute_ERel
from ..metrics.ecr import compute_ecr

LOGGER = logging.getLogger("event_eval.metrics.aggregate")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


def _portable_path(p: str) -> str:
    try:
        path = Path(p)
        parts = path.parts
        for anchor in ("outputs", "data", "ref4d_eval", "docs", "checkpoints", "third_party"):
            if anchor in parts:
                idx = parts.index(anchor)
                return Path(*parts[idx:]).as_posix()
        return path.name
    except Exception:
        return str(p)


def _branch_from_cfg(reg: Dict[str, Any], name: str, expected_features: List[str]) -> Dict[str, Any]:
    branch = reg.get(name, None)
    if not isinstance(branch, dict):
        raise ValueError(f"Missing aggregate.event_regressor.{name}.")
    features = branch.get("features", None)
    coef = branch.get("coef", None)
    intercept = branch.get("intercept", None)
    if list(features or []) != expected_features:
        raise ValueError(f"aggregate.event_regressor.{name}.features must be {expected_features}.")
    if not isinstance(coef, (list, tuple)) or len(coef) != len(expected_features):
        raise ValueError(f"aggregate.event_regressor.{name}.coef must have {len(expected_features)} values.")
    if intercept is None:
        raise ValueError(f"Missing aggregate.event_regressor.{name}.intercept.")
    try:
        c = [float(x) for x in coef]
        b = float(intercept)
    except Exception as exc:
        raise ValueError(f"aggregate.event_regressor.{name} values must be numeric.") from exc
    if not all(math.isfinite(x) for x in c):
        raise ValueError(f"aggregate.event_regressor.{name}.coef must be finite, got {c}")
    if not math.isfinite(b):
        raise ValueError(f"aggregate.event_regressor.{name}.intercept must be finite, got {b}")
    return {"features": expected_features, "coef": c, "intercept": b}


def _event_regressor_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    agg = cfg.get("aggregate") or {}
    reg = agg.get("event_regressor") or {}
    return {
        "valid_branch": _branch_from_cfg(reg, "valid_branch", ["EGA", "ERel", "ECR"]),
        "omitted_branch": _branch_from_cfg(reg, "omitted_branch", ["EGA", "ECR"]),
    }


def _std_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def aggregate_scores(ref_events_json: str, gen_events_json: str, pairs_json: str, cfg_path: str, out_json: str) -> Dict[str, Any]:
    cfg = read_yaml(cfg_path)
    regressors = _event_regressor_from_cfg(cfg)

    ega = compute_ega(pairs_json, ref_events_json, cfg_path)
    ERel = compute_ERel(pairs_json, ref_events_json, gen_events_json, cfg_path)
    ecr = compute_ecr(ref_events_json, gen_events_json, pairs_json)

    ega_valid = bool(ega.get("valid", False))
    erel_valid = bool(ERel.get("valid", False))
    ecr_valid = bool(ecr.get("valid", False))

    
    atomic_valid = ega_valid and ecr_valid
    used_branch = None
    if atomic_valid:
        x_ega = float(ega.get("score", 0.0))
        if not math.isfinite(x_ega):
            raise ValueError(f"EGA score must be finite when valid=True, got {x_ega}")
        x_ecr = float(ecr.get("score", 0.0))
        if not math.isfinite(x_ecr):
            raise ValueError(f"ECR score must be finite when valid=True, got {x_ecr}")
        if erel_valid:
            x_erel = float(ERel.get("score", 0.0))
            if not math.isfinite(x_erel):
                raise ValueError(f"ERel score must be finite when valid=True, got {x_erel}")
            branch = regressors["valid_branch"]
            event_score = float(branch["coef"][0] * x_ega + branch["coef"][1] * x_erel + branch["coef"][2] * x_ecr + branch["intercept"])
            used_branch = "valid_branch"
        else:
            branch = regressors["omitted_branch"]
            event_score = float(branch["coef"][0] * x_ega + branch["coef"][1] * x_ecr + branch["intercept"])
            used_branch = "omitted_branch"
        event_score_0_100 = float(100.0 * _std_norm_cdf(event_score))
    else:
        event_score = None
        event_score_0_100 = None

    out = {
        "EGA": ega,
        "ERel": ERel,
        "ECR": ecr,
        "event_score": event_score,
        "event_score_0_100": event_score_0_100,
        "regressor": {
            "valid_branch": regressors["valid_branch"],
            "omitted_branch": regressors["omitted_branch"],
            "used_branch": used_branch,
        },
        "paths": {
            "ref_events": _portable_path(ref_events_json),
            "gen_events": _portable_path(gen_events_json),
            "pairs": _portable_path(pairs_json),
            "config": _portable_path(cfg_path),
        },
    }

    ensure_dir(Path(out_json).parent)
    write_json(out, out_json, indent=2)
    if event_score is None:
        LOGGER.info(f"Wrote scores: {out_json}  (event_score=None)")
    else:
        LOGGER.info(
            f"Wrote scores: {out_json}  (event_score={event_score:.6f}, event_score_0_100={event_score_0_100:.2f})"
        )
    return out


def parse_args():
    ap = argparse.ArgumentParser(
        description="Aggregate event-level scores into event_score and event_score_0_100."
    )
    ap.add_argument(
        "--ref_events",
        type=str,
        required=True,
        help="Path to canonical merged reference event evidence JSON.",
    )
    ap.add_argument(
        "--gen_events",
        type=str,
        required=True,
        help="Path to canonical merged generated event evidence JSON.",
    )
    ap.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to outputs/event/cache/match/<pair_id>/pairs.json.",
    )
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configs/default.yaml containing aggregate.event_regressor.",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to write event_scores.json.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aggregate_scores(args.ref_events, args.gen_events, args.pairs, args.config, args.out)
