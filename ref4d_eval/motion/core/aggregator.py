

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

__all__ = [
    "MOTION_FEATURE_ORDER",
    "MotionAggregatorParams",
    "MotionAggregatorOutput",
    "validate_atomic_motion_features",
    "build_motion_feature_vector",
    "load_motion_aggregator_params",
    "predict_motion_score",
]


MOTION_FEATURE_ORDER: Tuple[str, ...] = (
    "S_dir",
    "S_mag",
    "S_smo",
    "RF",
    "LS",
)

_VALID_FLAG_BY_FEATURE: Dict[str, str] = {
    "S_dir": "valid_dir",
    "S_mag": "valid_mag",
    "S_smo": "valid_smo",
    "RF": "valid_rf",
    "LS": "valid_ls",
}


@dataclass(frozen=True)
class MotionAggregatorParams:

    feature_order: Tuple[str, ...]
    weights: Tuple[float, ...]
    bias: float
    use_cdf_100: bool = True


@dataclass(frozen=True)
class MotionAggregatorOutput:

    motion_score: float
    motion_score_0_100: float
    feature_vector: np.ndarray
    feature_order: Tuple[str, ...]
    is_valid: bool
    error: str = ""


def _std_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(float(x) / sqrt(2.0)))


def _as_bool_flag(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"atomic valid flag '{name}' must be bool, got {type(value).__name__}")


def _as_finite_float(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:  
        raise ValueError(f"atomic feature '{name}' is not numeric") from exc
    if not np.isfinite(out):
        raise ValueError(f"atomic feature '{name}' is not finite")
    return out


def validate_atomic_motion_features(atomic_features: Mapping[str, Any]) -> Tuple[bool, str]:
    missing = [k for k in MOTION_FEATURE_ORDER if k not in atomic_features]
    if missing:
        return False, f"missing atomic feature(s): {', '.join(missing)}"

    missing_valid = [v for v in _VALID_FLAG_BY_FEATURE.values() if v not in atomic_features]
    if missing_valid:
        return False, f"missing atomic valid flag(s): {', '.join(missing_valid)}"

    for feat_name in MOTION_FEATURE_ORDER:
        valid_name = _VALID_FLAG_BY_FEATURE[feat_name]
        try:
            is_valid = _as_bool_flag(atomic_features[valid_name], name=valid_name)
        except ValueError as exc:
            return False, str(exc)
        if not is_valid:
            return False, f"atomic feature '{feat_name}' is invalid according to '{valid_name}'"
        try:
            _as_finite_float(atomic_features[feat_name], name=feat_name)
        except ValueError as exc:
            return False, str(exc)

    return True, ""


def build_motion_feature_vector(atomic_features: Mapping[str, Any]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    ok, err = validate_atomic_motion_features(atomic_features)
    if not ok:
        raise ValueError(err)

    vec = np.asarray([
        _as_finite_float(atomic_features[name], name=name)
        for name in MOTION_FEATURE_ORDER
    ], dtype=np.float64)
    if vec.shape != (len(MOTION_FEATURE_ORDER),):
        raise ValueError(
            f"feature vector shape mismatch: expected {(len(MOTION_FEATURE_ORDER),)}, got {vec.shape}"
        )
    return vec, MOTION_FEATURE_ORDER


def load_motion_aggregator_params(config: Mapping[str, Any]) -> MotionAggregatorParams:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")

    section: Mapping[str, Any]
    if "motion_aggregator" in config:
        raw = config["motion_aggregator"]
        if not isinstance(raw, Mapping):
            raise ValueError("'motion_aggregator' must be a mapping")
        section = raw
    else:
        section = config

    if "feature_order" not in section:
        raise ValueError("missing 'motion_aggregator.feature_order'")
    if "weights" not in section:
        raise ValueError("missing 'motion_aggregator.weights'")
    if "bias" not in section:
        raise ValueError("missing 'motion_aggregator.bias'")

    feature_order_raw = section["feature_order"]
    if not isinstance(feature_order_raw, Sequence) or isinstance(feature_order_raw, (str, bytes)):
        raise ValueError("'motion_aggregator.feature_order' must be a sequence of feature names")
    feature_order = tuple(str(x) for x in feature_order_raw)
    if feature_order != MOTION_FEATURE_ORDER:
        raise ValueError(
            "feature order mismatch: "
            f"config={feature_order}, expected={MOTION_FEATURE_ORDER}"
        )

    weights_raw = section["weights"]
    if not isinstance(weights_raw, Sequence) or isinstance(weights_raw, (str, bytes)):
        raise ValueError("'motion_aggregator.weights' must be a numeric sequence")
    try:
        weights_arr = np.asarray(weights_raw, dtype=np.float64).reshape(-1)
    except Exception as exc:  
        raise ValueError("'motion_aggregator.weights' must be numeric") from exc
    if weights_arr.shape != (len(MOTION_FEATURE_ORDER),):
        raise ValueError(
            "weight dimension mismatch: "
            f"expected {len(MOTION_FEATURE_ORDER)}, got {weights_arr.shape[0]}"
        )
    if not np.all(np.isfinite(weights_arr)):
        raise ValueError("'motion_aggregator.weights' contains non-finite values")

    try:
        bias = float(section["bias"])
    except Exception as exc:  
        raise ValueError("'motion_aggregator.bias' must be numeric") from exc
    if not np.isfinite(bias):
        raise ValueError("'motion_aggregator.bias' must be finite")

    use_cdf_raw = section.get("use_cdf_100", True)
    if not isinstance(use_cdf_raw, (bool, np.bool_)):
        raise ValueError("'motion_aggregator.use_cdf_100' must be bool")
    use_cdf_100 = bool(use_cdf_raw)

    return MotionAggregatorParams(
        feature_order=MOTION_FEATURE_ORDER,
        weights=tuple(float(x) for x in weights_arr.tolist()),
        bias=float(bias),
        use_cdf_100=use_cdf_100,
    )


def _predict_raw_score(feature_vector: np.ndarray, params: MotionAggregatorParams) -> float:
    vec = np.asarray(feature_vector, dtype=np.float64).reshape(-1)
    if vec.shape != (len(MOTION_FEATURE_ORDER),):
        raise ValueError(
            f"feature vector shape mismatch: expected {(len(MOTION_FEATURE_ORDER),)}, got {vec.shape}"
        )
    w = np.asarray(params.weights, dtype=np.float64).reshape(-1)
    if w.shape != vec.shape:
        raise ValueError(f"weight dimension mismatch: weights={w.shape}, feature_vector={vec.shape}")
    return float(np.dot(w, vec) + float(params.bias))


def predict_motion_score(
    atomic_features: Mapping[str, Any],
    aggregator_cfg: Mapping[str, Any] | MotionAggregatorParams,
) -> MotionAggregatorOutput:
    params = (
        aggregator_cfg
        if isinstance(aggregator_cfg, MotionAggregatorParams)
        else load_motion_aggregator_params(aggregator_cfg)
    )

    ok, err = validate_atomic_motion_features(atomic_features)
    if not ok:
        nan_vec = np.full((len(MOTION_FEATURE_ORDER),), np.nan, dtype=np.float64)
        return MotionAggregatorOutput(
            motion_score=float("nan"),
            motion_score_0_100=float("nan"),
            feature_vector=nan_vec,
            feature_order=MOTION_FEATURE_ORDER,
            is_valid=False,
            error=err,
        )

    try:
        feature_vector, feature_order = build_motion_feature_vector(atomic_features)
        motion_score = _predict_raw_score(feature_vector, params)
    except Exception as exc:
        nan_vec = np.full((len(MOTION_FEATURE_ORDER),), np.nan, dtype=np.float64)
        return MotionAggregatorOutput(
            motion_score=float("nan"),
            motion_score_0_100=float("nan"),
            feature_vector=nan_vec,
            feature_order=MOTION_FEATURE_ORDER,
            is_valid=False,
            error=str(exc),
        )

    if params.use_cdf_100:
        motion_score_0_100 = float(100.0 * _std_normal_cdf(motion_score))
    else:
        motion_score_0_100 = float("nan")

    return MotionAggregatorOutput(
        motion_score=float(motion_score),
        motion_score_0_100=float(motion_score_0_100),
        feature_vector=feature_vector.astype(np.float64, copy=False),
        feature_order=feature_order,
        is_valid=True,
        error="",
    )
