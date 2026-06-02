
from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import copy
import yaml

DEFAULT_FACETS: Tuple[str, ...] = (
    "color", "pattern", "texture", "material", "size", "age", "sex", "state", "pose", "action",
    "orientation", "facing-direction", "position", "object-part", "tool-or-instrument", "equipment",
    "species-or-breed", "vehicle-type", "food-type", "brand-or-logo", "printed-text", "number-or-id",
    "art-medium", "style", "weather", "lighting", "scene", "camera-view"
)

DEFAULT_KEY_WEIGHT: Dict[str, float] = {
    **{k: 1.0 for k in DEFAULT_FACETS},
    "number-or-id": 2.0,
    "brand-or-logo": 2.0,
    "printed-text": 2.0,
    "pattern": 1.5,
    "pose": 1.5,
    "state": 1.5,
}

@dataclass
class EncoderConfig:
    model_name_or_path: str = "intfloat/e5-large-v2"
    device: str = "cuda"                 
    dtype: str = "bf16"                  
    batch_size: int = 128
    max_length_query: int = 64
    max_length_passage: int = 128
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    revision: Optional[str] = None
    trust_remote_code: bool = False

@dataclass
class PreprocConfig:
    
    lowercase: bool = True
    strip: bool = True
    canonical_hyphen: bool = True
    drop_nonalpha: bool = True
    drop_short_token_len: int = 1  

@dataclass
class ReprConfig:
    
    key_weight: Dict[str, float] = field(default_factory=lambda: copy.deepcopy(DEFAULT_KEY_WEIGHT))
    include_name_in_set: bool = False
    name_weight_in_set: float = 1.0
    name_channel_purpose: str = "query"
    set_channel_purpose: str = "passage"

@dataclass
class SimConfig:
    
    tau0: float = 0.50
    combine: str = "weighted"
    w_name: float = 0.5
    w_set: float = 0.5

@dataclass
class MatchingConfig:
    
    algorithm: str = "hungarian"
    min_score: float = 0.30   

@dataclass
class SemanticAggregatorConfig:
    
    weights: Dict[str, float] = field(default_factory=lambda: {
        "catcov": 1.0,
        "aic": 1.0,
        "hal": 1.0,
    })
    bias: float = 0.0

@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    repr: ReprConfig = field(default_factory=ReprConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    semantic_aggregator: SemanticAggregatorConfig = field(default_factory=SemanticAggregatorConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _merge_into_dataclass(dc_obj, overrides: Dict[str, Any]) -> None:
    if not is_dataclass(dc_obj) or not isinstance(overrides, dict):
        return
    for k, v in overrides.items():
        if not hasattr(dc_obj, k):
            continue
        cur = getattr(dc_obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _merge_into_dataclass(cur, v)
        else:
            setattr(dc_obj, k, v)

def _clip01(x: float) -> float:
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0

def _alias_yaml_to_internal(d: Dict[str, Any]) -> Dict[str, Any]:
    d = copy.deepcopy(d) if isinstance(d, dict) else {}

    if "preprocess" in d and "preproc" not in d:
        src = d.get("preprocess") or {}
        pre = {
            "lowercase": bool(src.get("lowercase", True)),
            "strip": bool(src.get("strip", True)),
            "canonical_hyphen": bool(src.get("unify_hyphen_underscore", True)),
            "drop_nonalpha": bool(src.get("drop_if_no_alpha", True)),
        }
        if "drop_short_token_len" in src:
            try:
                pre["drop_short_token_len"] = int(src["drop_short_token_len"])
            except Exception:
                pass
        d["preproc"] = pre

    if "similarity" in d and "sim" not in d:
        src = d.get("similarity") or {}
        sim = {
            "tau0": float(src.get("tau0", 0.50)),
            "combine": "weighted",
            "w_name": float(src.get("w_name", 0.5)),
            "w_set": float(src.get("w_set", 0.5)),
        }
        if "gate_min" in src:
            d.setdefault("matching", {})
            try:
                d["matching"]["min_score"] = float(src["gate_min"])
            except Exception:
                pass
        d["sim"] = sim

    if "sim" in d and isinstance(d["sim"], dict):
        d["sim"]["combine"] = "weighted"

    if "matching" in d:
        mt = d["matching"]
        if isinstance(mt, dict):
            if ("min_gate" in mt) and ("min_score" not in mt):
                try:
                    mt["min_score"] = float(mt["min_gate"])
                except Exception:
                    pass

            if "algo" in mt and "algorithm" not in mt:
                mt["algorithm"] = mt["algo"]

            mt["algorithm"] = "hungarian"

    return d

def _resolve_local_path_conservative(value: Optional[str], yaml_dir: Path) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    raw = value.strip()
    if not raw:
        return value

    if raw.startswith("~"):
        return str(Path(raw).expanduser().resolve())

    p = Path(raw)
    if p.is_absolute():
        return str(p.expanduser().resolve())

    if raw == "." or raw.startswith("./") or raw == ".." or raw.startswith("../"):
        return str((yaml_dir / p).resolve())

    candidate = yaml_dir / p
    if candidate.exists():
        return str(candidate.resolve())

    return value

def _resolve_paths_relative_to_yaml(cfg: Config, yaml_path: Path) -> Config:
    yaml_dir = yaml_path.resolve().parent
    cfg.encoder.model_name_or_path = _resolve_local_path_conservative(cfg.encoder.model_name_or_path, yaml_dir)
    cfg.encoder.cache_dir = _resolve_local_path_conservative(cfg.encoder.cache_dir, yaml_dir)
    return cfg

def _validate_and_fix(cfg: Config) -> Config:
    
    if cfg.encoder.device not in ("cuda", "cpu"):
        cfg.encoder.device = "cuda"
    if cfg.encoder.dtype not in ("bf16", "fp16", "fp32"):
        cfg.encoder.dtype = "bf16"
    cfg.encoder.batch_size = max(1, int(cfg.encoder.batch_size))
    cfg.encoder.max_length_query = max(8, int(cfg.encoder.max_length_query))
    cfg.encoder.max_length_passage = max(8, int(cfg.encoder.max_length_passage))

    cfg.preproc.drop_short_token_len = max(0, int(cfg.preproc.drop_short_token_len))

    fixed_kw: Dict[str, float] = {}
    for k, v in (cfg.repr.key_weight or {}).items():
        try:
            kk = str(k).strip().lower()
            if kk == "signature":
                continue
            fixed_kw[kk] = float(v)
        except Exception:
            continue
    for k, v in DEFAULT_KEY_WEIGHT.items():
        fixed_kw.setdefault(k, float(v))
    cfg.repr.key_weight = fixed_kw
    cfg.repr.include_name_in_set = bool(cfg.repr.include_name_in_set)
    try:
        cfg.repr.name_weight_in_set = float(cfg.repr.name_weight_in_set)
    except Exception:
        cfg.repr.name_weight_in_set = 1.0
    if cfg.repr.name_channel_purpose not in ("query", "passage"):
        cfg.repr.name_channel_purpose = "query"
    if cfg.repr.set_channel_purpose not in ("query", "passage"):
        cfg.repr.set_channel_purpose = "passage"

    cfg.sim.tau0 = min(0.99, max(0.0, float(cfg.sim.tau0)))
    cfg.sim.combine = "weighted"
    try:
        cfg.sim.w_name = float(cfg.sim.w_name)
    except Exception:
        cfg.sim.w_name = 0.5
    try:
        cfg.sim.w_set = float(cfg.sim.w_set)
    except Exception:
        cfg.sim.w_set = 0.5
    total_w = max(1e-12, cfg.sim.w_name + cfg.sim.w_set)
    cfg.sim.w_name = float(cfg.sim.w_name / total_w)
    cfg.sim.w_set = float(cfg.sim.w_set / total_w)

    cfg.matching.algorithm = "hungarian"
    cfg.matching.min_score = _clip01(float(cfg.matching.min_score))

    weights = cfg.semantic_aggregator.weights or {}
    fixed_weights: Dict[str, float] = {}
    for key in ("catcov", "aic", "hal"):
        try:
            fixed_weights[key] = float(weights.get(key, 1.0))
        except Exception:
            fixed_weights[key] = 1.0
    cfg.semantic_aggregator.weights = fixed_weights
    try:
        cfg.semantic_aggregator.bias = float(cfg.semantic_aggregator.bias)
    except Exception:
        cfg.semantic_aggregator.bias = 0.0

    return cfg

def load_config(yaml_path: Union[str, Path]) -> Config:
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Config YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw = _alias_yaml_to_internal(raw)

    cfg = Config()
    _merge_into_dataclass(cfg, raw)
    cfg = _resolve_paths_relative_to_yaml(cfg, p)
    cfg = _validate_and_fix(cfg)
    return cfg
