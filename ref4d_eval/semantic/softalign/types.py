
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

__all__ = [
    
    "Entity", "VideoDoc",
    
    "EntityRepr", "Embedding",
    
    "SimMatrix",
    
    "MatchPair", "MatchResult",
    
    "CatCovPairDetail",
    
    "AICCoverageByKey", "AICMisbindItem", "AICPairDetail",
    
    "HalExtraCategory", "HalExtraAttr",
    
    "AxisScores", "AxisDetails", "ScoreReport",
]

@dataclass
class Entity:
    id: str
    name: str
    attrs: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class VideoDoc:
    entities: List[Entity] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EntityRepr:
    entity: Entity
    name_text: str
    frag_texts: List[str] = field(default_factory=list)
    frag_weights: List[float] = field(default_factory=list)
    name_vec: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    set_vec: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))

@dataclass
class Embedding:
    vec: np.ndarray

@dataclass
class SimMatrix:
    name: np.ndarray
    set: np.ndarray
    fused: np.ndarray

@dataclass
class MatchPair:
    r_idx: int
    g_idx: int
    score: float

@dataclass
class MatchResult:
    pairs: List[MatchPair] = field(default_factory=list)
    method: str = "hungarian"

@dataclass
class CatCovPairDetail:
    r_idx: int
    g_idx: int
    sim: float
    note: str = ""

@dataclass
class AICCoverageByKey:
    key: str
    weighted_hit: float
    weighted_total: float
    score: float

@dataclass
class AICMisbindItem:
    g_idx: int
    key: str
    value: str
    s_star: float
    s_ref: float
    delta: float
    weight: float
    best_r_idx: Optional[int] = None

@dataclass
class AICPairDetail:
    r_idx: int
    g_idx: int
    coverage: float
    misbind: float
    coverage_by_key: List[AICCoverageByKey] = field(default_factory=list)
    misbind_items: List[AICMisbindItem] = field(default_factory=list)

@dataclass
class HalExtraCategory:
    g_idx: int
    best_r_idx: Optional[int]
    w_max: float
    penalty: float

@dataclass
class HalExtraAttr:
    r_idx: Optional[int]
    g_idx: int
    key: str
    value: str
    s_star: float
    penalty: float
    weight: float
    best_r_idx: Optional[int] = None

@dataclass
class AxisScores:
    catcov: float = 0.0
    aic: float = 0.0
    hal: float = 0.0

@dataclass
class AxisDetails:
    catcov_pairs: List[CatCovPairDetail] = field(default_factory=list)
    aic_pairs: List[AICPairDetail] = field(default_factory=list)
    hal_extra_categories: List[HalExtraCategory] = field(default_factory=list)
    hal_extra_attrs: List[HalExtraAttr] = field(default_factory=list)
    misc: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScoreReport:
    sample_id: str
    axis: AxisScores
    s_base: float
    details: AxisDetails = field(default_factory=AxisDetails)
    sizes: Dict[str, int] = field(default_factory=lambda: {"R": 0, "G": 0, "matched": 0})
    info: Dict[str, Any] = field(default_factory=dict)
