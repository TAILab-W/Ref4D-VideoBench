
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .config import Config, load_config
from .encoder import TextEncoder, build_text_encoder
from .scoring import SampleScorer
from .types import ScoreReport

class SoftAlignAPI:

    def __init__(self, cfg: Config, encoder: TextEncoder):
        self.cfg = cfg
        self.encoder = encoder
        self._scorer = SampleScorer(cfg, encoder)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "SoftAlignAPI":
        cfg: Config = load_config(yaml_path)
        enc: TextEncoder = build_text_encoder(cfg)
        return cls(cfg, enc)

    def score_pair(
        self,
        ref_doc_or_path: Dict | str | Path,
        gen_doc_or_path: Dict | str | Path,
        sample_id: Optional[str] = None,
    ) -> ScoreReport:
        return self._scorer.score_pair(ref_doc_or_path, gen_doc_or_path, sample_id=sample_id)

    def score_pair_from_files(
        self,
        ref_json_path: str | Path,
        gen_json_path: str | Path,
        sample_id: Optional[str] = None,
    ) -> ScoreReport:
        return self._scorer.score_pair_from_files(ref_json_path, gen_json_path, sample_id=sample_id)

def build_api(yaml_path: str | Path) -> SoftAlignAPI:
    return SoftAlignAPI.from_yaml(yaml_path)
