
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import Config, EncoderConfig
from .types import Embedding

class TextEncoder:
    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        purpose: str = "passage",
        max_length: int | None = None,
    ) -> List[Embedding]:
        raise NotImplementedError

    def close(self):
        pass

def _dtype_of(device: str, dtype_name: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    name = (dtype_name or "").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32

def _l2_normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(n, eps, None)

def _batch_iter(items: Sequence[str], batch_size: int):
    n = len(items)
    for i in range(0, n, batch_size):
        yield items[i : i + batch_size]

def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)  
    summed = (last_hidden_state * mask).sum(dim=1)  
    counts = mask.sum(dim=1).clamp(min=1e-6)       
    return summed / counts

class E5Encoder(TextEncoder):

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ecfg: EncoderConfig = cfg.encoder

        want_cuda = (self.ecfg.device or "cuda").lower() == "cuda"
        self.device = "cuda" if (want_cuda and torch.cuda.is_available()) else "cpu"
        self.torch_dtype = _dtype_of(self.device, self.ecfg.dtype)

        tok_kwargs = {
            "cache_dir": self.ecfg.cache_dir,
            "local_files_only": self.ecfg.local_files_only,
            "trust_remote_code": bool(self.ecfg.trust_remote_code),
        }
        tok_kwargs = {k: v for k, v in tok_kwargs.items() if v is not None}
        self.tokenizer = AutoTokenizer.from_pretrained(self.ecfg.model_name_or_path, **tok_kwargs)

        mdl_kwargs = {
            "torch_dtype": self.torch_dtype,
            "cache_dir": self.ecfg.cache_dir,
            "local_files_only": self.ecfg.local_files_only,
            "trust_remote_code": bool(self.ecfg.trust_remote_code),
        }
        if self.ecfg.revision:
            mdl_kwargs["revision"] = self.ecfg.revision
        mdl_kwargs = {k: v for k, v in mdl_kwargs.items() if v is not None}

        self.model = AutoModel.from_pretrained(self.ecfg.model_name_or_path, **mdl_kwargs)
        self.model.eval()
        self.model.to(self.device)

        self._cache: dict[Tuple[str, int, str], np.ndarray] = {}

    def _prefix(self, texts: Sequence[str], purpose: str) -> List[str]:
        if purpose == "query":
            pfx = self.ecfg.query_prefix or ""
        else:
            pfx = self.ecfg.passage_prefix or ""
        if not pfx:
            return list(texts)
        return [f"{pfx}{t}" for t in texts]

    def _max_len(self, purpose: str, max_length: int | None) -> int:
        if max_length is not None:
            return int(max_length)
        if purpose == "query":
            return int(self.ecfg.max_length_query or 128)
        return int(self.ecfg.max_length_passage or 256)

    @torch.inference_mode()
    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        purpose: str = "passage",
        max_length: int | None = None,
    ) -> List[Embedding]:
        if not isinstance(texts, (list, tuple)):
            texts = list(texts)

        resolved_max_len = self._max_len(purpose, max_length)

        cached_vecs: List[np.ndarray] = []
        uncached_texts: List[str] = []
        uncached_idx: List[int] = []
        for idx, t in enumerate(texts):
            key = (purpose, resolved_max_len, t)
            if key in self._cache:
                cached_vecs.append(self._cache[key])
            else:
                cached_vecs.append(None)  
                uncached_texts.append(t)
                uncached_idx.append(idx)

        if uncached_texts:
            prefixed = self._prefix(uncached_texts, purpose)
            bs = max(1, int(self.ecfg.batch_size or 8))

            out_chunks: List[np.ndarray] = []
            for chunk in _batch_iter(prefixed, bs):
                enc = self.tokenizer(
                    list(chunk),
                    padding=True,
                    truncation=True,
                    max_length=resolved_max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc)
                pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])  
                pooled = pooled.detach().to(torch.float32).cpu().numpy()
                out_chunks.append(pooled)

            arr = np.concatenate(out_chunks, axis=0)
            arr = _l2_normalize_rows(arr)

            for i, vec in zip(uncached_idx, list(arr)):
                key = (purpose, resolved_max_len, texts[i])
                self._cache[key] = vec.astype(np.float32, copy=False)

                cached_vecs[i] = self._cache[key]

        final = [Embedding(vec=v.astype(np.float32, copy=False)) for v in cached_vecs] if cached_vecs else []
        return final

    def close(self):
        try:
            del self.model
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._cache.clear()

def build_text_encoder(cfg: Config) -> TextEncoder:
    return E5Encoder(cfg)

__all__ = ["TextEncoder", "build_text_encoder", "E5Encoder"]
