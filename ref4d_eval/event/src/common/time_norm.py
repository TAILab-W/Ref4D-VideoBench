

"""
Time normalization & validation helpers.

Contract:
- Input "segments" can be:
  * List[Dict]: items containing at least {"s_abs": float, "e_abs": float}, and optional "id"
  * List[List[float]] or List[Tuple[float,float]]: interpreted as [s_abs, e_abs] in seconds
- We normalize each video's timeline independently to [0,1].

Outputs:
- Each segment becomes a dict with:
    {
      "id": str,                # if provided; else auto "e{idx:03d}"
      "s_abs": float, "e_abs": float,
      "s": float, "e": float,   # normalized to [0,1]
      "dur_abs": float, "dur": float
    }
- We do NOT modify ordering; callers may sort if needed.
"""

from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import math
import logging

LOGGER = logging.getLogger("event_eval.common.time_norm")
if not LOGGER.handlers:
    import logging as _logging
    h = _logging.StreamHandler()
    fmt = _logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    h.setFormatter(fmt)
    LOGGER.addHandler(h)
    LOGGER.setLevel(_logging.INFO)


@dataclass(frozen=True)
class Segment:
    s_abs: float
    e_abs: float
    id: str | None = None


def _to_segments(raw: Sequence[Any]) -> List[Segment]:
    out: List[Segment] = []
    for idx, item in enumerate(raw):
        if isinstance(item, dict):
            try:
                s_abs = float(item["s_abs"])
                e_abs = float(item["e_abs"])
            except Exception as e:
                raise ValueError(f"Event[{idx}] missing s_abs/e_abs or not float: {item}") from e
            seg_id = str(item.get("id")) if ("id" in item and item["id"] is not None) else None
            out.append(Segment(s_abs=s_abs, e_abs=e_abs, id=seg_id))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            s_abs = float(item[0]); e_abs = float(item[1])
            out.append(Segment(s_abs=s_abs, e_abs=e_abs, id=None))
        else:
            raise ValueError(f"Unsupported segment format at index {idx}: {item}")
    return out


def _sanitize_bounds(segments: List[Segment],
                     total_sec: Optional[float],
                     clamp: bool = True) -> Tuple[List[Segment], float]:
    if not segments:
        return [], 0.0

    inferred_total = max(seg.e_abs for seg in segments)
    if total_sec is None:
        total = max(inferred_total, 1e-12)
    else:
        total = max(float(total_sec), 1e-12)

    fixed: List[Segment] = []
    for i, seg in enumerate(segments):
        s, e = seg.s_abs, seg.e_abs
        if clamp:
            s = max(0.0, min(s, total))
            e = max(0.0, min(e, total))
        if not (0.0 <= s <= total and 0.0 <= e <= total):
            raise ValueError(f"Segment[{i}] out of bounds after clamp: s={s}, e={e}, total={total}")
        if e < s:
            
            LOGGER.warning(f"Segment[{i}] has e_abs < s_abs; swapping. ({s} > {e})")
            s, e = e, s
        fixed.append(Segment(s_abs=s, e_abs=e, id=seg.id))

    return fixed, total


def normalize_segments(raw_segments: Sequence[Any],
                       total_sec: Optional[float] = None,
                       eps: float = 1e-12,
                       clamp: bool = True) -> List[Dict[str, float | str]]:
    segs = _to_segments(raw_segments)
    segs, T = _sanitize_bounds(segs, total_sec, clamp=clamp)

    if T <= eps:
        raise ValueError(f"total duration too small to normalize: {T}")

    out: List[Dict[str, float | str]] = []
    for idx, seg in enumerate(segs):
        s = seg.s_abs / T
        e = seg.e_abs / T
        
        s = max(0.0, min(1.0, s))
        e = max(0.0, min(1.0, e))
        dur_abs = max(0.0, seg.e_abs - seg.s_abs)
        dur = max(0.0, e - s)
        out.append({
            "id": seg.id if seg.id is not None else f"e{idx:03d}",
            "s_abs": seg.s_abs,
            "e_abs": seg.e_abs,
            "s": s,
            "e": e,
            "dur_abs": dur_abs,
            "dur": dur,
        })
    return out




def is_concurrent(a: Dict[str, float], b: Dict[str, float], eps_sync: float = 1e-3) -> bool:
    return max(abs(float(a["s"]) - float(b["s"])),
               abs(float(a["e"]) - float(b["e"]))) <= float(eps_sync)


def precedence(a: Dict[str, float], b: Dict[str, float]) -> int:
    ea, sb = float(a["e"]), float(b["s"])
    eb, sa = float(b["e"]), float(a["s"])
    if ea <= sb:
        return +1
    if eb <= sa:
        return -1
    return 0
