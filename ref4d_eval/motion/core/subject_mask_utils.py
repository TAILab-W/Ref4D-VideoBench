

from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
import os
from pathlib import Path
import numpy as np
import cv2 as cv

from ..preprocess.subject_mask import (
    _gather_text_prompts,
    _detect_boxes_grounding_dino,
    _segment_video_sam2_with_boxes,
    _postprocess_mask,
)

__all__ = [
    "build_subject_masks_single",
    "build_fg_bg_masks_single",
    "make_bg_complement",
]


def _find_repo_root(base_dir: Optional[str] = None) -> Path:
    if base_dir:
        return Path(str(base_dir)).expanduser().resolve()

    env_root = str(os.environ.get("REF4D_REPO_ROOT", "") or "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    markers = ("ref4d_eval", "third_party", "checkpoints", "scripts")
    for p in [here.parent, *here.parents]:
        if all((p / m).exists() for m in markers):
            return p

    raise RuntimeError(
        "Unable to determine repository root: provide `base_dir`, set "
        "`REF4D_REPO_ROOT`, or run inside a standard Ref4D-VideoBench "
        "repository layout containing ref4d_eval/ third_party/ checkpoints/ "
        "and scripts/."
    )


def _resolve_repo_root_path(path_like: str, base_dir: Optional[str] = None) -> str:
    p = Path(str(path_like)).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((_find_repo_root(base_dir=base_dir) / p).resolve())


def _resolve_semantic_root(
    cfg_subject: Dict[str, Any],
    *,
    semantic_root: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> str:
    explicit = str(semantic_root or "").strip()
    if explicit:
        return _resolve_repo_root_path(explicit, base_dir=base_dir)

    cfg_root = str(cfg_subject.get("semantic_root", "") or "").strip()
    if cfg_root:
        return _resolve_repo_root_path(cfg_root, base_dir=base_dir)

    return str((_find_repo_root(base_dir=base_dir) / "data" / "metadata" / "semantic_evidence").resolve())


def _set_env_from_cfg(cfg_subject: Dict[str, Any], base_dir: Optional[str] = None) -> None:
    
    gdino_cfg = str(cfg_subject.get("gdino_cfg", "") or "").strip()
    gdino_ckpt = str(cfg_subject.get("gdino_ckpt", "") or "").strip()
    if gdino_cfg:
        gdino_cfg_resolved = _resolve_repo_root_path(gdino_cfg, base_dir=base_dir)
        if not Path(gdino_cfg_resolved).exists():
            raise FileNotFoundError(
                f"GroundingDINO config not found: {gdino_cfg} -> {gdino_cfg_resolved}"
            )
        os.environ["GROUNDING_DINO_CFG"] = gdino_cfg_resolved
    if gdino_ckpt:
        gdino_ckpt_resolved = _resolve_repo_root_path(gdino_ckpt, base_dir=base_dir)
        if not Path(gdino_ckpt_resolved).exists():
            raise FileNotFoundError(
                f"GroundingDINO checkpoint not found: {gdino_ckpt} -> {gdino_ckpt_resolved}"
            )
        os.environ["GROUNDING_DINO_WEIGHTS"] = gdino_ckpt_resolved

    
    sam2_cfg_name = str(cfg_subject.get("sam2_cfg_name", "") or "").strip()
    sam2_ckpt = str(cfg_subject.get("sam2_ckpt", "") or "").strip()
    if sam2_cfg_name:
        os.environ["SAM2_CFG_NAME"] = sam2_cfg_name
    if sam2_ckpt:
        sam2_ckpt_resolved = _resolve_repo_root_path(sam2_ckpt, base_dir=base_dir)
        if not Path(sam2_ckpt_resolved).exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {sam2_ckpt} -> {sam2_ckpt_resolved}"
            )
        os.environ["SAM2_CKPT"] = sam2_ckpt_resolved


def build_subject_masks_single(
    frames_bgr: List[np.ndarray],
    ref_path: str,
    cfg_subject: Dict[str, Any],
    *,
    sample_id: str,
    base_dir: Optional[str] = None,
    semantic_root: Optional[str] = None,
) -> Optional[List[Optional[np.ndarray]]]:
    if not bool(cfg_subject.get("enable", True)):
        return None

    _set_env_from_cfg(cfg_subject, base_dir=base_dir)
    semantic_root_resolved = _resolve_semantic_root(
        cfg_subject,
        semantic_root=semantic_root,
        base_dir=base_dir,
    )

    
    prompts = _gather_text_prompts(
        sample_id=sample_id,
        ref_path=ref_path,
        cfg_prompts=cfg_subject.get("text_prompts", None),
        semantic_root=semantic_root_resolved,
    )
    if not prompts:
        raise RuntimeError(
            f"[subject_mask_utils] Empty prompt set for sample_id={sample_id!r} "
            f"(semantic_root={semantic_root_resolved!r}, ref_path={ref_path!r})"
        )

    
    box_conf_thr = float(cfg_subject.get("box_conf_thr", 0.35))
    text_thr = float(cfg_subject.get("text_thr", 0.25))
    topk = int(cfg_subject.get("topk_instances", 3))
    boxes_per_frame = _detect_boxes_grounding_dino(
        frames_bgr=frames_bgr,
        text_prompts=prompts,
        box_conf_thr=box_conf_thr,
        text_thr=text_thr,
        topk_instances=topk,
    )

    
    masks_raw = _segment_video_sam2_with_boxes(
        frames_bgr=frames_bgr,
        per_frame_boxes=boxes_per_frame,
    )
    if masks_raw is None:
        return None

    
    erode = int(cfg_subject.get("post_erode", 1))
    dilate = int(cfg_subject.get("post_dilate", 2))
    fill_ratio = float(cfg_subject.get("post_fill_ratio", 0.005))
    masks_fg: List[Optional[np.ndarray]] = []
    for m in masks_raw:
        if m is None:
            masks_fg.append(None)
        else:
            masks_fg.append(
                _postprocess_mask(
                    m,
                    erode=erode,
                    dilate=dilate,
                    fill_hole_ratio=fill_ratio,
                )
            )
    return masks_fg


def make_bg_complement(mask_fg: np.ndarray) -> Optional[np.ndarray]:
    if mask_fg is None:
        return None
    fg = mask_fg.astype(bool)
    if fg.size == 0 or int(fg.sum()) == 0:
        return None
    return np.logical_not(fg)


def build_fg_bg_masks_single(
    frames_bgr: List[np.ndarray],
    ref_path: str,
    cfg_subject: Dict[str, Any],
    *,
    sample_id: str,
    base_dir: Optional[str] = None,
    semantic_root: Optional[str] = None,
) -> Optional[Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]]:
    semantic_root_resolved = _resolve_semantic_root(
        cfg_subject,
        semantic_root=semantic_root,
        base_dir=base_dir,
    )
    masks_fg = build_subject_masks_single(
        frames_bgr,
        ref_path,
        cfg_subject,
        sample_id=sample_id,
        base_dir=base_dir,
        semantic_root=semantic_root_resolved,
    )
    if masks_fg is None:
        return None

    masks_bg: List[Optional[np.ndarray]] = []
    for m in masks_fg:
        if m is None or int(m.sum()) == 0:
            masks_bg.append(None)
        else:
            masks_bg.append(make_bg_complement(m))
    return masks_fg, masks_bg


def repair_masks_temporal(
    masks_fg: list,
    masks_bg: list,
    *,
    enable: bool = True,
    max_skip: int = 3,          
    dilate_iters: int = 0,      
    min_area_px: int = 0,       
):
    if not enable:
        recomputed_bg = [
            None if (m is None or int(m.sum()) == 0) else make_bg_complement(m)
            for m in masks_fg
        ]
        return masks_fg, recomputed_bg, {
            "fixed_fg": 0,
            "fixed_bg": 0,
            "skipped_fg": sum(m is None for m in masks_fg),
            "skipped_bg": sum(m is None for m in recomputed_bg),
        }

    def _dilate_bool(mask_bool, it):
        if it <= 0:
            return mask_bool
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * it + 1, 2 * it + 1))
        out = cv.dilate(mask_bool.astype(np.uint8) * 255, k, iterations=1) > 0
        return out

    T = len(masks_fg)
    if T != len(masks_bg):
        raise ValueError("fg/bg mask length mismatch")

    out_fg = list(masks_fg)

    valid_fg = [i for i, m in enumerate(out_fg) if (m is not None and int(m.sum()) > 0)]

    def _nearest(valid_list, t):
        if not valid_list:
            return None
        return min(valid_list, key=lambda j: abs(j - t))

    fixed_fg = 0

    for t in range(T):
        if out_fg[t] is None or int(out_fg[t].sum()) == 0:
            j = _nearest(valid_fg, t)
            if j is not None and abs(j - t) <= max_skip and out_fg[j] is not None:
                f = out_fg[j].copy()
                if dilate_iters > 0:
                    f = _dilate_bool(f, dilate_iters)
                if (min_area_px > 0) and (int(f.sum()) < min_area_px):
                    f = _dilate_bool(f, 1)
                out_fg[t] = f
                fixed_fg += 1

    out_bg = [
        None if (m is None or int(m.sum()) == 0) else make_bg_complement(m)
        for m in out_fg
    ]

    diag = {
        "fixed_fg": int(fixed_fg),
        "fixed_bg": 0,
        "skipped_fg": int(sum(m is None for m in out_fg)),
        "skipped_bg": int(sum(m is None for m in out_bg)),
    }
    return out_fg, out_bg, diag
