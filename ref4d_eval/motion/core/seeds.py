
from __future__ import annotations

"""
Seed sampling helpers for the motion pipeline.

"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2 as cv

__all__ = ["sample_points", "sample_fg_bg_points"]


def _ensure_bool(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        m = m.astype(bool)
    return m


def _texture_score(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    gx = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    p95 = float(np.quantile(mag, 0.95)) if np.isfinite(mag).all() else 0.0
    if p95 > 1e-6:
        mag = np.clip(mag / p95, 0.0, 1.0)
    else:
        mag = np.zeros_like(mag, dtype=np.float32)
    return mag


def _apply_border(mask: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return mask
    H, W = mask.shape[:2]
    out = np.zeros_like(mask, dtype=bool)
    out[border:H - border, border:W - border] = mask[border:H - border, border:W - border]
    return out


def _edge_bonus_weights(mask: np.ndarray, edge_bonus: bool, w_edge: float = 2.0) -> np.ndarray:
    if not edge_bonus:
        return mask.astype(np.float32)
    m = mask.astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    eroded = cv.erode(m, k, iterations=1)
    edge = (m > 0) & (eroded == 0)
    w = mask.astype(np.float32)
    w[edge] *= float(w_edge)
    return w


def _weighted_choice(
    coords_xy: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    replace: bool = False,
) -> np.ndarray:
    N = int(coords_xy.shape[0])
    if (not replace) and k > N:
        raise ValueError(f"not enough candidate pixels: need {k}, have {N}")
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    if (not np.isfinite(s)) or s <= 0:
        raise ValueError("invalid weights for sampling (sum<=0 or non-finite)")
    p = w / s
    idx = rng.choice(N, size=int(k), replace=bool(replace), p=p)
    return coords_xy[idx]


def sample_points(
    mask: np.ndarray,
    N: int,
    *,
    border: int = 4,
    edge_bonus: bool = True,
    min_tex: Optional[float] = None,
    frame_bgr: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    seed: int = 2025,
    adaptive: bool = True,
    min_take: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(seed)
    if N <= 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)

    m_full = _ensure_bool(mask)
    if m_full.sum() == 0:
        raise ValueError("empty mask")

    if frame_bgr is not None:
        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        tex_map = _texture_score(gray)
    else:
        tex_map = np.ones_like(m_full, dtype=np.float32)

    stage_min = int(max(1, min_take)) if adaptive else 1

    def _build_pool(mask_stage: np.ndarray):
        ys, xs = np.nonzero(mask_stage)
        if ys.size == 0:
            return None
        coords_xy = np.stack([xs, ys], axis=1).astype(np.float32)
        w_edge_map = _edge_bonus_weights(mask_stage, edge_bonus=edge_bonus)
        w_pick = (
            w_edge_map[ys, xs].astype(np.float64)
            * (tex_map[ys, xs].astype(np.float64) + 1e-6)
        )
        s = float(w_pick.sum())
        if (not np.isfinite(s)) or s <= 0:
            w_pick = np.ones_like(w_pick, dtype=np.float64)
        wtex = tex_map[ys, xs].astype(np.float32)
        return coords_xy, w_pick, wtex

    m_border = _apply_border(m_full, border=border)

    stages = []
    if (min_tex is not None) and (frame_bgr is not None):
        stages.append(m_border & (tex_map > float(min_tex)))  
    else:
        stages.append(m_border)  
    stages.append(m_border)      
    stages.append(m_full)        

    last_pool = None
    for stage_mask in stages:
        pool = _build_pool(stage_mask)
        if pool is None:
            continue
        coords_xy, w_pick, wtex = pool
        avail = int(coords_xy.shape[0])
        last_pool = pool

        if avail < stage_min:
            continue
        if avail >= int(N):
            picked_xy = _weighted_choice(coords_xy, w_pick, int(N), rng, replace=False)
            
            coord_to_w = {tuple(map(float, xy)): w for xy, w in zip(coords_xy, wtex)}
            picked_w = np.asarray([coord_to_w[tuple(map(float, xy))] for xy in picked_xy], dtype=np.float32)
            return picked_xy.astype(np.float32), picked_w

    if last_pool is None:
        raise ValueError("no candidate pixels inside mask after all relaxations")

    coords_xy, w_pick, wtex = last_pool
    picked_xy = _weighted_choice(coords_xy, w_pick, int(N), rng, replace=True)
    coord_to_w = {tuple(map(float, xy)): w for xy, w in zip(coords_xy, wtex)}
    picked_w = np.asarray([coord_to_w[tuple(map(float, xy))] for xy in picked_xy], dtype=np.float32)
    return picked_xy.astype(np.float32), picked_w


def sample_fg_bg_points(
    masks_fg: List[Optional[np.ndarray]],
    masks_bg: List[Optional[np.ndarray]],
    *,
    num_fg: int,
    num_bg: int,
    border: int = 4,
    edge_bonus: bool = True,
    min_tex: Optional[float] = None,
    frames_bgr: Optional[List[np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
    seed: int = 2025,
    t0_stride: int = 1,
    t0_offset: int = 0,
    adaptive: bool = True,
    min_take_fg: int = 1,
    min_take_bg: int = 1,
) -> Dict[str, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(seed)
    if frames_bgr is None and min_tex is not None:
        raise ValueError("min_tex gating requires frames_bgr")
    assert len(masks_fg) == len(masks_bg), "masks_fg/masks_bg length mismatch"

    T = len(masks_fg)
    keyframes = [
        t for t in range(T)
        if ((t - int(t0_offset)) % max(1, int(t0_stride))) == 0
    ]
    if not keyframes:
        raise ValueError("no keyframes after applying t0_stride/t0_offset")

    fg_xy_list, fg_w_list, fg_t0_list = [], [], []
    bg_xy_list, bg_w_list, bg_t0_list = [], [], []

    skipped_none = 0
    skipped_empty = 0
    skipped_candidate = 0

    for t in keyframes:
        mfg = masks_fg[t]
        mbg = masks_bg[t]
        if (mfg is None) or (mbg is None):
            skipped_none += 1
            continue

        if _ensure_bool(mfg).sum() == 0 or _ensure_bool(mbg).sum() == 0:
            skipped_empty += 1
            continue

        f = frames_bgr[t] if frames_bgr is not None else None

        try:
            fg_xy, fg_wtex = sample_points(
                mfg,
                num_fg,
                border=border,
                edge_bonus=edge_bonus,
                min_tex=min_tex,
                frame_bgr=f,
                rng=rng,
                seed=seed,
                adaptive=adaptive,
                min_take=min_take_fg,
            )
            bg_xy, bg_wtex = sample_points(
                mbg,
                num_bg,
                border=border,
                edge_bonus=edge_bonus,
                min_tex=min_tex,
                frame_bgr=f,
                rng=rng,
                seed=seed,
                adaptive=adaptive,
                min_take=min_take_bg,
            )
        except ValueError:
            skipped_candidate += 1
            continue

        fg_xy_list.append(fg_xy)
        fg_w_list.append(fg_wtex)
        fg_t0_list.append(np.full((fg_xy.shape[0],), t, np.int32))

        bg_xy_list.append(bg_xy)
        bg_w_list.append(bg_wtex)
        bg_t0_list.append(np.full((bg_xy.shape[0],), t, np.int32))

    if (not fg_xy_list) and (not bg_xy_list):
        if skipped_none == len(keyframes):
            raise ValueError("all selected keyframes have None fg/bg masks")
        if skipped_empty + skipped_none == len(keyframes) and skipped_empty > 0 and skipped_candidate == 0:
            raise ValueError("all selected keyframes have empty fg/bg masks")
        if skipped_candidate + skipped_none + skipped_empty == len(keyframes):
            raise ValueError("no candidate pixels on selected keyframes after relaxations")
        raise ValueError("no seeds after keyframe filtering and mask validation")

    def _cat(lst, axis=0, dtype=None):
        if not lst:
            return np.zeros((0,), dtype=dtype if dtype is not None else np.float32)
        arr = np.concatenate(lst, axis=axis)
        return arr.astype(dtype) if dtype is not None else arr

    fg_xy = _cat(fg_xy_list, axis=0, dtype=np.float32)
    bg_xy = _cat(bg_xy_list, axis=0, dtype=np.float32)
    fg_w = _cat(fg_w_list, axis=0, dtype=np.float32)
    bg_w = _cat(bg_w_list, axis=0, dtype=np.float32)
    fg_t0 = _cat(fg_t0_list, axis=0, dtype=np.int32)
    bg_t0 = _cat(bg_t0_list, axis=0, dtype=np.int32)

    return {
        "fg_xy": fg_xy, "bg_xy": bg_xy,
        "fg_t0": fg_t0, "bg_t0": bg_t0,
        "fg_wtex": fg_w, "bg_wtex": bg_w,
    }
