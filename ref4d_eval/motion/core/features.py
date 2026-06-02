

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


__all__ = ["MotionFeaturePack", "compute_all"]


@dataclass
class MotionFeaturePack:
    
    r: np.ndarray            
    s: np.ndarray            
    theta: np.ndarray        
    valid_t: np.ndarray      

    
    hof: np.ndarray          

    
    acc: np.ndarray          
    jerk: np.ndarray         

    
    phi_stats: Dict[str, float]

    
    n_fg_valid: np.ndarray   
    n_bg_valid: np.ndarray   


def _safe_mean_2d(vecs: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    m = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if vecs.ndim != 2 or vecs.shape[1] != 2:
        raise ValueError("vecs must have shape [N,2]")
    if m.shape[0] != vecs.shape[0]:
        raise ValueError("valid_mask length mismatch")
    if not np.any(m):
        return np.full((2,), np.nan, dtype=np.float32)
    return np.mean(vecs[m], axis=0).astype(np.float32)


def _make_hof(theta_valid: np.ndarray, bins: int = 16) -> np.ndarray:
    th = np.asarray(theta_valid, dtype=np.float32).reshape(-1)
    th = th[np.isfinite(th)]
    if th.size == 0:
        return np.zeros((int(bins),), dtype=np.float32)
    edges = np.linspace(-np.pi, np.pi, num=int(bins) + 1, dtype=np.float32)
    hist, _ = np.histogram(th, bins=edges)
    h = hist.astype(np.float32)
    s = float(h.sum())
    return h if s <= 0 else (h / s)


def _iter_true_segments(mask: np.ndarray):
    m = np.asarray(mask, dtype=bool).reshape(-1)
    n = m.shape[0]
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and m[j + 1]:
            j += 1
        yield i, j
        i = j + 1


def _compute_acc_jerk_full(r: np.ndarray, valid_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    Tm1 = int(r.shape[0])
    acc = np.full((Tm1, 2), np.nan, dtype=np.float32)
    jerk = np.full((Tm1, 2), np.nan, dtype=np.float32)

    valid_seg_count = 0
    valid_step_count = int(np.sum(np.asarray(valid_t, dtype=bool)))

    for s_idx, e_idx in _iter_true_segments(valid_t):
        seg = r[s_idx:e_idx + 1]
        if seg.shape[0] == 0:
            continue
        valid_seg_count += 1
        if seg.shape[0] >= 2:
            seg_acc = np.diff(seg, axis=0).astype(np.float32)
            acc[s_idx + 1:e_idx + 1] = seg_acc
        if seg.shape[0] >= 3:
            seg_jerk = np.diff(np.diff(seg, axis=0), axis=0).astype(np.float32)
            jerk[s_idx + 2:e_idx + 1] = seg_jerk

    return acc, jerk, valid_seg_count, valid_step_count


def _summary_stats(acc: np.ndarray, jerk: np.ndarray, *, valid_seg_count: int, valid_step_count: int) -> Dict[str, float]:
    acc_mag = np.linalg.norm(acc, axis=1)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    acc_mag = acc_mag[np.isfinite(acc_mag)]
    jerk_mag = jerk_mag[np.isfinite(jerk_mag)]

    def _nan_stat(x: np.ndarray, kind: str) -> float:
        if x.size == 0:
            return float("nan")
        if kind == "mean":
            return float(np.mean(x))
        if kind == "median":
            return float(np.median(x))
        if kind == "p90":
            return float(np.quantile(x, 0.90))
        if kind == "std":
            return float(np.std(x))
        raise ValueError(f"unknown stat kind: {kind}")

    return {
        "acc_mag_mean": _nan_stat(acc_mag, "mean"),
        "acc_mag_median": _nan_stat(acc_mag, "median"),
        "acc_mag_p90": _nan_stat(acc_mag, "p90"),
        "jerk_mag_mean": _nan_stat(jerk_mag, "mean"),
        "jerk_mag_median": _nan_stat(jerk_mag, "median"),
        "jerk_mag_p90": _nan_stat(jerk_mag, "p90"),
        "jerk_mag_std": _nan_stat(jerk_mag, "std"),
        "valid_seg_count": float(valid_seg_count),
        "valid_step_count": float(valid_step_count),
    }


def compute_all(
    tracks_fg: np.ndarray,
    vis_fg: np.ndarray,
    tracks_bg: np.ndarray,
    vis_bg: np.ndarray,
    *,
    dir_bins: int = 16,
    min_speed_for_dir: float = 1e-6,
) -> Tuple[np.ndarray, MotionFeaturePack]:
    assert tracks_fg.ndim == 3 and tracks_bg.ndim == 3
    assert tracks_fg.shape[2] == 2 and tracks_bg.shape[2] == 2
    assert vis_fg.shape[:2] == tracks_fg.shape[:2]
    assert vis_bg.shape[:2] == tracks_bg.shape[:2]

    Nf, T, _ = tracks_fg.shape
    Nb, T2, _ = tracks_bg.shape
    assert T == T2 and T >= 2, "tracks time length mismatch or too short"

    vis_fg = np.asarray(vis_fg, dtype=bool)
    vis_bg = np.asarray(vis_bg, dtype=bool)

    
    u_fg = tracks_fg[:, 1:, :] - tracks_fg[:, :-1, :]   
    u_bg = tracks_bg[:, 1:, :] - tracks_bg[:, :-1, :]   
    m_fg = (vis_fg[:, 1:] & vis_fg[:, :-1])            
    m_bg = (vis_bg[:, 1:] & vis_bg[:, :-1])            

    n_fg_valid = np.sum(m_fg, axis=0).astype(np.int32)  
    n_bg_valid = np.sum(m_bg, axis=0).astype(np.int32)  
    valid_t = (n_fg_valid >= 1) & (n_bg_valid >= 1)     

    v_fg = np.full((T - 1, 2), np.nan, dtype=np.float32)
    v_bg = np.full((T - 1, 2), np.nan, dtype=np.float32)
    for t in range(T - 1):
        v_fg[t] = _safe_mean_2d(u_fg[:, t, :], m_fg[:, t])
        v_bg[t] = _safe_mean_2d(u_bg[:, t, :], m_bg[:, t])

    r = (v_fg - v_bg).astype(np.float32)               
    r[~valid_t] = np.nan

    s = np.linalg.norm(r, axis=1).astype(np.float32)   
    s[~valid_t] = np.nan

    theta = np.full((T - 1,), np.nan, dtype=np.float32)
    dir_valid = valid_t & np.isfinite(s) & (s >= float(min_speed_for_dir))
    if np.any(dir_valid):
        theta[dir_valid] = np.arctan2(r[dir_valid, 1], r[dir_valid, 0]).astype(np.float32)

    hof = _make_hof(theta[dir_valid], bins=dir_bins)   

    acc, jerk, valid_seg_count, valid_step_count = _compute_acc_jerk_full(r, valid_t)

    phi_stats = _summary_stats(acc, jerk, valid_seg_count=valid_seg_count, valid_step_count=valid_step_count)
    phi_stats.update({
        "min_speed_for_dir": float(min_speed_for_dir),
        "dir_bins": float(dir_bins),
    })

    pack = MotionFeaturePack(
        r=r,
        s=s,
        theta=theta,
        valid_t=valid_t.astype(bool),
        hof=hof,
        acc=acc,
        jerk=jerk,
        phi_stats=phi_stats,
        n_fg_valid=n_fg_valid,
        n_bg_valid=n_bg_valid,
    )
    return r, pack
