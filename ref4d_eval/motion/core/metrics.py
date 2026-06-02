

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2 as cv

__all__ = [
    "distances",
    "to_scores",
    "rf_unique_ratio_ncc",
    "compute_rf_ls",
    "compute_motion_ref_meta",
    "build_motion_atomic_features",
]






def _prepare_hist(h: Any) -> Optional[np.ndarray]:
    arr = np.asarray(h, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if np.any(arr < 0):
        return None
    s = float(arr.sum())
    if s <= 0.0:
        return None
    return (arr / s).astype(np.float64)


def _emd_1d_unit(h1: np.ndarray, h2: np.ndarray) -> float:
    diff = np.cumsum(h1 - h2)
    return float(np.sum(np.abs(diff)))


def _emd_circular_1d_unit(h1: np.ndarray, h2: np.ndarray) -> float:
    diff = np.asarray(h1, dtype=np.float64) - np.asarray(h2, dtype=np.float64)
    flow_prefix = np.cumsum(diff)
    shift = float(np.median(flow_prefix))
    return float(np.sum(np.abs(flow_prefix - shift)))


def _hist_distance_1d(h1: Any, h2: Any) -> Tuple[float, bool]:
    p1 = _prepare_hist(h1)
    p2 = _prepare_hist(h2)
    if p1 is None or p2 is None:
        return float("nan"), False
    if p1.shape != p2.shape:
        raise ValueError(f"histogram shape mismatch: {p1.shape} vs {p2.shape}")
    return _emd_1d_unit(p1, p2), True


def _hist_distance_circular(h1: Any, h2: Any) -> Tuple[float, bool]:
    p1 = _prepare_hist(h1)
    p2 = _prepare_hist(h2)
    if p1 is None or p2 is None:
        return float("nan"), False
    if p1.shape != p2.shape:
        raise ValueError(f"histogram shape mismatch: {p1.shape} vs {p2.shape}")
    return _emd_circular_1d_unit(p1, p2), True


def _finite_samples(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _finite_vector_norm_samples(x: Any) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.empty((0,), dtype=np.float64), False
    finite_rows = np.all(np.isfinite(arr), axis=1)
    if not np.any(finite_rows):
        return np.empty((0,), dtype=np.float64), True
    return np.linalg.norm(arr[finite_rows], axis=1).astype(np.float64), True


def _wasserstein_1d_from_samples(x: Any, y: Any) -> Tuple[float, bool]:
    xs = _finite_samples(x)
    ys = _finite_samples(y)
    if xs.size == 0 or ys.size == 0:
        return float("nan"), False

    xs = np.sort(xs)
    ys = np.sort(ys)
    n = max(xs.size, ys.size)
    q = np.linspace(0.0, 1.0, n, endpoint=True)
    xq = np.quantile(xs, q, method="linear")
    yq = np.quantile(ys, q, method="linear")
    return float(np.mean(np.abs(xq - yq))), True






def distances(
    feats_ref: Dict[str, Any],
    feats_gen: Dict[str, Any],
    *,
    w_acc: float = 0.3,
    w_jerk: float = 0.7,
    d_mag_winsor_quantile_ref: Optional[float] = None,
) -> Dict[str, Any]:
    required = ("hof", "s", "acc", "jerk")
    missing_ref = [k for k in required if k not in feats_ref]
    missing_gen = [k for k in required if k not in feats_gen]
    if missing_ref:
        raise KeyError(f"missing required ref features: {missing_ref}")
    if missing_gen:
        raise KeyError(f"missing required gen features: {missing_gen}")

    
    D_dir, valid_dir = _hist_distance_circular(feats_ref["hof"], feats_gen["hof"])

    s_ref = _finite_samples(feats_ref["s"])
    s_gen = _finite_samples(feats_gen["s"])
    if d_mag_winsor_quantile_ref is not None and s_ref.size > 0 and s_gen.size > 0:
        q = float(d_mag_winsor_quantile_ref)
        if not (0.0 < q < 1.0):
            raise ValueError("d_mag_winsor_quantile_ref must be between 0 and 1")
        cap = float(np.quantile(s_ref, q))
        s_ref = np.clip(s_ref, None, cap)
        s_gen = np.clip(s_gen, None, cap)
    D_mag, valid_mag = _wasserstein_1d_from_samples(s_ref, s_gen)

    
    acc_ref, acc_ref_shape_valid = _finite_vector_norm_samples(feats_ref["acc"])
    acc_gen, acc_gen_shape_valid = _finite_vector_norm_samples(feats_gen["acc"])
    jerk_ref, jerk_ref_shape_valid = _finite_vector_norm_samples(feats_ref["jerk"])
    jerk_gen, jerk_gen_shape_valid = _finite_vector_norm_samples(feats_gen["jerk"])

    if not acc_ref_shape_valid or not acc_gen_shape_valid:
        raise ValueError("acc must have shape [T, 2]")
    if not jerk_ref_shape_valid or not jerk_gen_shape_valid:
        raise ValueError("jerk must have shape [T, 2]")

    D_acc, valid_acc = _wasserstein_1d_from_samples(acc_ref, acc_gen)
    D_jerk, valid_jerk = _wasserstein_1d_from_samples(jerk_ref, jerk_gen)
    if valid_acc and valid_jerk:
        D_smo = float(w_acc) * float(D_acc) + float(w_jerk) * float(D_jerk)
        valid_smo = True
    else:
        D_smo = float("nan")
        valid_smo = False

    return {
        "D_dir": float(D_dir) if valid_dir else float("nan"),
        "D_mag": float(D_mag) if valid_mag else float("nan"),
        "D_smo": float(D_smo) if valid_smo else float("nan"),
        "valid_dir": bool(valid_dir),
        "valid_mag": bool(valid_mag),
        "valid_smo": bool(valid_smo),
        "valid_acc": bool(valid_acc),
        "valid_jerk": bool(valid_jerk),
    }


def to_scores(D: Dict[str, Any]) -> Dict[str, Any]:
    def _score(name: str, valid_name: str, score_name: str) -> Tuple[float, bool]:
        valid = bool(D.get(valid_name, False))
        val = D.get(name, float("nan"))
        if (not valid) or (not np.isfinite(val)):
            return float("nan"), False
        return float(np.exp(-float(val))), True

    s_dir, v_dir = _score("D_dir", "valid_dir", "S_dir")
    s_mag, v_mag = _score("D_mag", "valid_mag", "S_mag")
    s_smo, v_smo = _score("D_smo", "valid_smo", "S_smo")

    return {
        "S_dir": s_dir,
        "S_mag": s_mag,
        "S_smo": s_smo,
        "valid_dir": v_dir,
        "valid_mag": v_mag,
        "valid_smo": v_smo,
    }


def build_motion_atomic_features(scores_dict: Dict[str, Any], rf: float, ls: float) -> Dict[str, Any]:
    valid_rf = bool(np.isfinite(rf))
    valid_ls = bool(np.isfinite(ls))
    return {
        "S_dir": float(scores_dict.get("S_dir", np.nan)),
        "S_mag": float(scores_dict.get("S_mag", np.nan)),
        "S_smo": float(scores_dict.get("S_smo", np.nan)),
        "RF": float(rf) if valid_rf else float("nan"),
        "LS": float(ls) if valid_ls else float("nan"),
        "valid_dir": bool(scores_dict.get("valid_dir", False)),
        "valid_mag": bool(scores_dict.get("valid_mag", False)),
        "valid_smo": bool(scores_dict.get("valid_smo", False)),
        "valid_rf": valid_rf,
        "valid_ls": valid_ls,
    }






def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[-1] == 3:
        g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        g = img
    else:
        raise ValueError("unexpected frame shape")
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8, copy=False)
    return g


def _ncc_patch(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    A = a.astype(np.float32)
    B = b.astype(np.float32)
    Am = A - A.mean()
    Bm = B - B.mean()

    ssA = float((Am * Am).sum())
    ssB = float((Bm * Bm).sum())

    if ssA <= float(eps) and ssB <= float(eps):
        return 1.0 if np.array_equal(a, b) else 0.0

    num = float((Am * Bm).sum())
    den = float(np.sqrt(ssA) * np.sqrt(ssB) + eps)
    return num / den


def _bbox_from_mask(m: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    if m is None:
        return None
    mm = np.asarray(m, dtype=bool)
    ys, xs = np.where(mm)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _intersect_bbox(
    a: Optional[Tuple[int, int, int, int]],
    b: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    if a is None or b is None:
        return None
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    y0, y1 = max(ay0, by0), min(ay1, by1)
    x0, x1 = max(ax0, bx0), min(ax1, bx1)
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, y1, x0, x1


def _union_bbox(
    a: Optional[Tuple[int, int, int, int]],
    b: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    return min(ay0, by0), max(ay1, by1), min(ax0, bx0), max(ax1, bx1)


def _expand_bbox(bb: Tuple[int, int, int, int], pad: int, H: int, W: int) -> Tuple[int, int, int, int]:
    if pad <= 0:
        return bb
    y0, y1, x0, x1 = bb
    return max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)


def rf_unique_ratio_ncc(
    frames_bgr: List[np.ndarray],
    *,
    patch: int = 32,
    stride: Optional[int] = None,
    ncc_thr: float = 0.90,
    unique_min_ratio: float = 0.15,
    roi_masks: Optional[List[Optional[np.ndarray]]] = None,
    roi_bbox_expand: int = 0,
    roi_on_empty: str = "union",  
) -> Tuple[float, List[float]]:
    if len(frames_bgr) < 2:
        raise ValueError("need at least 2 frames for RF")
    if roi_masks is not None and len(roi_masks) != len(frames_bgr):
        raise ValueError("len(roi_masks) must equal len(frames_bgr)")

    patch = int(patch)
    if patch <= 0:
        raise ValueError("patch must be positive")
    stride = int(stride or patch)

    uniq_ratios: List[float] = []
    for t in range(1, len(frames_bgr)):
        g0 = _to_gray_uint8(frames_bgr[t - 1])
        g1 = _to_gray_uint8(frames_bgr[t])
        H, W = g0.shape
        if g1.shape != (H, W):
            raise ValueError("frame size mismatch")

        if roi_masks is not None:
            bb0 = _bbox_from_mask(roi_masks[t - 1])
            bb1 = _bbox_from_mask(roi_masks[t])
            bb = _intersect_bbox(bb0, bb1)
            if bb is None:
                mode = str(roi_on_empty or "union").lower()
                if mode == "union":
                    bb = _union_bbox(bb0, bb1)
                elif mode == "use_curr":
                    bb = bb1 if bb1 is not None else bb0
                elif mode == "use_prev":
                    bb = bb0 if bb0 is not None else bb1
                elif mode == "full":
                    bb = (0, H, 0, W)
                elif mode == "skip":
                    continue
                else:
                    raise ValueError(f"invalid roi_on_empty: {roi_on_empty}")
                if bb is None:
                    bb = (0, H, 0, W)
            y0, y1, x0, x1 = _expand_bbox(bb, int(roi_bbox_expand), H, W)
        else:
            y0, y1, x0, x1 = 0, H, 0, W

        scores: List[float] = []
        if (y1 - y0) >= patch and (x1 - x0) >= patch:
            for y in range(y0, y1 - patch + 1, stride):
                for x in range(x0, x1 - patch + 1, stride):
                    a = g0[y:y + patch, x:x + patch]
                    b = g1[y:y + patch, x:x + patch]
                    scores.append(_ncc_patch(a, b))
        else:
            if H < patch or W < patch:
                raise ValueError(f"frame smaller than patch at t={t}: frame=({H},{W}), patch={patch}")
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            y = int(np.clip(cy - patch // 2, 0, max(0, H - patch)))
            x = int(np.clip(cx - patch // 2, 0, max(0, W - patch)))
            a = g0[y:y + patch, x:x + patch]
            b = g1[y:y + patch, x:x + patch]
            scores.append(_ncc_patch(a, b))

        if not scores:
            raise ValueError(f"no patches for NCC at t={t} (check patch/stride/ROI size)")

        unique_ratio = float(np.mean(np.asarray(scores, np.float32) < float(ncc_thr)))
        uniq_ratios.append(unique_ratio)

    if not uniq_ratios:
        raise ValueError("no valid frame pairs for RF")

    rf = float(np.mean(np.asarray(uniq_ratios, np.float32) < float(unique_min_ratio)))
    return rf, uniq_ratios


def compute_motion_ref_meta(
    s_ref: np.ndarray,
    *,
    tau_s_quantile_ref: float = 0.40,
) -> Dict[str, float]:
    s_ref_f = _finite_samples(s_ref)
    if s_ref_f.size == 0:
        raise ValueError("empty s_ref in compute_motion_ref_meta")

    q = float(tau_s_quantile_ref)
    tau_s = float(np.quantile(s_ref_f, q))
    return {
        "tau_s": tau_s,
        "tau_s_quantile_ref": q,
        "median_s_ref": float(np.median(s_ref_f)),
    }


def compute_rf_ls(
    frames_gen: List[np.ndarray],
    s_gen: np.ndarray,
    *,
    s_ref: Optional[np.ndarray] = None,
    ref_meta: Optional[Dict[str, Any]] = None,
    tau_s_quantile_ref: float = 0.40,
    ncc_patch: int = 32,
    ncc_stride: Optional[int] = None,
    ncc_thr: float = 0.90,
    unique_min_ratio: float = 0.15,
    roi_masks_gen: Optional[List[Optional[np.ndarray]]] = None,
    roi_bbox_expand: int = 0,
    roi_on_empty: str = "union",
) -> Tuple[float, float, Dict[str, float]]:
    if frames_gen is None or len(frames_gen) == 0:
        raise ValueError("empty frames_gen in compute_rf_ls")

    s_gen_f = _finite_samples(s_gen)
    if s_gen_f.size == 0:
        raise ValueError("empty s_gen in compute_rf_ls")

    if ref_meta is not None:
        if "tau_s" not in ref_meta:
            raise ValueError("ref_meta must contain 'tau_s'")
        tau_s = float(ref_meta["tau_s"])
        q = float(ref_meta.get("tau_s_quantile_ref", tau_s_quantile_ref))
    else:
        if s_ref is None:
            raise ValueError("either s_ref or ref_meta must be provided")
        ref_meta = compute_motion_ref_meta(s_ref, tau_s_quantile_ref=tau_s_quantile_ref)
        tau_s = float(ref_meta["tau_s"])
        q = float(ref_meta["tau_s_quantile_ref"])

    rf, uniq_ratios = rf_unique_ratio_ncc(
        frames_gen,
        patch=ncc_patch,
        stride=ncc_stride,
        ncc_thr=ncc_thr,
        unique_min_ratio=unique_min_ratio,
        roi_masks=roi_masks_gen,
        roi_bbox_expand=roi_bbox_expand,
        roi_on_empty=roi_on_empty,
    )
    ls = float(np.mean(s_gen_f < tau_s))

    diag = {
        "tau_s": tau_s,
        "tau_s_quantile_ref": q,
        "uniq_ratio_mean": float(np.mean(uniq_ratios)),
        "uniq_ratio_min": float(np.min(uniq_ratios)),
        "uniq_ratio_max": float(np.max(uniq_ratios)),
    }
    return rf, ls, diag
