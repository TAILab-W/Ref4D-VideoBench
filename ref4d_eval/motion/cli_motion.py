

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from ref4d_eval.motion.preprocess.io_video import load_video_cv2, resample_video
from ref4d_eval.motion.core.subject_mask_utils import build_fg_bg_masks_single, repair_masks_temporal
from ref4d_eval.motion.core.seeds import sample_fg_bg_points
from ref4d_eval.motion.track_ate.tapir_infer import track_points_tapir
from ref4d_eval.motion.core.features import MotionFeaturePack, compute_all as compute_motion_features
from ref4d_eval.motion.core.metrics import (
    distances as compute_motion_distances,
    to_scores as motion_to_scores,
    compute_rf_ls,
    build_motion_atomic_features,
)
from ref4d_eval.motion.core.aggregator import predict_motion_score


def _load_and_resample(path: str, short_side: int, fps: int) -> List[np.ndarray]:
    frames, fps_src = load_video_cv2(path, bgr=True, return_fps=True)
    return resample_video(frames, short_side=short_side, fps=fps, src_fps=fps_src)


def _ensure_bool_list(lst: Sequence[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
    return [None if m is None else np.asarray(m, dtype=bool) for m in lst]



def _run_tapir_grouped(
    frames: List[np.ndarray],
    seeds_xy: np.ndarray,
    seeds_t0: np.ndarray,
    cfg_tapir: Dict[str, Any],
    base_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(frames)
    if seeds_xy.shape[0] != seeds_t0.shape[0]:
        raise ValueError("seeds_xy and seeds_t0 length mismatch")

    N = int(seeds_xy.shape[0])
    tracks = np.zeros((N, T, 2), np.float32)
    vis = np.zeros((N, T), dtype=bool)
    if N == 0:
        return tracks, vis

    uniq_t0 = np.unique(seeds_t0.astype(int))
    for t in uniq_t0:
        idx = np.where(seeds_t0 == t)[0]
        if idx.size == 0:
            continue
        xy = seeds_xy[idx]
        sub_frames = frames[t:]
        tr_sub, vs_sub = track_points_tapir(sub_frames, xy, cfg_tapir, base_dir=base_dir)
        tracks[idx, t:t + tr_sub.shape[1], :] = tr_sub
        vis[idx, t:t + vs_sub.shape[1]] = vs_sub
        tracks[idx, t, :] = xy
        vis[idx, t] = True
    return tracks, vis



def _unwrap_object_scalar(value: Any) -> Any:
    arr = value
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.ndim == 0:
            return arr.item()
        flat = arr.reshape(-1)
        if flat.size == 1:
            return flat[0].item() if isinstance(flat[0], np.ndarray) else flat[0]
    return value



def _resolve_cache_root(base_dir: str, cache_root: str) -> str:
    if os.path.isabs(cache_root):
        return cache_root
    return os.path.abspath(os.path.join(base_dir, cache_root))



def _resolve_cfg_path(base_dir: str, cfg_path: str) -> str:
    if os.path.isabs(cfg_path):
        return cfg_path
    return os.path.abspath(os.path.join(base_dir, cfg_path))



def _resolve_gen_path(base_dir: str, gen_path: str) -> str:
    if os.path.isabs(gen_path):
        return gen_path
    return os.path.abspath(os.path.join(base_dir, gen_path))



def _path_for_output(path: str, base_dir: str, fallback: Optional[str] = None) -> str:
    abs_path = os.path.abspath(path)
    try:
        return os.path.relpath(abs_path, base_dir)
    except ValueError:
        return fallback if fallback is not None else abs_path



def _load_ref_cache(sample_id: str, cache_root: str) -> Dict[str, Any]:
    npz_path = os.path.join(cache_root, f"{sample_id}.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"[ref.cache] missing cache for sample_id={sample_id}: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    required = [
        "schema_version", "r", "s", "theta", "valid_t", "hof", "acc", "jerk",
        "phi_stats", "n_fg_valid", "n_bg_valid", "motion_ref_meta",
    ]
    for k in required:
        if k not in data.files:
            raise KeyError(f"[ref.cache] {npz_path} missing key '{k}'")

    schema_version = _unwrap_object_scalar(data["schema_version"])
    if isinstance(schema_version, bytes):
        schema_version = schema_version.decode("utf-8", errors="replace")
    schema_version = str(schema_version)
    if schema_version != "motion_ref_cache":
        raise ValueError(
            f"[ref.cache] unsupported schema_version for sample_id={sample_id}: "
            f"{schema_version!r} (expected 'motion_ref_cache')"
        )

    phi_stats = _unwrap_object_scalar(data["phi_stats"])
    motion_ref_meta = _unwrap_object_scalar(data["motion_ref_meta"])
    if not isinstance(phi_stats, Mapping):
        raise ValueError(f"[ref.cache] invalid phi_stats for sample_id={sample_id}")
    if not isinstance(motion_ref_meta, Mapping):
        raise ValueError(f"[ref.cache] invalid motion_ref_meta for sample_id={sample_id}")

    pack_ref = MotionFeaturePack(
        r=np.asarray(data["r"]),
        s=np.asarray(data["s"]),
        theta=np.asarray(data["theta"]),
        valid_t=np.asarray(data["valid_t"], dtype=bool),
        hof=np.asarray(data["hof"]),
        acc=np.asarray(data["acc"]),
        jerk=np.asarray(data["jerk"]),
        phi_stats=dict(phi_stats),
        n_fg_valid=np.asarray(data["n_fg_valid"]),
        n_bg_valid=np.asarray(data["n_bg_valid"]),
    )

    return {
        "pack": pack_ref,
        "motion_ref_meta": dict(motion_ref_meta),
        "cache_path": npz_path,
    }



def main() -> None:
    ap = argparse.ArgumentParser("Single-sample motion evaluation CLI")
    ap.add_argument("--sample-id", "-sample-id", required=True, type=str, help="reference cache key / sample_id")
    ap.add_argument(
        "--gen",
        "-gen",
        required=True,
        type=str,
        help="generated video path (absolute, or relative to --base)",
    )
    ap.add_argument("--cfg", "-cfg", required=True, type=str, help="motion YAML config")
    ap.add_argument("--base", "-base", required=True, type=str, help="repo base (Ref4D-VideoBench)")
    ap.add_argument("--out", "-out", required=True, type=str, help="output JSON path")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base)
    cfg_path = _resolve_cfg_path(base_dir, args.cfg)
    gen_path_abs = _resolve_gen_path(base_dir, args.gen)

    with open(cfg_path, "r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)

    ref_cfg = cfg.get("ref", {}) or {}
    cache_root_raw = ref_cfg.get("cache_root", None)
    if not cache_root_raw:
        raise ValueError("missing required config: ref.cache_root")
    cache_root = _resolve_cache_root(base_dir, str(cache_root_raw))

    sample_id = str(args.sample_id)
    ref_cache = _load_ref_cache(sample_id, cache_root)
    pack_ref: MotionFeaturePack = ref_cache["pack"]
    motion_ref_meta: Dict[str, Any] = ref_cache["motion_ref_meta"]

    sample_cfg = cfg.get("sample", {}) or {}
    ss = int(sample_cfg.get("short_side", 448))
    fps = int(sample_cfg.get("fps", 8))

    cfg_subject = cfg.get("subject", {}) or {}
    tr = cfg.get("tracking", {}) or {}
    num_fg = int(tr.get("num_fg", 128))
    num_bg = int(tr.get("num_bg", 128))
    border = int(tr.get("border", 2))
    edge_bonus = bool(tr.get("edge_bonus", True))
    min_tex = tr.get("min_tex", None)
    min_tex = float(min_tex) if min_tex is not None else None
    t0_stride = int(tr.get("t0_stride", 8))
    t0_offset = int(tr.get("t0_offset", 0))
    seed = int(tr.get("seed", 2025))

    fallback_mask_cfg = (cfg.get("fallback", {}) or {}).get("mask", {}) or {}
    do_repair = bool(fallback_mask_cfg.get("enable", True))
    rep_max_skip = int(fallback_mask_cfg.get("max_skip", 6))
    rep_dilate_it = int(fallback_mask_cfg.get("dilate_iters", 1))
    rep_min_area = int(fallback_mask_cfg.get("min_area_px", 200))

    cfg_tapir = cfg.get("tapir", {}) or {}

    feat_cfg = cfg.get("features", {}) or {}
    dir_bins = int(feat_cfg.get("dir_bins", 8))
    min_speed_for_dir = float(feat_cfg.get("min_speed_for_dir", 0.05))

    motion_metric_cfg = cfg.get("motion_metrics", {}) or {}
    tau_s_quantile_ref = float(motion_metric_cfg.get("tau_s_quantile_ref", 0.40))
    ncc_patch = int(motion_metric_cfg.get("ncc_patch", 32))
    ncc_stride = motion_metric_cfg.get("ncc_stride", 32)
    ncc_stride = int(ncc_stride) if ncc_stride is not None else None
    ncc_thr = float(motion_metric_cfg.get("ncc_thr", 0.90))
    unique_min_ratio = float(motion_metric_cfg.get("unique_min_ratio", 0.15))
    roi_bbox_expand = int(motion_metric_cfg.get("roi_bbox_expand", 0))
    roi_on_empty = str(motion_metric_cfg.get("roi_on_empty", "union"))
    w_acc = float(motion_metric_cfg.get("smoothness_acc_weight", 0.3))
    w_jerk = float(motion_metric_cfg.get("smoothness_jerk_weight", 0.7))
    d_mag_winsor_quantile_ref_raw = motion_metric_cfg.get("d_mag_winsor_quantile_ref", None)
    d_mag_winsor_quantile_ref = (
        None
        if d_mag_winsor_quantile_ref_raw is None
        else float(d_mag_winsor_quantile_ref_raw)
    )

    gen_path_out = _path_for_output(gen_path_abs, base_dir)
    ref_cache_path_out = _path_for_output(ref_cache["cache_path"], base_dir)

    out: Dict[str, Any] = {
        "sample_id": sample_id,
        "gen": gen_path_out,
        "D_dir": float("nan"),
        "D_mag": float("nan"),
        "D_smo": float("nan"),
        "valid_dir": False,
        "valid_mag": False,
        "valid_smo": False,
        "S_dir": float("nan"),
        "S_mag": float("nan"),
        "S_smo": float("nan"),
        "RF": float("nan"),
        "LS": float("nan"),
        "valid_rf": False,
        "valid_ls": False,
        "is_valid_motion": False,
        "motion_score": float("nan"),
        "motion_score_0_100": float("nan"),
        "error": "",
        "meta": {
            "sample_id": sample_id,
            "ref_cache_key": sample_id,
            "ref_cache_path": ref_cache_path_out,
            "gen": gen_path_out,
            "short_side": ss,
            "fps": fps,
        },
    }

    try:
        frames_gen = _load_and_resample(gen_path_abs, ss, fps)

        masks = build_fg_bg_masks_single(
            frames_gen,
            gen_path_abs,
            cfg_subject,
            sample_id=sample_id,
            base_dir=base_dir,
            semantic_root=cfg_subject.get("semantic_root"),
        )
        if masks is None:
            raise RuntimeError("mask-build failed: build_fg_bg_masks_single returned None")
        masks_fg_gen, masks_bg_gen = masks
        masks_fg_gen = _ensure_bool_list(masks_fg_gen)
        masks_bg_gen = _ensure_bool_list(masks_bg_gen)
        if do_repair:
            masks_fg_gen, masks_bg_gen, _ = repair_masks_temporal(
                masks_fg_gen,
                masks_bg_gen,
                max_skip=rep_max_skip,
                dilate_iters=rep_dilate_it,
                min_area_px=rep_min_area,
            )

        seeds_gen = sample_fg_bg_points(
            masks_fg_gen,
            masks_bg_gen,
            num_fg=num_fg,
            num_bg=num_bg,
            border=border,
            edge_bonus=edge_bonus,
            min_tex=min_tex,
            frames_bgr=frames_gen,
            seed=seed,
            t0_stride=t0_stride,
            t0_offset=t0_offset,
        )
        tracks_fg_gen, vis_fg_gen = _run_tapir_grouped(frames_gen, seeds_gen["fg_xy"], seeds_gen["fg_t0"], cfg_tapir, base_dir=base_dir)
        tracks_bg_gen, vis_bg_gen = _run_tapir_grouped(frames_gen, seeds_gen["bg_xy"], seeds_gen["bg_t0"], cfg_tapir, base_dir=base_dir)

        _, pack_gen = compute_motion_features(
            tracks_fg_gen,
            vis_fg_gen,
            tracks_bg_gen,
            vis_bg_gen,
            dir_bins=dir_bins,
            min_speed_for_dir=min_speed_for_dir,
        )

        D = compute_motion_distances(
            {
                "hof": pack_ref.hof,
                "s": pack_ref.s,
                "acc": pack_ref.acc,
                "jerk": pack_ref.jerk,
                "valid_t": pack_ref.valid_t,
                "phi_stats": pack_ref.phi_stats,
            },
            {
                "hof": pack_gen.hof,
                "s": pack_gen.s,
                "acc": pack_gen.acc,
                "jerk": pack_gen.jerk,
                "valid_t": pack_gen.valid_t,
                "phi_stats": pack_gen.phi_stats,
            },
            w_acc=w_acc,
            w_jerk=w_jerk,
            d_mag_winsor_quantile_ref=d_mag_winsor_quantile_ref,
        )
        S = motion_to_scores(D)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    else:
        out.update({
            "D_dir": float(D.get("D_dir", np.nan)),
            "D_mag": float(D.get("D_mag", np.nan)),
            "D_smo": float(D.get("D_smo", np.nan)),
            "valid_dir": bool(S.get("valid_dir", False)),
            "valid_mag": bool(S.get("valid_mag", False)),
            "valid_smo": bool(S.get("valid_smo", False)),
            "S_dir": float(S.get("S_dir", np.nan)),
            "S_mag": float(S.get("S_mag", np.nan)),
            "S_smo": float(S.get("S_smo", np.nan)),
        })

        rf_err = ""
        try:
            rf, ls, _diag = compute_rf_ls(
                frames_gen,
                pack_gen.s,
                ref_meta=motion_ref_meta,
                tau_s_quantile_ref=tau_s_quantile_ref,
                ncc_patch=ncc_patch,
                ncc_stride=ncc_stride,
                ncc_thr=ncc_thr,
                unique_min_ratio=unique_min_ratio,
                roi_masks_gen=masks_fg_gen,
                roi_bbox_expand=roi_bbox_expand,
                roi_on_empty=roi_on_empty,
            )
        except Exception as e:
            rf = float("nan")
            ls = float("nan")
            rf_err = f"rf_ls invalid: {type(e).__name__}: {e}"

        atomic = build_motion_atomic_features(S, rf, ls)
        agg = predict_motion_score(atomic, cfg)

        out.update({
            "RF": float(atomic.get("RF", np.nan)),
            "LS": float(atomic.get("LS", np.nan)),
            "valid_rf": bool(atomic.get("valid_rf", False)),
            "valid_ls": bool(atomic.get("valid_ls", False)),
            "is_valid_motion": bool(agg.is_valid),
            "motion_score": float(agg.motion_score),
            "motion_score_0_100": float(agg.motion_score_0_100),
            "error": rf_err or str(agg.error or ""),
        })

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as h:
        json.dump(out, h, ensure_ascii=False, indent=2)

    print(json.dumps({
        "sample_id": sample_id,
        "is_valid_motion": out["is_valid_motion"],
        "motion_score": out["motion_score"],
        "motion_score_0_100": out["motion_score_0_100"],
        "error": out["error"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
