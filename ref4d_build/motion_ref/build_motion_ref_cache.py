

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

from ref4d_eval.motion.preprocess.io_video import load_video_cv2, resample_video
from ref4d_eval.motion.core.subject_mask_utils import build_fg_bg_masks_single, repair_masks_temporal
from ref4d_eval.motion.core.seeds import sample_fg_bg_points
from ref4d_eval.motion.track_ate.tapir_infer import track_points_tapir
from ref4d_eval.motion.core.features import MotionFeaturePack, compute_all as compute_motion_features
from ref4d_eval.motion.core.metrics import compute_motion_ref_meta

SCHEMA_VERSION = "motion_ref_cache"


def _load_and_resample(path: str, short_side: int, fps: int) -> List[np.ndarray]:
    frames, fps_src = load_video_cv2(path, bgr=True, return_fps=True)
    frames = resample_video(frames, short_side=short_side, fps=fps, src_fps=fps_src)
    return frames


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


def _resolve_existing_path(base_dir: str, raw: str) -> Optional[str]:
    p = Path(raw)
    candidates = [p] if p.is_absolute() else [Path(base_dir) / raw]
    for cand in candidates:
        if cand.is_file():
            return str(cand.resolve())
    return None


def _record_to_sid_and_path(
    base_dir: str,
    rec: Dict[str, Any],
    metadata_path: str,
    line_no: int,
) -> Tuple[str, str]:
    sid = rec.get("sample_id")
    if sid is None:
        raise ValueError(f"missing required field 'sample_id' at {metadata_path}:{line_no}")
    sid = str(sid).strip()
    if not sid:
        raise ValueError(f"empty 'sample_id' at {metadata_path}:{line_no}")

    raw = rec.get("ref_video")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"missing required field 'ref_video' at {metadata_path}:{line_no}")

    resolved = _resolve_existing_path(base_dir, raw.strip())
    if resolved is None:
        raise FileNotFoundError(
            f"ref_video does not exist at {metadata_path}:{line_no}: {raw.strip()}"
        )
    return sid, resolved

_VALID_CACHE_KEYS = {
    "schema_version",
    "r",
    "s",
    "theta",
    "valid_t",
    "hof",
    "acc",
    "jerk",
    "phi_stats",
    "n_fg_valid",
    "n_bg_valid",
    "motion_ref_meta",
}


def _read_schema_version(npz: np.lib.npyio.NpzFile) -> Optional[str]:
    if "schema_version" not in npz.files:
        return None
    value = npz["schema_version"]
    try:
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return str(value.item())
            if value.size == 1:
                return str(value.reshape(()).item())
        return str(value)
    except Exception:
        return None


def _is_valid_existing_cache(path: str) -> bool:
    try:
        with np.load(path, allow_pickle=True) as npz:
            keys = set(npz.files)
            if keys != _VALID_CACHE_KEYS:
                return False
            if _read_schema_version(npz) != SCHEMA_VERSION:
                return False
        return True
    except Exception:
        return False


def _load_ref_items(base_dir: str, cfg: Dict[str, Any]) -> Dict[str, str]:
    dataset_cfg = cfg.get("dataset", {}) or {}
    metadata_path = dataset_cfg.get("meta_path")
    if not isinstance(metadata_path, str) or not metadata_path.strip():
        raise ValueError("cfg.dataset.meta_path is required")
    metadata_path = metadata_path.strip()
    if not os.path.isabs(metadata_path):
        metadata_path = os.path.join(base_dir, metadata_path)
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    sid2ref: Dict[str, str] = {}
    sid2line: Dict[str, int] = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                raise ValueError(f"invalid JSONL at {metadata_path}:{line_no}: {e}") from e
            if not isinstance(rec, dict):
                raise ValueError(f"JSONL record must be an object at {metadata_path}:{line_no}")

            sid, path = _record_to_sid_and_path(base_dir, rec, metadata_path, line_no)
            prev = sid2ref.get(sid)
            if prev is not None:
                prev_line = sid2line.get(sid, "?")
                if prev == path:
                    raise ValueError(
                        f"duplicate sample_id={sid} at {metadata_path}:{line_no}; "
                        f"first seen at line {prev_line}"
                    )
                raise ValueError(
                    f"conflicting ref_video for sample_id={sid} at {metadata_path}:{line_no}: "
                    f"{prev} vs {path}"
                )
            sid2ref[sid] = path
            sid2line[sid] = line_no

    return sid2ref

def _save_motion_cache(
    out_path: str,
    pack_ref: MotionFeaturePack,
    motion_ref_meta: Dict[str, Any],
) -> None:
    np.savez_compressed(
        out_path,
        schema_version=np.array(SCHEMA_VERSION),
        r=pack_ref.r,
        s=pack_ref.s,
        theta=pack_ref.theta,
        valid_t=pack_ref.valid_t,
        hof=pack_ref.hof,
        acc=pack_ref.acc,
        jerk=pack_ref.jerk,
        phi_stats=np.array(pack_ref.phi_stats, dtype=object),
        n_fg_valid=pack_ref.n_fg_valid,
        n_bg_valid=pack_ref.n_bg_valid,
        motion_ref_meta=np.array(motion_ref_meta, dtype=object),
    )


def _process_one_ref(args: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        idx,
        total,
        sid,
        ref_path,
        out_root,
        ss,
        fps,
        cfg_subject,
        num_fg,
        num_bg,
        border,
        edge_bonus,
        min_tex,
        t0_stride,
        t0_offset,
        seed,
        base_dir,
        do_repair,
        rep_max_skip,
        rep_dilate_it,
        rep_min_area,
        cfg_tapir,
        dir_bins,
        min_speed_for_dir,
        tau_s_quantile_ref,
    ) = args

    out_path = os.path.join(out_root, f"{sid}.npz")

    try:
        if os.path.isfile(out_path):
            if _is_valid_existing_cache(out_path):
                print(f"[{idx}/{total}] skip_exists {sid}: {out_path}")
                return {"sid": sid, "status": "skip_exists"}
            print(f"[{idx}/{total}] rebuild_stale_cache {sid}: {out_path}")

        print(f"[{idx}/{total}] build {sid} -> {out_path}")

        try:
            frames_ref = _load_and_resample(ref_path, ss, fps)
        except Exception as e:
            raise RuntimeError(f"video-load failed: {e}") from e

        try:
            masks = build_fg_bg_masks_single(
                frames_ref,
                ref_path,
                cfg_subject,
                sample_id=sid,
                base_dir=base_dir,
                semantic_root=cfg_subject.get("semantic_root"),
            )
            if masks is None:
                raise RuntimeError("build_fg_bg_masks_single returned None")
            masks_fg_ref, masks_bg_ref = masks
            masks_fg_ref = _ensure_bool_list(masks_fg_ref)
            masks_bg_ref = _ensure_bool_list(masks_bg_ref)
            if do_repair:
                masks_fg_ref, masks_bg_ref, _ = repair_masks_temporal(
                    masks_fg_ref,
                    masks_bg_ref,
                    max_skip=rep_max_skip,
                    dilate_iters=rep_dilate_it,
                    min_area_px=rep_min_area,
                )
        except Exception as e:
            raise RuntimeError(f"mask-build failed: {e}") from e

        try:
            seeds_ref = sample_fg_bg_points(
                masks_fg_ref,
                masks_bg_ref,
                num_fg=num_fg,
                num_bg=num_bg,
                border=border,
                edge_bonus=edge_bonus,
                min_tex=min_tex,
                frames_bgr=frames_ref,
                seed=seed,
                t0_stride=t0_stride,
                t0_offset=t0_offset,
            )
        except Exception as e:
            raise RuntimeError(f"sampling failed: {e}") from e

        try:
            tracks_fg_ref, vis_fg_ref = _run_tapir_grouped(
                frames_ref,
                seeds_ref["fg_xy"],
                seeds_ref["fg_t0"],
                cfg_tapir,
                base_dir=base_dir,
            )
            tracks_bg_ref, vis_bg_ref = _run_tapir_grouped(
                frames_ref,
                seeds_ref["bg_xy"],
                seeds_ref["bg_t0"],
                cfg_tapir,
                base_dir=base_dir,
            )
        except Exception as e:
            raise RuntimeError(f"tapir failed: {e}") from e

        try:
            _, pack_ref = compute_motion_features(
                tracks_fg_ref,
                vis_fg_ref,
                tracks_bg_ref,
                vis_bg_ref,
                dir_bins=dir_bins,
                min_speed_for_dir=min_speed_for_dir,
            )
            motion_ref_meta = compute_motion_ref_meta(
                pack_ref.s,
                tau_s_quantile_ref=tau_s_quantile_ref,
            )
        except Exception as e:
            raise RuntimeError(f"feature-extract failed: {e}") from e

        try:
            _save_motion_cache(out_path, pack_ref, motion_ref_meta)
        except Exception as e:
            raise RuntimeError(f"save failed: {e}") from e

        return {"sid": sid, "status": "done"}

    except Exception as e:
        print(f"[{idx}/{total}] error {sid}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return {"sid": sid, "status": "error", "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser("Build reference-side motion evidence cache")
    ap.add_argument("--cfg", required=True, type=str)
    ap.add_argument("--base", required=True, type=str, help="repo base (Ref4D-VideoBench)")
    ap.add_argument("--limit", type=int, default=0, help="optional: only process first N samples")
    ap.add_argument("--workers", type=int, default=1, help="number of worker processes; 1 = single process")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base)

    cfg_path = args.cfg if os.path.isabs(args.cfg) else os.path.join(base_dir, args.cfg)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    ref_cfg = cfg.get("ref", {}) or {}
    cache_root_cfg = ref_cfg.get("cache_root")
    if not isinstance(cache_root_cfg, str) or not cache_root_cfg.strip():
        raise ValueError("cfg.ref.cache_root is required for building motion reference cache")
    cache_root_cfg = cache_root_cfg.strip()
    out_root = cache_root_cfg if os.path.isabs(cache_root_cfg) else os.path.join(base_dir, cache_root_cfg)
    os.makedirs(out_root, exist_ok=True)

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

    sid2ref = _load_ref_items(base_dir, cfg)
    if not sid2ref:
        raise RuntimeError("no reference videos resolved from cfg.dataset.meta_path")
    sids = sorted(sid2ref.keys())
    if args.limit > 0:
        sids = sids[: args.limit]

    total = len(sids)
    print(f"[build_motion_ref_cache] total refs: {total}, out_root={out_root} (from cfg.ref.cache_root)")

    jobs: List[Tuple[Any, ...]] = []
    for i, sid in enumerate(sids, start=1):
        jobs.append(
            (
                i,
                total,
                sid,
                sid2ref[sid],
                out_root,
                ss,
                fps,
                cfg_subject,
                num_fg,
                num_bg,
                border,
                edge_bonus,
                min_tex,
                t0_stride,
                t0_offset,
                seed,
                base_dir,
                do_repair,
                rep_max_skip,
                rep_dilate_it,
                rep_min_area,
                cfg_tapir,
                dir_bins,
                min_speed_for_dir,
                tau_s_quantile_ref,
            )
        )

    workers = int(args.workers)
    if workers <= 0:
        workers = max(1, os.cpu_count() or 1)

    done = 0
    skip_exists = 0
    error = 0

    if workers == 1:
        for job in jobs:
            ret = _process_one_ref(job)
            status = ret.get("status")
            if status == "done":
                done += 1
            elif status == "skip_exists":
                skip_exists += 1
            else:
                error += 1
    else:
        print(f"[build_motion_ref_cache] use multiprocessing with workers={workers}")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for ret in pool.imap_unordered(_process_one_ref, jobs):
                status = ret.get("status")
                if status == "done":
                    done += 1
                elif status == "skip_exists":
                    skip_exists += 1
                else:
                    error += 1

    print(
        f"[build_motion_ref_cache] done={done}, skip_exists={skip_exists}, error={error}, total={total}"
    )


if __name__ == "__main__":
    main()
