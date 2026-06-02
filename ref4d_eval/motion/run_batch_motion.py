

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
import traceback
import multiprocessing as mp
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import yaml


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

from ref4d_eval.motion.preprocess.io_video import load_video_cv2, resample_video
from ref4d_eval.motion.core.subject_mask_utils import build_fg_bg_masks_single, repair_masks_temporal
from ref4d_eval.motion.core.seeds import sample_fg_bg_points
from ref4d_eval.motion.track_ate.tapir_infer import track_points_tapir
from ref4d_eval.motion.core.features import compute_all as compute_motion_features, MotionFeaturePack
from ref4d_eval.motion.core.metrics import (
    distances as compute_motion_distances,
    to_scores as motion_to_scores,
    compute_rf_ls,
    build_motion_atomic_features,
)
from ref4d_eval.motion.core.aggregator import predict_motion_score


STD_HEADER: List[str] = [
    "modelname",
    "sample_id",
    "D_dir",
    "D_mag",
    "D_smo",
    "S_dir",
    "S_mag",
    "S_smo",
    "RF",
    "LS",
    "is_valid_motion",
    "motion_score",
    "motion_score_0_100",
    "error",
]


def _infer_sample_id(ref_key: str) -> str:
    basename = os.path.basename(str(ref_key))
    name, ext = os.path.splitext(basename)
    return name if ext else str(ref_key)


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


def _to_rel_output_path(path: str, base_dir: str) -> str:
    abs_path = os.path.abspath(path)
    base_abs = os.path.abspath(base_dir)
    try:
        rel = os.path.relpath(abs_path, base_abs)
    except Exception:
        return path
    if rel.startswith(".."):
        return path
    return rel


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


def _nan_result(error: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    bool_fields = {"is_valid_motion"}
    text_fields = {"modelname", "sample_id", "error"}
    for k in STD_HEADER:
        if k in text_fields:
            continue
        row[k] = False if k in bool_fields else float("nan")
    row["error"] = error
    return row


def _atomic_from_pair(ref_key: str, gen_path: str, cfg: Dict[str, Any], cache_ref: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ref_cfg = cfg.get("ref", {}) or {}
    cache_root_raw = ref_cfg.get("cache_root", None)
    if not cache_root_raw:
        raise ValueError("missing required config: ref.cache_root")
    base_dir = cfg.get("_base_dir", "")
    cache_root = _resolve_cache_root(str(base_dir), str(cache_root_raw))

    sid = _infer_sample_id(ref_key)

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

    if sid not in cache_ref:
        cache_ref[sid] = _load_ref_cache(sid, cache_root)
    pack_ref: MotionFeaturePack = cache_ref[sid]["pack"]
    motion_ref_meta: Dict[str, Any] = cache_ref[sid]["motion_ref_meta"]

    out = _nan_result("")

    frames_gen = _load_and_resample(gen_path, ss, fps)
    masks = build_fg_bg_masks_single(
        frames_gen,
        gen_path,
        cfg_subject,
        sample_id=sid,
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
    out.update({
        "RF": float(atomic.get("RF", np.nan)),
        "LS": float(atomic.get("LS", np.nan)),
        "valid_rf": bool(atomic.get("valid_rf", False)),
        "valid_ls": bool(atomic.get("valid_ls", False)),
    })

    try:
        agg = predict_motion_score(atomic, cfg)
        out.update({
            "is_valid_motion": bool(agg.is_valid),
            "motion_score": float(agg.motion_score),
            "motion_score_0_100": float(agg.motion_score_0_100),
            "error": rf_err or str(agg.error or ""),
        })
    except Exception as e:
        out.update({
            "is_valid_motion": False,
            "motion_score": float("nan"),
            "motion_score_0_100": float("nan"),
            "error": rf_err or f"{type(e).__name__}: {e}",
        })

    return out


def _scan_dataset_ref4d(
    base: str,
    meta_path: str,
    gen_video_root: str,
    models_filter: Optional[Set[str]] = None,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[Tuple[str, str]]]]:
    meta_file = meta_path if os.path.isabs(meta_path) else os.path.join(base, meta_path)

    sample_map: Dict[str, Dict[str, str]] = {}
    seen_lines: Dict[str, int] = {}
    with open(meta_file, "r", encoding="utf-8") as f:
        for line_no, ln in enumerate(f, start=1):
            ln = ln.strip()
            if not ln:
                continue
            js = json.loads(ln)
            sid = str(js["sample_id"])
            if sid in sample_map:
                raise ValueError(
                    f"duplicate sample_id={sid} in metadata {meta_file}:{line_no}; "
                    f"first seen at line {seen_lines[sid]}"
                )
            sample_map[sid] = {"ref": sid}
            seen_lines[sid] = line_no

    gen_root = gen_video_root if os.path.isabs(gen_video_root) else os.path.join(base, gen_video_root)
    model_map: Dict[str, List[Tuple[str, str]]] = {sid: [] for sid in sample_map.keys()}
    for model_dir in sorted(glob.glob(os.path.join(gen_root, "*"))):
        if not os.path.isdir(model_dir):
            continue
        modelname = os.path.basename(model_dir)
        if models_filter and modelname not in models_filter:
            continue
        for sid in list(sample_map.keys()):
            gp = os.path.join(model_dir, f"{sid}.mp4")
            if os.path.isfile(gp):
                model_map[sid].append((modelname, gp))

    sample_map = {sid: info for sid, info in sample_map.items() if model_map.get(sid)}
    if limit is not None:
        keep = set(sorted(sample_map.keys())[: max(0, int(limit))])
        sample_map = {sid: info for sid, info in sample_map.items() if sid in keep}
    model_map = {sid: lst for sid, lst in model_map.items() if lst}
    if limit is not None:
        model_map = {sid: lst for sid, lst in model_map.items() if sid in sample_map}
    return sample_map, model_map


def _worker_loop(
    wid: int,
    tasks: List[str],
    sample_map: Dict[str, Dict[str, str]],
    model_map: Dict[str, List[Tuple[str, str]]],
    cfg: Dict[str, Any],
    chunk_csv: str,
    log_txt: str,
    skip_pairs: Set[Tuple[str, str]],
) -> None:
    cache_ref: Dict[str, Dict[str, Any]] = {}
    chunk_dir = os.path.dirname(chunk_csv)
    if chunk_dir:
        os.makedirs(chunk_dir, exist_ok=True)
    with open(chunk_csv, "w", newline="", encoding="utf-8") as fw, open(log_txt, "w", encoding="utf-8") as flog:
        writer = csv.writer(fw)
        writer.writerow(STD_HEADER)
        for sid in tasks:
            ref_key = sample_map[sid]["ref"]
            for modelname, gen_path in model_map[sid]:
                if (modelname, sid) in skip_pairs:
                    print(f"[worker {wid}] skip cached {modelname}/{sid}", file=flog, flush=True)
                    continue

                row: Dict[str, Any] = {
                    "modelname": modelname,
                    "sample_id": sid,
                    "ref": ref_key,
                    "gen": _to_rel_output_path(gen_path, str(cfg.get("_base_dir", ""))),
                    **_nan_result(""),
                }
                t0 = time.time()
                try:
                    out = _atomic_from_pair(ref_key, gen_path, cfg, cache_ref)
                    row.update(out)
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"
                    traceback.print_exc(file=flog)
                writer.writerow([row.get(col, "") for col in STD_HEADER])
                fw.flush()
                dt = time.time() - t0
                print(f"[worker {wid}] {modelname}/{sid} done in {dt:.1f}s", file=flog, flush=True)


def _idx(header: List[str], name: str, default: Optional[int] = None) -> int:
    try:
        return header.index(name)
    except ValueError:
        if default is None:
            raise
        return default


def _csv_bool_is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _read_existing_csv(path: str) -> Tuple[List[str], List[List[str]], Set[Tuple[str, str]]]:
    header, rows, skip_pairs = [], [], set()
    with open(path, "r", encoding="utf-8") as fr:
        rdr = csv.reader(fr)
        header = next(rdr, None) or []
        i_model = _idx(header, "modelname", 0)
        i_sid = _idx(header, "sample_id", 1)
        i_error = _idx(header, "error", default=-1)
        i_valid = _idx(header, "is_valid_motion", default=-1)
        for row in rdr:
            if not row:
                continue
            rows.append(row)
            if len(row) <= max(i_model, i_sid):
                continue
            error_ok = True
            valid_ok = False
            if i_error >= 0 and i_error < len(row):
                error_ok = (str(row[i_error]).strip() == "")
            if i_valid >= 0 and i_valid < len(row):
                valid_ok = _csv_bool_is_true(row[i_valid])
            if error_ok and valid_ok:
                skip_pairs.add((row[i_model], row[i_sid]))
    return header, rows, skip_pairs


def _remap_row_to_header(src_header: Sequence[str], row: Sequence[str], dst_header: Sequence[str]) -> List[str]:
    pos = {name: idx for idx, name in enumerate(src_header)}
    out: List[str] = []
    for col in dst_header:
        idx = pos.get(col, None)
        if idx is None or idx >= len(row):
            out.append("")
        else:
            out.append(row[idx])
    return out


def main() -> None:
    ap = argparse.ArgumentParser("Batch motion scorer for the latest Ref4D layout")
    ap.add_argument("--cfg", required=True, type=str)
    ap.add_argument("--base", required=True, type=str, help="repo base")
    ap.add_argument("--out", required=True, type=str, help="final CSV path")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--models", default="", help="Optional comma-separated model filter")
    ap.add_argument("--gen-video-root", default="data/genvideo", help="Generated-video root; expected layout is <root>/<model>/<sample_id>.mp4")
    ap.add_argument("--limit", type=int, default=None, help="Optional maximum number of discovered samples")
    ap.add_argument("--force", action="store_true", help="overwrite existing results")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base)
    cfg_path = args.cfg if os.path.isabs(args.cfg) else os.path.join(base_dir, args.cfg)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_cfg_path"] = os.path.abspath(cfg_path)
    cfg["_base_dir"] = base_dir

    dataset_cfg = cfg.get("dataset", {}) or {}
    meta_path = dataset_cfg.get("meta_path", None)
    if not meta_path:
        raise ValueError("missing required config: dataset.meta_path")
    models_filter = {m.strip() for m in str(args.models).split(",") if m.strip()}
    sample_map, model_map = _scan_dataset_ref4d(
        base_dir,
        str(meta_path),
        str(args.gen_video_root),
        models_filter or None,
        args.limit,
    )

    all_sids = sorted(sample_map.keys())
    if not all_sids:
        print(f"No samples found. Check dataset metadata and generated videos under {args.gen_video_root}.", file=sys.stderr)
        sys.exit(1)

    skip_pairs: Set[Tuple[str, str]] = set()
    existing_rows: List[List[str]] = []
    existing_header: List[str] = []
    if (not args.force) and os.path.isfile(args.out):
        try:
            existing_header, existing_rows, skip_pairs = _read_existing_csv(args.out)
            print(f"[info] cache found: {len(skip_pairs)} pairs in {args.out}")
        except Exception as e:
            print(f"[warn] failed to read existing out: {e}", file=sys.stderr)

    sid_pending_counts: Dict[str, int] = {}
    pending_sids: List[str] = []
    total_pending_pairs = 0
    for sid in all_sids:
        cnt = 0
        for modelname, _ in model_map.get(sid, []):
            if (modelname, sid) not in skip_pairs:
                cnt += 1
        if cnt > 0:
            sid_pending_counts[sid] = cnt
            pending_sids.append(sid)
            total_pending_pairs += cnt

    if not pending_sids:
        with open(args.out, "w", newline="", encoding="utf-8") as fw:
            wr = csv.writer(fw)
            wr.writerow(STD_HEADER)
            for row in existing_rows:
                wr.writerow(_remap_row_to_header(existing_header, row, STD_HEADER))
        print("[done] nothing to do. reused existing CSV.")
        return

    W = max(1, int(args.workers))
    buckets: List[List[str]] = [[] for _ in range(W)]
    loads: List[int] = [0 for _ in range(W)]
    for sid in sorted(pending_sids, key=lambda s: sid_pending_counts[s], reverse=True):
        wid = min(range(W), key=lambda i: loads[i])
        buckets[wid].append(sid)
        loads[wid] += sid_pending_counts[sid]

    print(f"[plan] total pending pairs: {total_pending_pairs} over {len(pending_sids)} samples; workers={W}")
    for i in range(W):
        print(f"[plan] worker-{i}: samples={len(buckets[i])}, est_pairs={loads[i]}")

    ctx = mp.get_context("spawn")
    procs: List[Tuple[int, mp.Process, str]] = []
    chunk_paths: List[str] = []
    for wid in range(W):
        if not buckets[wid]:
            continue
        chunk_csv = f"{os.path.splitext(args.out)[0]}.part{wid}.csv"
        log_txt = f"{os.path.splitext(args.out)[0]}.part{wid}.log"
        chunk_paths.append(chunk_csv)
        p = ctx.Process(
            target=_worker_loop,
            args=(wid, buckets[wid], sample_map, model_map, cfg, chunk_csv, log_txt, skip_pairs),
        )
        p.daemon = False
        p.start()
        procs.append((wid, p, chunk_csv))

    for _, p, _ in procs:
        p.join()

    worker_errors: List[str] = []
    for wid, p, chunk_csv in procs:
        if p.exitcode != 0:
            worker_errors.append(f"worker-{wid} exited with code {p.exitcode}")
        if not os.path.isfile(chunk_csv):
            worker_errors.append(f"worker-{wid} missing output part CSV: {chunk_csv}")
    if worker_errors:
        raise RuntimeError("batch motion workers failed before merge: " + "; ".join(worker_errors))

    written_pairs: Set[Tuple[str, str]] = set()

    with open(args.out, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(STD_HEADER)

        if existing_rows and (not args.force):
            i_model_old = _idx(existing_header, "modelname", 0)
            i_sid_old = _idx(existing_header, "sample_id", 1)
            for row in existing_rows:
                if not row:
                    continue
                if len(row) <= max(i_model_old, i_sid_old):
                    continue
                key = (row[i_model_old], row[i_sid_old])
                if key not in skip_pairs:
                    continue
                if key in written_pairs:
                    continue
                writer.writerow(_remap_row_to_header(existing_header, row, STD_HEADER))
                written_pairs.add(key)

        for part in chunk_paths:
            if not os.path.isfile(part):
                continue
            with open(part, "r", encoding="utf-8") as fr:
                rdr = csv.reader(fr)
                part_header = next(rdr, None) or []
                i_model_part = _idx(part_header, "modelname", 0)
                i_sid_part = _idx(part_header, "sample_id", 1)
                for row in rdr:
                    if not row or len(row) <= max(i_model_part, i_sid_part):
                        continue
                    key = (row[i_model_part], row[i_sid_part])
                    if key in written_pairs:
                        continue
                    writer.writerow(_remap_row_to_header(part_header, row, STD_HEADER))
                    written_pairs.add(key)

    print(f"[done] merged CSV -> {args.out}")


if __name__ == "__main__":
    main()
