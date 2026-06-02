
from __future__ import annotations

"""Low-level single-video subject detection / segmentation helpers for motion evaluation.

This module gathers subject prompts, runs GroundingDINO box detection, and
propagates foreground masks with SAM2 for a single video. It does not build
background masks, does not perform temporal mask repair, and does not compute
motion features or motion metrics. Those responsibilities are handled by higher-
level wrappers such as ``subject_mask_utils.py`` and the motion evaluation entry
points. Prompt lookup follows the latest Ref4D motion contract: semantic
evidence indexed by ``sample_id`` is primary, explicit user/config prompts are
secondary, and empty prompt sets fail loudly. When ``semantic_root`` is not
provided, the default semantic-evidence root is
``<repo_root>/data/metadata/semantic_evidence``. For GroundingDINO and SAM2,
explicit Python package locations are used when provided; otherwise repo-local
``third_party`` fallback paths are resolved and inserted at the front of
``sys.path`` so that those versions are imported with highest priority.
"""

from typing import List, Tuple, Optional
import os
import sys
import json
import shutil
import tempfile

import numpy as np
import cv2 as cv
import torch

__all__ = ["_gather_text_prompts", "_detect_boxes_grounding_dino", "_segment_video_sam2_with_boxes", "_postprocess_mask"]


def _repo_root() -> str:
    repo_root = os.environ.get("REF4D_REPO_ROOT", "").strip()
    if repo_root:
        return os.path.abspath(repo_root)
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _resolve_grounding_dino_pydir() -> str:
    gdino_pydir = os.environ.get("GROUNDING_DINO_PYDIR", "").strip()
    if gdino_pydir:
        return gdino_pydir
    candidate = os.path.join(_repo_root(), "third_party", "GroundingDINO")
    if os.path.isdir(candidate):
        return candidate
    return ""



def _dbg_on() -> bool:
    return os.environ.get("SUBJ_DEBUG", "0") not in ("0", "", "false", "False", "OFF")


def _log(*args):
    if _dbg_on():
        print("[SUBJ]", *args, flush=True)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _postprocess_mask(
    m: np.ndarray,
    erode: int = 1,
    dilate: int = 2,
    fill_hole_ratio: float = 0.005,
) -> np.ndarray:
    if m is None or m.size == 0:
        return m
    m = (m > 0).astype(np.uint8)
    if erode > 0:
        m = cv.erode(
            m,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1)),
        )
    if dilate > 0:
        m = cv.dilate(
            m,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)),
        )
    H, W = m.shape[:2]
    if fill_hole_ratio > 0:
        thr = int(fill_hole_ratio * H * W)
        inv = (1 - m).astype(np.uint8)
        num, lab = cv.connectedComponents(inv, 8)
        for i in range(1, num):
            if int((lab == i).sum()) <= thr:
                inv[lab == i] = 0
        m = (1 - inv).astype(np.uint8)
    return m.astype(bool)


def _default_semantic_root(semantic_root: Optional[str]) -> str:
    if semantic_root is not None and str(semantic_root).strip():
        return os.path.abspath(str(semantic_root))
    return os.path.join(_repo_root(), "data", "metadata", "semantic_evidence")


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as h:
            return json.load(h)
    except Exception:
        try:
            txt = open(path, "r", encoding="utf-8", errors="ignore").read()
            txt = "\n".join([ln for ln in txt.splitlines() if not ln.strip().startswith("//")])
            return json.loads(txt.lstrip("\ufeff"))
        except Exception as e2:
            _log("semantics read error:", e2)
            return None


STOP_TOKENS = {"refvideo", "genvideo", "object", "video", "mp4"}


def _normalize_prompt_phrase(s: object) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    if not s:
        return ""
    for ch in [",", "/", "-", "_", "|", ";", ":"]:
        s = s.replace(ch, " ")
    words = [w for w in s.split() if w.isalpha() and len(w) >= 3 and w not in STOP_TOKENS]
    if not words:
        return ""
    return " ".join(words)


def _read_semantics_for(
    sample_id: str,
    semantic_root: Optional[str],
    ref_path: Optional[str] = None,
) -> List[str]:
    sample_id = str(sample_id).strip()
    if not sample_id:
        raise ValueError("[subject_mask] sample_id must be non-empty for semantic prompt lookup")

    sem_root = _default_semantic_root(semantic_root)
    used_file = os.path.join(sem_root, sample_id + ".json")
    js = _read_json(used_file) if os.path.exists(used_file) else None

    if js is None or not isinstance(js, dict):
        if _dbg_on():
            _log("[SEM] no semantic json for", sample_id, "under", sem_root, "ref_path=", ref_path)
        return []

    if _dbg_on():
        _log("[SEM] use file:", used_file)

    toks: List[str] = []

    try:
        fine = js.get("fine", {}) or {}
        ents = fine.get("entities", []) or []

        def span_len(e: dict) -> float:
            L = 0.0
            for s in e.get("spans", []) or []:
                if isinstance(s, (list, tuple)) and len(s) == 2:
                    try:
                        L += float(s[1]) - float(s[0])
                    except Exception:
                        pass
            return L

        ents_sorted = sorted(ents, key=span_len, reverse=True)

        for e in ents_sorted[:3]:
            name = _normalize_prompt_phrase(e.get("name", ""))
            if name:
                toks.append(name)

            attrs = e.get("attributes", {}) or {}
            for ak in ("species-or-breed", "species", "breed"):
                vs = attrs.get(ak, [])
                if isinstance(vs, str):
                    v = _normalize_prompt_phrase(vs)
                    if v:
                        toks.append(v)
                elif isinstance(vs, (list, tuple)):
                    for x in vs:
                        v = _normalize_prompt_phrase(x)
                        if v:
                            toks.append(v)

        if not toks:
            views = js.get("views", {}) or {}
            oc = views.get("objects_count", {}) or views.get("objects_count_display", {}) or {}
            if isinstance(oc, dict) and len(oc) > 0:
                for k, _v in sorted(oc.items(), key=lambda kv: -int(kv[1]))[:2]:
                    v = _normalize_prompt_phrase(k)
                    if v:
                        toks.append(v)
    except Exception as e:
        if _dbg_on():
            _log("semantics parse warn:", e)

    uniq: List[str] = []
    for t in toks:
        if t and t not in uniq:
            uniq.append(t)

    if _dbg_on():
        _log("[SEM] cleaned:", uniq[:10], " (total", len(uniq), ")")

    return uniq


def _gather_text_prompts(
    sample_id: str,
    ref_path: str,
    cfg_prompts: Optional[List[str]],
    semantic_root: Optional[str] = None,
) -> List[str]:
    sems = _read_semantics_for(sample_id, semantic_root, ref_path=ref_path)

    user: List[str] = []
    if isinstance(cfg_prompts, (list, tuple)):
        user = [_normalize_prompt_phrase(x) for x in cfg_prompts]
        user = [x for x in user if x]

    allp = sems + user
    uniq: List[str] = []
    for t in allp:
        if t and t not in uniq:
            uniq.append(t)

    if _dbg_on():
        _log("[PROM] sample_id:", sample_id)
        _log("[PROM] path:", ref_path)
        _log("[PROM] semantic_root:", _default_semantic_root(semantic_root))
        _log("[PROM]  sems:", sems[:10], " (total", len(sems), ")")
        _log("[PROM]  user:", user)
        _log("[PROM] final:", uniq[:12], " ... total:", len(uniq))

    if not uniq:
        raise RuntimeError(
            f"[subject_mask] empty subject prompts for sample_id={sample_id!r}; "
            "semantic evidence and explicit prompts are both missing or empty"
        )
    return uniq



def _detect_boxes_grounding_dino(
    frames_bgr: List[np.ndarray],
    text_prompts: List[str],
    box_conf_thr: float,
    text_thr: float,
    topk_instances: int,
) -> List[Optional[np.ndarray]]:
    
    
    
    gdino_pydir = _resolve_grounding_dino_pydir()
    if gdino_pydir and gdino_pydir not in sys.path:
        sys.path.insert(0, gdino_pydir)

    try:
        from groundingdino.util.inference import load_model, predict, load_image
        from groundingdino.util import box_ops
    except Exception as e:
        raise RuntimeError(
            f"[subject_mask] GroundingDINO import failed: {e} "
            f"(GROUNDING_DINO_PYDIR={gdino_pydir!r})"
        )

    model_path = os.environ.get("GROUNDING_DINO_WEIGHTS", "").strip()
    model_cfg = os.environ.get("GROUNDING_DINO_CFG", "").strip()
    missing = []
    if not model_cfg:
        missing.append("GROUNDING_DINO_CFG")
    if not model_path:
        missing.append("GROUNDING_DINO_WEIGHTS")
    if missing:
        raise RuntimeError(
            "[subject_mask] Missing required GroundingDINO setting(s): "
            + ", ".join(missing)
        )

    model = load_model(model_cfg, model_path)
    model.eval()

    caption = ", ".join([str(t).strip().lower() for t in text_prompts if str(t).strip()])

    tmp_dir = tempfile.mkdtemp(prefix="gdino_")

    out_boxes: List[Optional[np.ndarray]] = []
    try:
        for idx, bgr in enumerate(frames_bgr):
            try:
                jpg = os.path.join(tmp_dir, f"{idx:05d}.jpg")
                cv.imwrite(jpg, bgr)
                image_source, image = load_image(jpg)
                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption=caption,
                    box_threshold=box_conf_thr,
                    text_threshold=text_thr,
                )
                if boxes is None or (hasattr(boxes, "__len__") and len(boxes) == 0):
                    out_boxes.append(None)
                    continue
                xyxy = (
                    box_ops.box_cxcywh_to_xyxy(boxes)
                    if getattr(boxes, "shape", None) is not None and boxes.max() <= 1.1
                    else boxes
                )
                xyxy = np.asarray(xyxy)
                if logits is not None and len(logits) == len(xyxy):
                    idxs = np.argsort(-np.asarray(logits).reshape(-1))[: max(1, topk_instances)]
                    xyxy = xyxy[idxs]
                else:
                    xyxy = xyxy[: max(1, topk_instances)]
                out_boxes.append(xyxy.astype(np.float32))
            except Exception as e:
                _log(f"GDINO frame{idx} error: {e}")
                out_boxes.append(None)
    finally:
        if os.environ.get("SUBJ_DEBUG", "0") != "2":
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_boxes



def _build_sam2_predictor_from_env():
    sam2_pydir = os.environ.get("SAM2_PYDIR", "").strip()

    if not sam2_pydir:
        repo_root = _repo_root()
        candidate = os.path.join(repo_root, "third_party", "sam2")
        if os.path.isdir(candidate):
            sam2_pydir = candidate

    
    
    
    if sam2_pydir and sam2_pydir not in sys.path:
        sys.path.insert(0, sam2_pydir)

    try:
        from sam2.build_sam import build_sam2_video_predictor
    except Exception as e:
        raise RuntimeError(f"[subject_mask] SAM2 import failed: {e} (SAM2_PYDIR={sam2_pydir!r})")

    
    cfg_env = os.environ.get("SAM2_CFG_NAME", "").strip()
    ckpt = os.environ.get("SAM2_CHECKPOINT", "") or os.environ.get("SAM2_CKPT", "")
    if not ckpt or not os.path.exists(ckpt):
        raise RuntimeError("[subject_mask] SAM2 checkpoint missing (SAM2_CHECKPOINT / SAM2_CKPT)")

    candidates: List[str] = []
    if cfg_env:
        candidates.append(cfg_env)
        if cfg_env.endswith(".yaml"):
            base = cfg_env.replace(".yaml", "")
            if "/" in base and not base.startswith("configs/"):
                candidates.append("configs/" + base)
    candidates.append("configs/sam2.1/sam2.1_hiera_l")

    last_err = None
    for name in candidates:
        try:
            _log("[TRY] build_sam2_video_predictor(", name, ",", ckpt, ")")
            pred = build_sam2_video_predictor(name, ckpt)
            _log("[OK ] predictor built with:", name)
            return pred
        except Exception as e:
            _log("[ERR]", repr(e))
            last_err = e
    raise RuntimeError(f"[subject_mask] build_sam2_video_predictor failed. Last error: {repr(last_err)}")


def _boxes_to_pos_points(boxes_xyxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.int32)
    bb = np.asarray(boxes_xyxy, np.float32).reshape(-1, 4)
    cx = 0.5 * (bb[:, 0] + bb[:, 2])
    cy = 0.5 * (bb[:, 1] + bb[:, 3])
    pts = np.stack([cx, cy], axis=-1).astype(np.float32)
    labels = np.ones((pts.shape[0],), np.int32)
    return pts, labels


def _segment_video_sam2_with_boxes(
    frames_bgr: List[np.ndarray],
    per_frame_boxes: List[Optional[np.ndarray]],
) -> List[Optional[np.ndarray]]:
    predictor = _build_sam2_predictor_from_env()

    tmp_dir = tempfile.mkdtemp(prefix="subj_sam2_")
    try:
        for i, bgr in enumerate(frames_bgr):
            cv.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpeg"), bgr)

        with torch.inference_mode():
            state = predictor.init_state(video_path=tmp_dir)
            predictor.reset_state(state)

            first_idx = None
            for t, b in enumerate(per_frame_boxes):
                if b is not None and len(b) > 0:
                    first_idx = t
                    break
            if first_idx is None:
                _log("SAM2 skip: no boxes at any frame")
                return [None] * len(frames_bgr)

            pts, lbs = _boxes_to_pos_points(per_frame_boxes[first_idx])
            if pts.shape[0] == 0:
                _log("SAM2 skip: boxes->points empty")
                return [None] * len(frames_bgr)

            obj_id = 1
            predictor.add_new_points(
                inference_state=state,
                frame_idx=int(first_idx),
                obj_id=obj_id,
                points=pts,
                labels=lbs,
            )

            masks_bool: List[Optional[np.ndarray]] = [None] * len(frames_bgr)
            for frame_idx, object_ids, masks_out in predictor.propagate_in_video(state):
                ml = masks_out
                if isinstance(ml, dict):
                    ml = ml.get("masks", ml.get("mask_logits", None))
                if ml is None:
                    masks_bool[frame_idx] = None
                    continue
                ml = torch.as_tensor(ml)
                if ml.ndim == 4 and ml.shape[1] == 1:
                    ml = ml[:, 0, ...]
                elif ml.ndim != 3:
                    masks_bool[frame_idx] = None
                    continue
                m_np = (ml > 0).any(dim=0).to(torch.uint8).cpu().numpy()
                masks_bool[frame_idx] = m_np.astype(bool)

            cov = [
                float(np.count_nonzero(m)) / float(m.size)
                for m in masks_bool
                if m is not None and m.size > 0
            ]
            med = float(np.median(cov)) if len(cov) else 0.0
            _log(f"SAM2 coverage: med={med:.6f}, frames={len(cov)}/{len(frames_bgr)}")
            return masks_bool
    finally:
        if os.environ.get("SUBJ_DEBUG", "0") != "2":
            shutil.rmtree(tmp_dir, ignore_errors=True)
