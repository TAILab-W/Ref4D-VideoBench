from __future__ import annotations

from typing import List, Optional

import cv2 as cv
import numpy as np

__all__ = [
    "load_video_cv2",
    "save_video_cv2",
    "resize_keep_short_side",
    "resample_video",
]


def _maybe_rotate_by_meta(frame: np.ndarray, orientation: int) -> np.ndarray:
    if not isinstance(orientation, (int, float)):
        return frame
    o = int(orientation)
    if o == 90:
        return cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    if o == 180:
        return cv.rotate(frame, cv.ROTATE_180)
    if o == 270:
        return cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def load_video_cv2(path: str, bgr: bool = True, return_fps: bool = False):
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    try:
        fps_src = float(cap.get(cv.CAP_PROP_FPS))
        if not np.isfinite(fps_src) or fps_src <= 0:
            fps_src = 0.0
    except Exception:
        fps_src = 0.0

    try:
        orientation = cap.get(cv.CAP_PROP_ORIENTATION_META)
    except Exception:
        orientation = 0

    frames: List[np.ndarray] = []
    ok, frame = cap.read()
    while ok:
        if not bgr:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = _maybe_rotate_by_meta(frame, orientation)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Video opened successfully but contains no valid frames: {path}")

    if return_fps:
        return frames, fps_src
    return frames


def save_video_cv2(path: str, frames: List[np.ndarray], fps: int = 25) -> None:
    if len(frames) == 0:
        raise ValueError("No frames to save.")
    h, w = frames[0].shape[:2]
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    vw = cv.VideoWriter(path, fourcc, float(fps), (w, h))
    for img in frames:
        if img.shape[:2] != (h, w):
            img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        vw.write(img)
    vw.release()


def resize_keep_short_side(img: np.ndarray, short_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if short_side <= 0 or h == 0 or w == 0:
        return img
    m = min(h, w)
    if m == short_side:
        return img
    if h < w:
        new_h = short_side
        new_w = int(round(w * (short_side / h)))
    else:
        new_w = short_side
        new_h = int(round(h * (short_side / w)))
    return cv.resize(
        img,
        (new_w, new_h),
        interpolation=cv.INTER_AREA if (new_w < w or new_h < h) else cv.INTER_LINEAR,
    )


def _time_resample_indices(T: int, src_fps: float, tgt_fps: float) -> np.ndarray:
    if not np.isfinite(src_fps) or src_fps <= 0:
        raise ValueError(f"Invalid source FPS for motion resampling: src_fps={src_fps}")
    if not np.isfinite(tgt_fps) or tgt_fps <= 0:
        raise ValueError(f"Invalid target FPS for motion resampling: tgt_fps={tgt_fps}")
    if T <= 1:
        return np.arange(T, dtype=np.int32)
    effective_tgt_fps = min(float(tgt_fps), float(src_fps))
    duration = T / float(src_fps)
    tgt_len = max(1, int(round(duration * effective_tgt_fps)))
    idx = np.linspace(0, T - 1, num=tgt_len).round().astype(np.int32)
    return np.clip(idx, 0, T - 1)


def resample_video(
    frames: List[np.ndarray],
    short_side: int = 448,
    fps: int = 12,
    src_fps: Optional[float] = None,
) -> List[np.ndarray]:
    if len(frames) == 0:
        return []
    scaled = [resize_keep_short_side(f, short_side) for f in frames]
    idx = _time_resample_indices(len(scaled), float(src_fps or 0.0), float(fps))
    return [scaled[i] for i in idx.tolist()]
