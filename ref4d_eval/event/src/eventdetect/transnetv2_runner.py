
"""
TransNetV2 shot detection runner (strict, transnetv2_pytorch only).

- Input: RGB frames resized to [27,48], kept as uint8, channels-last.
- Model: transnetv2_pytorch.TransNetV2 (expects inputs.dtype=torch.uint8, shape [B,T,27,48,3]).
- Inference: sliding windows with win/stride and overlap averaging.
- Post-processing: Gaussian smoothing + find_peaks (threshold / minimum scene length).
- Output: typically outputs/event/cache/scenes/<video_id>.scenes.json
          with schema {"scenes": [[s, e], ...]} in seconds and strict 0 <= s < e.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import yaml
import torch
from scipy.signal import find_peaks

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_repo_relative_path(path_str: Optional[str]) -> Optional[str]:
    if path_str is None:
        return None
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return str(p)
    return str((_PROJECT_ROOT / p).resolve())




def _resolve_device(device_str: str) -> str:
    d = str(device_str).strip().lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d in {"cpu", "cuda"}:
        if d == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' but CUDA is not available.")
        return d
    raise ValueError(f"Unsupported device: {device_str!r}. Expected one of: auto, cpu, cuda.")

def _validate_scenes(scenes: List[Tuple[float, float]]) -> None:
    prev_s = None
    for idx, item in enumerate(scenes):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid scene at index {idx}: expected [s, e], got {item!r}")
        s, e = item
        try:
            s = float(s)
            e = float(e)
        except Exception as exc:
            raise ValueError(f"Invalid scene at index {idx}: non-numeric endpoints {item!r}") from exc
        if not (np.isfinite(s) and np.isfinite(e)):
            raise ValueError(f"Invalid scene at index {idx}: non-finite endpoints {item!r}")
        if s < 0:
            raise ValueError(f"Invalid scene at index {idx}: start must be >= 0, got {item!r}")
        if not (e > s):
            raise ValueError(f"Invalid scene at index {idx}: must satisfy e > s, got {item!r}")
        if prev_s is not None and s < prev_s:
            raise ValueError(f"Invalid scenes order at index {idx}: start time decreased ({s} < {prev_s})")
        prev_s = s


def _read_video_opencv(video_path: str) -> tuple[list[np.ndarray], float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS for video: {video_path} (fps={fps})")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  
    cap.release()
    return frames, fps, len(frames)


def _scenes_from_transitions(transitions: List[int], n_frames: int, fps: float) -> List[Tuple[float, float]]:
    if n_frames <= 0:
        return []
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and > 0, got {fps}")
    ts = sorted(set([t for t in transitions if 0 <= t < n_frames]))
    
    cuts = [0] + ts + [n_frames]
    scenes: List[Tuple[float, float]] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        s = a / fps
        e = b / fps
        if not (e > s):
            raise ValueError(f"Invalid scene interval derived from transitions: ({s}, {e})")
        scenes.append((float(s), float(e)))
    return scenes


def _save_scenes_json(out_path: str, scenes: List[Tuple[float, float]]) -> None:
    _validate_scenes(scenes)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"scenes": [[float(s), float(e)] for s, e in scenes]}, f, ensure_ascii=False, indent=2)


def _extract_probs_from_output(out) -> np.ndarray:
    if isinstance(out, (list, tuple)):
        pred = out[0]
    else:
        pred = out
    pred = pred.detach().cpu().numpy()
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[:, :, 0]
    if pred.ndim == 2:
        pred = pred[0]
    elif pred.ndim == 1:
        pass
    elif pred.ndim == 0:
        pred = np.asarray([float(pred)], dtype=np.float32)
    return pred.astype(np.float32).reshape(-1)


def _infer_transnetv2_probs(
    frames_rgb: List[np.ndarray],
    device: str = "cuda",
    weights_path: Optional[str] = None,
    resize_hw: tuple[int, int] = (27, 48),
    win: int = 100,
    stride: int = 50,
) -> np.ndarray:
    try:
        from transnetv2_pytorch import TransNetV2  
    except Exception as e:
        raise RuntimeError("transnetv2_pytorch is not available. Please install `transnetv2-pytorch` first.") from e

    H, W = resize_hw
    n = len(frames_rgb)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    arr_uint8 = np.stack([cv2.resize(f, (W, H)) for f in frames_rgb], axis=0).astype(np.uint8)  

    model = TransNetV2().to(device)
    model.eval()
    if weights_path:
        sd = torch.load(weights_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)

    probs = np.zeros((n,), dtype=np.float32)
    counts = np.zeros((n,), dtype=np.float32)

    with torch.inference_mode():
        if n <= win:
            x = torch.from_numpy(arr_uint8).unsqueeze(0).to(device=device, dtype=torch.uint8)  
            out = model(x)
            pred = _extract_probs_from_output(out)  
            t = min(len(pred), n)
            probs[:t] += pred[:t]
            counts[:t] += 1.0
        else:
            starts = list(range(0, n - win + 1, stride))
            tail_start = n - win
            if not starts or starts[-1] != tail_start:
                starts.append(tail_start)
            for start in starts:
                end = min(n, start + win)
                clip = arr_uint8[start:end]  
                x = torch.from_numpy(clip).unsqueeze(0).to(device=device, dtype=torch.uint8)  
                out = model(x)
                pred = _extract_probs_from_output(out)  
                t = min(len(pred), end - start)
                probs[start : start + t] += pred[:t]
                counts[start : start + t] += 1.0

    counts[counts == 0] = 1.0
    probs = probs / counts
    return probs.astype(np.float32)


def _detect_shots_transnet_only(
    video_path: str,
    weights_path: Optional[str],
    device: str = "cuda",
    threshold: float = 0.5,
    min_scene_len_frames: Optional[int] = None,
    min_scene_len_sec: Optional[float] = None,
    win: int = 100,
    stride: int = 50,
) -> List[Tuple[float, float]]:
    frames_rgb, fps, n_frames = _read_video_opencv(video_path)
    if n_frames == 0:
        return []

    probs = _infer_transnetv2_probs(
        frames_rgb, device=device, weights_path=weights_path, resize_hw=(27, 48), win=win, stride=stride
    )

    probs = cv2.GaussianBlur(probs.reshape(-1, 1), (9, 1), 0).reshape(-1)

    if min_scene_len_frames is None:
        if min_scene_len_sec is not None:
            min_scene_len_frames = max(1, int(round(min_scene_len_sec * fps)))
        else:
            min_scene_len_frames = 6

    peaks, _ = find_peaks(probs, height=threshold, distance=int(min_scene_len_frames))
    scenes = _scenes_from_transitions(peaks.tolist(), n_frames, fps)
    return scenes


def main():
    ap = argparse.ArgumentParser(
        description="TransNetV2 shot detection. Input is a video; output is typically outputs/event/cache/scenes/<video_id>.scenes.json."
    )
    ap.add_argument("--video", required=True, help="Input video path.")
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSON path, typically outputs/event/cache/scenes/<video_id>.scenes.json.",
    )
    ap.add_argument("--config", required=True, help="Path to model_shot.yaml.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    device = _resolve_device(cfg.get("device", "cuda"))
    weights = _resolve_repo_relative_path(cfg.get("weights", None))
    threshold = float(cfg.get("threshold", 0.5))
    min_scene_len = cfg.get("min_scene_len", None)
    min_scene_sec = cfg.get("min_scene_sec", None)
    win = int(cfg.get("win", 100))
    stride = int(cfg.get("stride", 50))

    scenes = _detect_shots_transnet_only(
        args.video,
        weights_path=weights,
        device=device,
        threshold=threshold,
        min_scene_len_frames=(int(min_scene_len) if min_scene_len is not None else None),
        min_scene_len_sec=(float(min_scene_sec) if min_scene_sec is not None else None),
        win=win,
        stride=stride,
    )
    _save_scenes_json(args.out, scenes)
    print(f"[TransNetV2] scenes saved -> {args.out} #scenes={len(scenes)}")


if __name__ == "__main__":
    main()
