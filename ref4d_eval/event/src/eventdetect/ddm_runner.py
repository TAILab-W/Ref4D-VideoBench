
"""
DDM-Net GEBD runner for event interval extraction.

Inputs:
- Video: typically provided by the pipeline from generated/reference videos
- Optional scenes json: typically outputs/event/cache/scenes/<video_id>.scenes.json
- Config: model_gebd.yaml

Outputs:
- Events json: typically outputs/event/cache/events/{ref|gen}/<video_id>.events.json
- Raw event interval contract: [{id, s_abs, e_abs, s, e}, ...]

Notes:
- Uses official DDM-Net getModel(model_name, args) for direct inference.
- Keeps strict 5D input [B, T, 3, H, W].
- Automatically retries with the temporal window length expected by positional encodings.
- Output intervals are validated strictly before saving.
"""

import argparse
import json
import os
import re
import sys
import inspect
import random
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[4]

def _resolve_repo_relative_path(path_str: Optional[str]) -> Optional[str]:
    if path_str is None:
        return None
    p = Path(os.path.expandvars(str(path_str))).expanduser()
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return str(p.resolve())



def _ensure_dir(p: Union[str, Path]):
    Path(p).mkdir(parents=True, exist_ok=True)

def _read_video_meta(video_path: str) -> Tuple[float, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n / fps if fps > 0 else 0.0
    cap.release()
    return float(fps), n, float(dur)

def _load_scenes_json(scenes_json: Optional[str]) -> Optional[List[Tuple[float, float]]]:
    if not scenes_json:
        return None
    if not os.path.isfile(scenes_json):
        raise FileNotFoundError(f"scenes json not found: {scenes_json}")
    with open(scenes_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"scenes json must be an object with key 'scenes': {scenes_json}")
    scenes = data.get("scenes", None)
    if not isinstance(scenes, list):
        raise ValueError(f"scenes json field 'scenes' must be a list: {scenes_json}")

    out: List[Tuple[float, float]] = []
    for idx, item in enumerate(scenes, start=1):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid scene at index {idx}: expected [start, end], got {item!r}")
        s, e = item
        try:
            s = float(s)
            e = float(e)
        except Exception as exc:
            raise ValueError(f"Invalid scene at index {idx}: start/end must be numeric, got {item!r}") from exc
        if not (math.isfinite(s) and math.isfinite(e)):
            raise ValueError(f"Invalid scene at index {idx}: non-finite boundary, got {item!r}")
        if e < s:
            raise ValueError(f"Invalid scene at index {idx}: end < start, got {item!r}")
        out.append((s, e))
    return out

def _normalize_events(events_se: List[Tuple[float, float]], total_dur: float) -> List[Dict]:
    if not math.isfinite(total_dur) or total_dur <= 0:
        raise ValueError(f"Invalid total duration for normalization: {total_dur!r}")
    items = []
    for k, (s_abs, e_abs) in enumerate(events_se, start=1):
        try:
            s_abs = float(s_abs)
            e_abs = float(e_abs)
        except Exception as exc:
            raise ValueError(f"Invalid event interval at index {k}: boundaries must be numeric.") from exc
        if not (math.isfinite(s_abs) and math.isfinite(e_abs)):
            raise ValueError(f"Invalid event interval at index {k}: non-finite s_abs/e_abs ({s_abs}, {e_abs}).")
        if e_abs < s_abs:
            raise ValueError(f"Invalid event interval at index {k}: e_abs < s_abs ({s_abs}, {e_abs}).")

        s = max(0.0, min(1.0, s_abs / total_dur))
        e = max(0.0, min(1.0, e_abs / total_dur))
        if e < s:
            raise ValueError(f"Invalid normalized event interval at index {k}: e < s ({s}, {e}).")
        if not (math.isfinite(s) and math.isfinite(e)):
            raise ValueError(f"Invalid normalized event interval at index {k}: non-finite s/e ({s}, {e}).")
        if s < -1e-6 or e > 1.0 + 1e-6:
            raise ValueError(f"Invalid normalized event interval at index {k}: s/e out of range ({s}, {e}).")

        items.append(
            {
                "id": f"e{k:04d}",
                "s_abs": float(s_abs),
                "e_abs": float(e_abs),
                "s": float(s),
                "e": float(e),
            }
        )
    return items

def _validate_event_items(items: List[Dict]) -> None:
    last_s_abs = None
    for idx, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            raise ValueError(f"Invalid event item at index {idx}: expected dict.")
        eid = it.get("id")
        if not isinstance(eid, str) or not eid:
            raise ValueError(f"Invalid event item at index {idx}: missing non-empty id.")
        try:
            s_abs = float(it["s_abs"])
            e_abs = float(it["e_abs"])
            s = float(it["s"])
            e = float(it["e"])
        except Exception as exc:
            raise ValueError(f"Invalid event item at index {idx}: missing numeric s_abs/e_abs/s/e.") from exc
        if not all(math.isfinite(x) for x in (s_abs, e_abs, s, e)):
            raise ValueError(f"Invalid event item at index {idx}: non-finite interval values.")
        if e_abs < s_abs:
            raise ValueError(f"Invalid event item at index {idx}: e_abs < s_abs ({s_abs}, {e_abs}).")
        if s < -1e-6 or e > 1.0 + 1e-6 or e < s:
            raise ValueError(f"Invalid event item at index {idx}: invalid normalized interval ({s}, {e}).")
        if last_s_abs is not None and s_abs < last_s_abs:
            raise ValueError(f"Invalid event ordering at index {idx}: s_abs not non-decreasing.")
        last_s_abs = s_abs

def _save_events_json(out_path: str, items: List[Dict]):
    _ensure_dir(Path(out_path).parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)



def _get_stats(cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    ddm = cfg.get("ddm", {})
    mean = ddm.get("mean", [0.485, 0.456, 0.406])
    std  = ddm.get("std",  [0.229, 0.224, 0.225])
    return np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)

def _read_frames_rgb(video_path: str, t0: float, t1: float, fps: float, n_frames: int,
                     size: Tuple[int, int]) -> List[np.ndarray]:
    W, H = size
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    start_f = max(0, int(round(t0 * fps)))
    end_f   = min(n_frames - 1, int(round(t1 * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames = []
    for _ in range(start_f, end_f + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        frames.append(frame.astype(np.uint8))
    cap.release()
    return frames

def _to_tensor(frames: List[np.ndarray], mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    if len(frames) == 0:
        return torch.empty(0, 3, 224, 224)
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  
    arr = (arr - mean) / std
    arr = np.transpose(arr, (0, 3, 1, 2))  
    return torch.from_numpy(arr)  



def _build_windows_bct(x: torch.Tensor, win: int, stride: int) -> Tuple[torch.Tensor, List[int]]:
    if win % 2 == 0 and win > 1:
        win -= 1
    T = x.shape[0]
    if T < win:
        return torch.empty(0), []
    half = win // 2
    centers, chunks = [], []
    for c in range(half, T - half, stride):
        s = c - half
        e = c + half + 1
        clip = x[s:e].unsqueeze(0)  
        chunks.append(clip)
        centers.append(c)
    if not chunks:
        return torch.empty(0), []
    windows = torch.cat(chunks, dim=0)  
    return windows, centers



def _import_ddm_repo(repo_dir: str):
    repo_root = Path(repo_dir).resolve()
    if (repo_root / "utils" / "getter.py").is_file():
        ddm_repo = repo_root
    else:
        ddm_repo = repo_root / "DDM-Net"
        if not (ddm_repo / "utils" / "getter.py").is_file():
            raise RuntimeError(f"Invalid repo_dir (DDM-Net repo not found): {repo_dir}")
    repo_root_str = str(repo_root)
    ddm_repo_str = str(ddm_repo)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    if ddm_repo_str not in sys.path:
        sys.path.insert(0, ddm_repo_str)

class _DummyArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __getattr__(self, name):
        return None

def _build_model(cfg: Dict) -> torch.nn.Module:
    ddm = cfg.get("ddm", {})
    model_name = ddm.get("model", None)
    if not model_name or not isinstance(model_name, str):
        raise RuntimeError("Config 'ddm.model' is missing or not a string; please set a valid DDM getModel name.")
    try:
        from utils.getter import getModel  
    except Exception as e:
        raise RuntimeError(f"Failed to import utils.getter.getModel from the DDM repo: {e}")

    num_classes = int(ddm.get("num_classes", 2))
    args = _DummyArgs(num_classes=num_classes)
    for k in ["backbone", "resnet_type", "drop_path_rate", "pretrained", "arch", "embed_dim",
              "depths", "num_heads", "mlp_ratio", "qkv_bias", "attn_drop_rate", "drop_rate"]:
        if not hasattr(args, k):
            setattr(args, k, None)

    sig = inspect.signature(getModel)
    if len(sig.parameters) >= 1:
        model = getModel(model_name=model_name, args=args)
    else:
        model = getModel(model_name, args)
    return model

def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)

    if not isinstance(state, dict) or len(state) == 0:
        raise RuntimeError(f"Invalid checkpoint state_dict: {ckpt_path}")

    new_state = {}
    for k, v in state.items():
        if not isinstance(k, str):
            raise RuntimeError(f"Invalid checkpoint key type: {type(k)!r}")
        new_key = k[7:] if k.startswith("module.") else k
        if not new_key:
            raise RuntimeError(f"Invalid checkpoint key after normalization: original key={k!r}")
        new_state[new_key] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)

    if missing or unexpected:
        preview_n = 20
        msg = [
            f"DDM checkpoint load mismatch: {ckpt_path}",
            f"missing={len(missing)}, unexpected={len(unexpected)}",
        ]
        if missing:
            msg.append(f"missing preview: {missing[:preview_n]}")
        if unexpected:
            msg.append(f"unexpected preview: {unexpected[:preview_n]}")
        raise RuntimeError("\n".join(msg))



def _flatten_tensors(obj) -> List[torch.Tensor]:
    out = []
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            out.extend(_flatten_tensors(x))
    elif isinstance(obj, dict):
        for k in obj:
            out.extend(_flatten_tensors(obj[k]))
    return out

def _pick_logits_from_any(out_any, batch: int) -> torch.Tensor:
    tens = _flatten_tensors(out_any)
    if not tens:
        raise RuntimeError("Model forward returned no tensor outputs.")
    c2 = [t for t in tens if t.ndim == 2 and t.shape[0] == batch]
    if c2:
        return c2[-1]
    c3 = [t for t in tens if t.ndim == 3 and t.shape[0] == batch]
    if c3:
        return c3[-1].mean(dim=1)
    c1 = [t for t in tens if t.ndim == 1 and t.shape[0] == batch]
    if c1:
        return c1[-1].unsqueeze(-1)
    shapes = [tuple(t.shape) for t in tens]
    raise RuntimeError(f"Cannot pick logits from outputs. Candidate tensor shapes: {shapes} (expected batch={batch})")

def _parse_expected_T_from_error(err: Exception) -> Optional[int]:
    s = str(err)
    m = re.search(r"size of tensor a \((\d+)\) .* size of tensor b \((\d+)\).*dimension 0", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return max(a, b)
    nums = [int(x) for x in re.findall(r"\((\d+)\)", s) if int(x) > 1]
    return max(nums) if nums else None

@torch.no_grad()
def _forward_prob(model: torch.nn.Module, inp5d: torch.Tensor) -> torch.Tensor:
    b = inp5d.shape[0]
    if b == 1:
        inp5d_run = torch.cat([inp5d, inp5d], dim=0)
        out_any = model(inp5d_run)
        logits = _pick_logits_from_any(out_any, batch=inp5d_run.shape[0])  
        if logits.ndim == 2 and logits.shape[-1] >= 2:
            prob = F.softmax(logits, dim=-1)[:, 1]
        else:
            prob = torch.sigmoid(logits.squeeze(-1))
        return prob[:1].detach().float().cpu()
    else:
        out_any = model(inp5d)
        logits = _pick_logits_from_any(out_any, batch=b)
        if logits.ndim == 2 and logits.shape[-1] >= 2:
            prob = F.softmax(logits, dim=-1)[:, 1]
        else:
            prob = torch.sigmoid(logits.squeeze(-1))
        return prob.detach().float().cpu()

@torch.no_grad()
def _infer_segment_scores(model: torch.nn.Module,
                          x: torch.Tensor,
                          device: torch.device,
                          win: int,
                          stride: int,
                          batch_size: int) -> Tuple[np.ndarray, List[int]]:
    def run_windows(windows, centers):
        scores = []
        for st in range(0, windows.shape[0], batch_size):
            ed = min(windows.shape[0], st + batch_size)
            inp = windows[st:ed].to(device)                    
            prob = _forward_prob(model, inp)                   
            scores.append(prob)
        probs = torch.cat(scores, dim=0).numpy()
        return probs, centers

    windows, centers = _build_windows_bct(x, win=win, stride=stride)  
    if windows.numel() == 0:
        return np.zeros((0,), dtype=np.float32), []
    try:
        return run_windows(windows, centers)
    except RuntimeError as e:
        T_expected = _parse_expected_T_from_error(e)
        if T_expected is None or T_expected == win:
            raise
        windows2, centers2 = _build_windows_bct(x, win=T_expected, stride=stride)
        if windows2.numel() == 0:
            raise
        return run_windows(windows2, centers2)



def _scores_to_boundaries(scores: np.ndarray,
                          centers: List[int],
                          threshold: float,
                          min_distance: int,
                          min_prominence: float = 0.08) -> List[int]:
    import scipy.signal
    if len(scores) == 0:
        return []
    s = scores.astype(np.float32).reshape(-1)
    try:
        s = cv2.GaussianBlur(s.reshape(-1, 1), (9, 1), 0).reshape(-1)
    except Exception:
        pass
    peaks, _ = scipy.signal.find_peaks(
        s,
        height=float(threshold),
        distance=int(min_distance),
        prominence=float(min_prominence) if min_prominence is not None else None
    )
    return [int(centers[p]) for p in peaks.tolist()]

def _rseed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def _merge_short_intervals(
    intervals: List[Tuple[float, float]],
    min_event_dur: float,
    eps: float = 1e-6,
) -> List[Tuple[float, float]]:
    """
    Merge short intervals into neighboring intervals instead of dropping them.

    Contract:
    - Input intervals are expected to be sorted and contiguous within one scene/range.
    - Intervals shorter than min_event_dur are merged into a neighbor.
    - If only one interval exists, keep it even if it is shorter than min_event_dur.
    - This preserves full temporal coverage of [seg_t0, seg_t1].
    """
    clean: List[Tuple[float, float]] = []
    for s, e in intervals:
        s = float(s)
        e = float(e)
        if e - s > eps:
            clean.append((s, e))

    if not clean:
        return []

    min_event_dur = max(0.0, float(min_event_dur))
    if min_event_dur <= eps:
        return clean

    intervals = clean

    while len(intervals) > 1:
        durations = [max(0.0, e - s) for s, e in intervals]
        short_idxs = [i for i, d in enumerate(durations) if d < min_event_dur - eps]
        if not short_idxs:
            break

        # Merge the shortest remaining interval first.
        i = min(short_idxs, key=lambda k: durations[k])

        if i == 0:
            # Head short segment: merge into the right neighbor.
            merged = (intervals[0][0], intervals[1][1])
            intervals = [merged] + intervals[2:]
        elif i == len(intervals) - 1:
            # Tail short segment: merge into the left neighbor.
            merged = (intervals[i - 1][0], intervals[i][1])
            intervals = intervals[: i - 1] + [merged]
        else:
            # Middle short segment: merge into the shorter neighbor to avoid
            # creating unnecessarily long intervals.
            left_d = durations[i - 1]
            right_d = durations[i + 1]
            if left_d <= right_d:
                merged = (intervals[i - 1][0], intervals[i][1])
                intervals = intervals[: i - 1] + [merged] + intervals[i + 1 :]
            else:
                merged = (intervals[i][0], intervals[i + 1][1])
                intervals = intervals[:i] + [merged] + intervals[i + 2 :]

    return intervals


def _boundaries_to_events(boundary_frames: List[int],
                          fps: float,
                          seg_t0: float,
                          seg_t1: float,
                          min_event_dur: float) -> List[Tuple[float, float]]:
    """
    Convert GEBD boundary frames to event intervals.

    Unlike the old filtering behavior, this function does not drop short
    intervals. It first builds a complete contiguous timeline over
    [seg_t0, seg_t1], then merges intervals shorter than min_event_dur into
    neighboring intervals. This preserves temporal coverage while avoiding
    overly fragmented event segments.
    """
    seg_t0 = float(seg_t0)
    seg_t1 = float(seg_t1)
    eps = 1e-6

    if seg_t1 - seg_t0 <= eps:
        return []

    fps_safe = max(float(fps), eps)

    # Keep only valid internal boundaries. Boundary points at scene edges would
    # create zero-length intervals and should be ignored.
    bsec = sorted({
        max(seg_t0, min(seg_t1, float(bf) / fps_safe))
        for bf in boundary_frames
    })
    bsec = [b for b in bsec if (seg_t0 + eps) < b < (seg_t1 - eps)]

    cuts = [seg_t0] + bsec + [seg_t1]
    intervals: List[Tuple[float, float]] = []
    for s, e in zip(cuts[:-1], cuts[1:]):
        s = float(s)
        e = float(e)
        if e - s > eps:
            intervals.append((s, e))

    return _merge_short_intervals(intervals, min_event_dur=float(min_event_dur), eps=eps)



def _run_ddm_direct(video_path: str,
                    cfg: Dict,
                    scenes: Optional[List[Tuple[float, float]]] = None) -> List[Tuple[float, float]]:
    ddm = cfg.get("ddm", {})
    repo_dir   = _resolve_repo_relative_path(ddm.get("repo_dir"))
    ckpt_path  = _resolve_repo_relative_path(ddm.get("ckpt"))
    device_str = ddm.get("device", "auto")
    win        = int(ddm.get("window", 5))
    stride     = int(ddm.get("stride", 1))
    input_size = int(ddm.get("input_size", 224))

    
    seed             = int(cfg.get("seed", 1337))
    batch_size       = int(cfg.get("batch_size", 128))
    threshold        = float(cfg.get("threshold", 0.5))
    min_peak_dist    = int(cfg.get("min_peak_distance", 3))
    min_peak_prom    = float(cfg.get("min_peak_prominence", 0.08))
    min_event_dur    = float(cfg.get("min_event_dur", 0.10))

    if not repo_dir or not os.path.isdir(repo_dir):
        raise RuntimeError(f"Invalid repo_dir: {repo_dir}")
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    _rseed_all(seed)
    _import_ddm_repo(repo_dir)

    if device_str == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_str in {"cpu", "cuda"}:
        resolved_device = device_str
    else:
        raise RuntimeError(f"Unsupported ddm.device: {device_str!r}. Expected one of: auto, cpu, cuda.")
    device = torch.device(resolved_device)
    model = _build_model(cfg).to(device)
    _load_checkpoint(model, ckpt_path, device)
    model.eval()

    mean, std = _get_stats(cfg)
    fps, n_frames, dur = _read_video_meta(video_path)

    def run_one_range(t0: float, t1: float) -> List[Tuple[float, float]]:
        frames = _read_frames_rgb(video_path, t0, t1, fps, n_frames, size=(input_size, input_size))
        if len(frames) < win:
            x = _to_tensor(frames, mean, std)
            scores, centers_local = np.zeros((0,), dtype=np.float32), []
        else:
            x = _to_tensor(frames, mean, std)  
            scores, centers_local = _infer_segment_scores(
                model, x, device, win=win, stride=stride, batch_size=batch_size
            )
        seg_start_f = max(0, int(round(t0 * fps)))
        centers_global = [c + seg_start_f for c in centers_local]
        bframes = _scores_to_boundaries(
            scores, centers_global, threshold=threshold,
            min_distance=min_peak_dist, min_prominence=min_peak_prom
        )
        return _boundaries_to_events(bframes, fps=fps, seg_t0=t0, seg_t1=t1, min_event_dur=min_event_dur)

    events_all: List[Tuple[float, float]] = []
    if scenes:
        for (s, e) in scenes:
            s = float(s); e = float(e)
            if e - s <= 1e-6:
                continue
            events_all.extend(run_one_range(s, e))
    else:
        events_all.extend(run_one_range(0.0, dur))

    
    events_all.sort(key=lambda x: x[0])
    return events_all



def main():
    ap = argparse.ArgumentParser(description="DDM-Net GEBD runner for event interval extraction")
    ap.add_argument("--video", required=True, help="Input video path.")
    ap.add_argument("--out", required=True, help="Output events json, e.g. outputs/event/cache/events/{ref|gen}/<video_id>.events.json")
    ap.add_argument("--config", required=True, help="Path to model_gebd.yaml")
    ap.add_argument("--scenes", default=None, help="Optional scenes json, e.g. outputs/event/cache/scenes/<video_id>.scenes.json")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    scenes = _load_scenes_json(args.scenes) if args.scenes else None
    fps, n_frames, dur = _read_video_meta(args.video)
    events = _run_ddm_direct(args.video, cfg, scenes=scenes)

    items = _normalize_events(events, total_dur=dur)
    _validate_event_items(items)
    _save_events_json(args.out, items)
    print(f"[DDM-Net] events saved -> {args.out}  #events={len(items)}")

if __name__ == "__main__":
    main()
