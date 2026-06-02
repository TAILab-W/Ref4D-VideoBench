
from __future__ import annotations

"""Torch-only TAPIR tracking helpers for motion evidence extraction.
"""

import os
import numpy as np


def _repo_root(base_dir: str | None = None) -> str:
    if base_dir:
        return os.path.abspath(os.path.expanduser(str(base_dir)))
    env_root = str(os.environ.get("REF4D_REPO_ROOT", "") or "").strip()
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _default_ckpt_dir(base_dir: str | None = None) -> str:
    return os.path.join(_repo_root(base_dir=base_dir), "checkpoints", "tapnet_checkpoints")


def _resolve_repo_root_path(path_str: str | None, base_dir: str | None = None) -> str | None:
    if path_str is None:
        return None
    s = str(path_str).strip()
    if not s:
        return None
    if os.path.isabs(s):
        return s
    return os.path.join(_repo_root(base_dir=base_dir), s)





def _load_local_torch_tapir_model(base_dir: str | None = None):
    import sys
    import types
    import importlib.util

    base_dir_repo = os.path.join(_repo_root(base_dir=base_dir), "third_party", "tapir")
    tapnet_dir = os.path.join(base_dir_repo, "tapnet")
    torch_dir = os.path.join(tapnet_dir, "torch")
    torch_file = os.path.join(torch_dir, "tapir_model.py")
    if not os.path.isfile(torch_file):
        raise FileNotFoundError(f"Torch TAPIR file not found: {torch_file}")

    
    if "tapnet" not in sys.modules:
        tapnet_pkg = types.ModuleType("tapnet")
        tapnet_pkg.__path__ = [tapnet_dir]
        sys.modules["tapnet"] = tapnet_pkg
    if "tapnet.torch" not in sys.modules:
        torch_pkg = types.ModuleType("tapnet.torch")
        torch_pkg.__path__ = [torch_dir]
        sys.modules["tapnet.torch"] = torch_pkg

    spec = importlib.util.spec_from_file_location("tapnet.torch.tapir_model", torch_file)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "tapnet.torch"
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_sibling_module(basename: str, base_dir: str | None = None):
    import sys
    import types
    import importlib.util

    base_dir_repo = os.path.join(_repo_root(base_dir=base_dir), "third_party", "tapir")
    tapnet_dir = os.path.join(base_dir_repo, "tapnet")
    torch_dir = os.path.join(tapnet_dir, "torch")

    if "tapnet" not in sys.modules:
        tapnet_pkg = types.ModuleType("tapnet")
        tapnet_pkg.__path__ = [tapnet_dir]
        sys.modules["tapnet"] = tapnet_pkg
    if "tapnet.torch" not in sys.modules:
        torch_pkg = types.ModuleType("tapnet.torch")
        torch_pkg.__path__ = [torch_dir]
        sys.modules["tapnet.torch"] = torch_pkg

    cand = [
        os.path.join(torch_dir, f"{basename}.py"),
        os.path.join(torch_dir, f"{basename}_utils.py"),
    ]
    for p in cand:
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location(f"tapnet.torch.{basename}", p)
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = "tapnet.torch"
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod
    return None


def _torch_available(base_dir: str | None = None) -> bool:
    try:
        import torch  
        _ = _load_local_torch_tapir_model(base_dir=base_dir)
        return True
    except Exception:
        return False





def _threshold_bool(x, vis_thresh: float, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == np.bool_:
        return arr.astype(bool, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        return arr >= float(vis_thresh)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(bool)
    raise TypeError(f"Unsupported {name} dtype: {arr.dtype}")



def _normalize_tracks_visibility(
    raw_tracks,
    *,
    raw_visibility=None,
    raw_occluded=None,
    vis_thresh: float = 0.5,
    source: str = "tapnet",
):
    tracks = np.asarray(raw_tracks)
    if tracks.ndim == 4:
        if tracks.shape[0] != 1:
            raise ValueError(f"{source}: unsupported tracks shape {tracks.shape}")
        tracks = tracks[0]
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"{source}: expected tracks shape [N,T,2] or [T,N,2] with optional batch dim, got {tracks.shape}")

    vis = None
    if raw_visibility is not None:
        vis = np.asarray(raw_visibility)
        if vis.ndim == 3:
            if vis.shape[0] != 1:
                raise ValueError(f"{source}: unsupported visibility shape {vis.shape}")
            vis = vis[0]
        if vis.ndim != 2:
            raise ValueError(f"{source}: expected visibility shape [N,T] or [T,N], got {vis.shape}")
        vis = _threshold_bool(vis, vis_thresh, "visibility")
    elif raw_occluded is not None:
        occ = np.asarray(raw_occluded)
        if occ.ndim == 3:
            if occ.shape[0] != 1:
                raise ValueError(f"{source}: unsupported occluded shape {occ.shape}")
            occ = occ[0]
        if occ.ndim != 2:
            raise ValueError(f"{source}: expected occluded shape [N,T] or [T,N], got {occ.shape}")
        vis = ~_threshold_bool(occ, vis_thresh, "occluded")

    if vis is not None:
        if tracks.shape[:2] == vis.shape:
            pass
        elif (tracks.shape[1], tracks.shape[0]) == vis.shape:
            tracks = np.transpose(tracks, (1, 0, 2))
            vis = vis.T
        else:
            raise ValueError(
                f"{source}: tracks/visibility shape mismatch: tracks={tracks.shape}, vis={vis.shape}"
            )
    else:
        vis = np.ones(tracks.shape[:2], dtype=bool)

    return tracks.astype(np.float32), vis.astype(bool)





class TapirTorchBackend:

    def __init__(self, ckpt_path_pt: str, base_dir: str | None = None):
        import torch

        tapir_model = _load_local_torch_tapir_model(base_dir=base_dir)
        self.torch = torch
        self.model = tapir_model.TAPIR()
        if not os.path.isfile(ckpt_path_pt):
            raise FileNotFoundError(f"TAPIR Torch checkpoint not found: {ckpt_path_pt}")
        state = torch.load(ckpt_path_pt, map_location="cpu")
        
        
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        self._infer_fn = None
        self._infer_kind = None  
        if hasattr(self.model, "infer") and callable(getattr(self.model, "infer")):
            self._infer_kind = "method"
        else:
            infer_mod = _load_sibling_module("inference", base_dir=base_dir)
            if infer_mod is not None and hasattr(infer_mod, "infer") and callable(infer_mod.infer):
                self._infer_fn = infer_mod.infer
                self._infer_kind = "module"
            elif callable(getattr(self.model, "__call__", None)) or callable(getattr(self.model, "forward", None)):
                self._infer_kind = "forward"
            else:
                raise AttributeError("Neither TAPIR.infer nor inference.infer nor forward() is available in Torch TAPIR.")

    def _normalize_output(self, out, vis_thresh: float):
        import torch

        def _to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        raw_tracks = None
        raw_visibility = None
        raw_occluded = None

        if isinstance(out, dict):
            if "tracks" in out:
                raw_tracks = _to_np(out["tracks"])
            raw_visibility = _to_np(out.get("visibility", None)) if "visibility" in out else None
            raw_occluded = _to_np(out.get("occluded", None)) if "occluded" in out else None
        elif isinstance(out, (tuple, list)) and len(out) >= 1:
            raw_tracks = _to_np(out[0])
            if len(out) >= 2:
                raw_visibility = _to_np(out[1])
        else:
            raise ValueError("Torch TAPIR output must be a dict or tuple/list with tracks.")

        if raw_tracks is None:
            raise ValueError("Torch TAPIR output does not contain 'tracks'.")

        return _normalize_tracks_visibility(
            raw_tracks,
            raw_visibility=raw_visibility,
            raw_occluded=raw_occluded,
            vis_thresh=vis_thresh,
            source="tapir-torch",
        )

    def __call__(self, frames_rgb: np.ndarray, seeds_xy: np.ndarray, seeds_t0: int = 0, vis_thresh: float = 0.5):
        import torch
        import cv2 as cv

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        max_hw_env = os.getenv("MOTION_TAPIR_MAX_HW", "").strip()
        try:
            max_hw0 = int(max_hw_env) if len(max_hw_env) else 384
        except Exception:
            max_hw0 = 384
        use_fp16 = os.getenv("MOTION_TAPIR_FP16", "1") not in ("0", "false", "False")

        def _resize_video(frames_f32: np.ndarray, max_hw: int, mult: int = 8):
            t, h, w, c = frames_f32.shape
            if max(h, w) <= max_hw and (h % mult == 0) and (w % mult == 0):
                return frames_f32, 1.0, 1.0
            s = min(1.0, max_hw / float(max(h, w)))
            new_h = max(mult, int(np.floor(h * s / mult) * mult))
            new_w = max(mult, int(np.floor(w * s / mult) * mult))
            new_h = max(new_h, mult)
            new_w = max(new_w, mult)
            out = np.empty((t, new_h, new_w, c), dtype=np.float32)
            for i in range(t):
                out[i] = cv.resize(frames_f32[i], (new_w, new_h), interpolation=cv.INTER_AREA)
            sx = new_w / float(w)
            sy = new_h / float(h)
            return out, sx, sy

        
        seeds_xy = seeds_xy.astype(np.float32)
        q_base = np.concatenate(
            [
                np.full((seeds_xy.shape[0], 1), seeds_t0, np.float32),
                seeds_xy[:, 1:2],
                seeds_xy[:, 0:1],
            ],
            axis=1,
        )

        max_hw = max_hw0
        last_err = None
        for _attempt in range(4):
            try:
                vid_f32 = frames_rgb.astype(np.float32)
                vid_scaled, sx, sy = _resize_video(vid_f32, max_hw=max_hw, mult=8)

                q_scaled = q_base.copy()
                q_scaled[:, 1] *= sy
                q_scaled[:, 2] *= sx

                v = torch.from_numpy(vid_scaled[None]).to(device)
                q_t = torch.from_numpy(q_scaled[None]).to(device)

                with torch.no_grad():
                    def _do_infer():
                        if self._infer_kind == "method":
                            return self.model.infer(v, q_t)
                        if self._infer_kind == "module" and self._infer_fn is not None:
                            try:
                                return self._infer_fn(self.model, v, q_t)
                            except TypeError:
                                return self._infer_fn(self.model, video=v, queries=q_t)
                        if self._infer_kind == "forward":
                            try:
                                return self.model(v, q_t)
                            except Exception:
                                try:
                                    return self.model(v, query_points=q_t)
                                except Exception:
                                    return self.model({"video": v, "query_points": q_t})
                        raise RuntimeError("Unknown TAPIR torch infer kind")

                    if use_fp16 and device.type == "cuda":
                        
                        
                        
                        
                        with torch.autocast("cuda", dtype=torch.float16):
                            out = _do_infer()
                    else:
                        out = _do_infer()

                tracks_s, vis = self._normalize_output(out, vis_thresh=vis_thresh)
                tracks = tracks_s.copy()
                tracks[..., 0] /= (sx if sx != 0 else 1.0)
                tracks[..., 1] /= (sy if sy != 0 else 1.0)
                return tracks.astype(np.float32), vis.astype(bool)

            except RuntimeError as e:
                msg = str(e)
                last_err = e
                is_oom = ("out of memory" in msg.lower()) or ("cuda error: out of memory" in msg.lower())
                if is_oom and device.type == "cuda" and max_hw > 192:
                    torch.cuda.empty_cache()
                    max_hw = max(192, int(max_hw // 1.5))
                    continue
                raise

        raise RuntimeError(f"Torch TAPIR inference failed after retries (last={repr(last_err)})")





_BACKEND_CACHE = {}


def _get_or_create_backend(
    backend: str,
    ckpt_dir: str,
    torch_ckpt_name: str,
    base_dir: str | None = None,
):
    if backend == "torch":
        key = ("torch", os.path.join(ckpt_dir, torch_ckpt_name))
        if key not in _BACKEND_CACHE:
            _BACKEND_CACHE[key] = TapirTorchBackend(key[1], base_dir=base_dir)
        return _BACKEND_CACHE[key]

    raise ValueError(f"unsupported TAPIR backend: {backend}; only 'torch' and 'auto' are supported")





def run_tapir_tracks(
    frames_bgr: np.ndarray,
    seeds_xy: np.ndarray,
    seeds_t0: int = 0,
    backend: str = "auto",
    ckpt_dir: str | None = None,
    torch_ckpt_name: str = "bootstapir_checkpoint_v2.pt",
    vis_thresh: float = 0.5,
    base_dir: str | None = None,
):
    assert frames_bgr.ndim == 4 and frames_bgr.shape[-1] == 3
    frames_rgb = frames_bgr[..., ::-1].astype(np.float32) / 255.0
    ckpt_dir = _resolve_repo_root_path(ckpt_dir, base_dir=base_dir) or _default_ckpt_dir(base_dir=base_dir)

    if backend not in ("torch", "auto"):
        raise ValueError(f"unsupported TAPIR backend: {backend}; only 'torch' and 'auto' are supported")

    if not _torch_available(base_dir=base_dir):
        raise RuntimeError(
            "Torch TAPIR backend is unavailable. Install/configure Torch TAPIR explicitly; "
            "non-TAPIR fallback tracking is disabled by default."
        )
    be = _get_or_create_backend("torch", ckpt_dir, torch_ckpt_name, base_dir=base_dir)
    return be(frames_rgb, seeds_xy, seeds_t0, vis_thresh=vis_thresh)




def track_points_tapir(frames, seeds_xy, cfg_tapir=None, base_dir: str | None = None):
    cfg = cfg_tapir or {}
    backend = cfg.get("backend", "auto")

    weights = _resolve_repo_root_path(cfg.get("weights", None), base_dir=base_dir)
    if isinstance(weights, str) and len(weights) > 0:
        ckpt_dir = os.path.dirname(weights)
        torch_ckpt = os.path.basename(weights)
    else:
        ckpt_dir = _resolve_repo_root_path(
            cfg.get("ckpt_dir", _default_ckpt_dir(base_dir=base_dir)),
            base_dir=base_dir,
        )
        torch_ckpt = cfg.get("torch_ckpt") or "bootstapir_checkpoint_v2.pt"

    vis_thresh = float(cfg.get("vis_thresh", cfg.get("vis_thr", 0.5)))

    return run_tapir_tracks(
        frames_bgr=np.stack(frames, 0),
        seeds_xy=seeds_xy,
        seeds_t0=0,
        backend=backend,
        ckpt_dir=ckpt_dir,
        torch_ckpt_name=torch_ckpt,
        vis_thresh=vis_thresh,
        base_dir=base_dir,
    )
