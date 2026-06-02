"""
Segment-wise event description with VideoLLaMA3.

Path contract:
- Input video : user-provided video path
- Input events: usually outputs/event/cache/events/{ref|gen}/{sample_id}.events.json
- Output VLM  : usually outputs/event/cache/vlm/{ref|gen}/{sample_id}.vlm.json

Each output item preserves the original event fields and appends:
  - text: concise event description for that interval
"""
from __future__ import annotations
import os, math, json, time, argparse, logging, traceback, re
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import decord
    from decord import VideoReader, cpu as decord_cpu
    _HAVE_DECORD = True
except Exception:
    _HAVE_DECORD = False

try:
    import cv2
    _HAVE_OPENCV = True
except Exception:
    _HAVE_OPENCV = False

from ..common.io import read_json, write_json, read_yaml, ensure_dir, set_random_seed


LOGGER = logging.getLogger("event.vlm.vllama3")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


_PROJECT_ROOT = Path(__file__).resolve().parents[4]

def _resolve_repo_relative_path(path_str: str) -> str:
    p = Path(str(path_str)).expanduser()
    if p.is_absolute():
        return str(p)
    return str((_PROJECT_ROOT / p).resolve())

def _validate_event_record(seg: Dict[str, Any], idx: int) -> Dict[str, Any]:
    if not isinstance(seg, dict):
        raise ValueError(f"events[{idx}] must be a dict")
    seg_id = seg.get("id") or seg.get("eid") or seg.get("event_id")
    if not isinstance(seg_id, str) or not seg_id.strip():
        raise ValueError(f"events[{idx}] missing non-empty id")
    required = ("s_abs", "e_abs", "s", "e")
    for key in required:
        if key not in seg:
            raise ValueError(f"events[{idx}] ({seg_id}) missing required field: {key}")
    try:
        s_abs = float(seg["s_abs"]); e_abs = float(seg["e_abs"])
        s = float(seg["s"]); e = float(seg["e"])
    except Exception as exc:
        raise ValueError(f"events[{idx}] ({seg_id}) has non-numeric time fields") from exc
    vals = {"s_abs": s_abs, "e_abs": e_abs, "s": s, "e": e}
    for k, v in vals.items():
        if not math.isfinite(v):
            raise ValueError(f"events[{idx}] ({seg_id}) field {k} must be finite")
    if e_abs < s_abs:
        raise ValueError(f"events[{idx}] ({seg_id}) requires e_abs >= s_abs")
    eps = 1e-6
    if s < -eps or e < -eps or s > 1.0 + eps or e > 1.0 + eps or e < s - eps:
        raise ValueError(f"events[{idx}] ({seg_id}) requires normalized 0 <= s <= e <= 1")
    return seg






def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "fp16").lower()
    if s == "bf16":
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    return torch.float16


def _resolve_device_mode(device_str: str) -> tuple[str, str, Optional[str]]:
    req = (device_str or "auto").lower()
    if req not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Unsupported device: {device_str!r}. Expected one of: auto / cpu / cuda")

    has_cuda = torch.cuda.is_available()
    if req == "cuda" and not has_cuda:
        raise RuntimeError("device='cuda' was requested, but CUDA is unavailable on this machine.")

    resolved = "cuda" if ((req == "cuda") or (req == "auto" and has_cuda)) else "cpu"
    device_map = "auto" if (req == "auto" and resolved == "cuda") else None
    return req, resolved, device_map

def _postprocess_caption(text: str) -> str:
    t = (text or "").strip()
    t = t.replace("<think>", "").replace("</think>", "")
    t = t.replace("```", " ").replace("\n", " ").replace("\r", " ")
    t = " ".join(t.split())
    if len(t) > 160:
        for stop in [". ", "! ", "? ", "。", "！", "？"]:
            k = t.find(stop)
            if k != -1 and k < 160:
                t = t[:k+1]
                break
        if len(t) > 160:
            t = t[:160].rstrip(" ,;")
    return t

def _strip_prompt_prefix(text: str, prompt_text: str) -> str:
    t = (text or "").strip()
    if prompt_text and t.lower().startswith(prompt_text.lower()):
        t = t[len(prompt_text):].lstrip(":：- ").strip()
    return t

def _expand_placeholders(s: str, cfg: Dict[str, Any]) -> str:
    data_root = cfg.get("paths", {}).get("data_root", "outputs/event/cache")
    return s.replace("${paths.data_root}", str(data_root))

def _contains_chinese(s: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', s or ""))






def _resize_short_keep_ar(img: Image.Image, short: int) -> Image.Image:
    if not short or short <= 0: return img
    w, h = img.size
    if min(w, h) == short: return img
    if w <= h:
        new_w = short
        new_h = int(round(h * short / w))
    else:
        new_h = short
        new_w = int(round(w * short / h))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)

def _sample_frame_indices_by_fps(start_s: float, end_s: float, fps_req: float,
                                 max_frames: int, fps_video: float, n_total: int) -> List[int]:
    duration = max(0.0, end_s - start_s)
    if duration <= 0:
        center = int(min(max(0, round(start_s * fps_video)), max(0, n_total - 1)))
        return [center]
    want = int(math.ceil(duration * max(0.0, fps_req)))
    want = max(1, min(max_frames, want))
    t = np.linspace(start_s, max(start_s, end_s - 1e-6), num=want, dtype=np.float64)
    idx = np.clip((t * fps_video).round().astype(np.int64), 0, max(0, n_total - 1))
    return idx.tolist()

def _frames_by_decord(video_path: str, s_abs: float, e_abs: float,
                      fps_req: float, max_frames: int, short_edge: int) -> List[Image.Image]:
    vr = VideoReader(video_path, ctx=decord_cpu(0))
    n_total = len(vr)
    try:
        fps_video = float(vr.get_avg_fps())
    except Exception:
        fps_video = 25.0
    fps_video = fps_video if fps_video > 0 else 25.0
    idxs = _sample_frame_indices_by_fps(s_abs, e_abs, fps_req, max_frames, fps_video, n_total)
    batch = vr.get_batch(idxs).asnumpy()  
    images: List[Image.Image] = []
    for arr in batch:
        img = Image.fromarray(arr)  
        img = _resize_short_keep_ar(img, short_edge)
        images.append(img)
    if not images and n_total > 0:
        arr = vr[0].asnumpy()
        images = [_resize_short_keep_ar(Image.fromarray(arr), short_edge)]
    return images

def _frames_by_opencv(video_path: str, s_abs: float, e_abs: float,
                      fps_req: float, max_frames: int, short_edge: int) -> List[Image.Image]:
    if not _HAVE_OPENCV:
        raise RuntimeError("OpenCV not available and decord failed.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fps_video = fps_video if fps_video > 0 else 25.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idxs = _sample_frame_indices_by_fps(s_abs, e_abs, fps_req, max_frames, fps_video, n_total)
        images: List[Image.Image] = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None: continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = _resize_short_keep_ar(Image.fromarray(frame), short_edge)
            images.append(img)
        if not images and n_total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if ok and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images = [_resize_short_keep_ar(Image.fromarray(frame), short_edge)]
        return images
    finally:
        cap.release()

def sample_frames(video_path: str, s_abs: float, e_abs: float,
                  backend: str, fps_req: float, max_frames: int, short_edge: int) -> List[Image.Image]:
    backend = (backend or "decord").lower()
    if backend == "decord" and _HAVE_DECORD:
        try:
            return _frames_by_decord(video_path, s_abs, e_abs, fps_req, max_frames, short_edge)
        except Exception as e:
            LOGGER.warning(f"decord failed, fallback to opencv. err={e}")
    return _frames_by_opencv(video_path, s_abs, e_abs, fps_req, max_frames, short_edge)







def _health_check_local_model_dir(model_dir: Path) -> None:
    must_exist = ["config.json", "processor_config.json"]
    missing = [x for x in must_exist if not (model_dir / x).exists()]
    has_weights = any(model_dir.glob("*.safetensors")) or \
                  (model_dir / "pytorch_model.bin").exists() or \
                  (model_dir / "pytorch_model.bin.index.json").exists()
    if missing or not has_weights:
        raise FileNotFoundError(
            f"[VLM] Local model dir incomplete: {model_dir}\n"
            f"Missing: {missing}; weights_present={has_weights}"
        )

def load_vlmodel(cfg: Dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoProcessor

    vcfg = cfg.get("vllama3", {}) or {}
    model_path = _resolve_repo_relative_path(vcfg.get("model", "checkpoints/videollama3-7b"))
    device_req, target_device, device_map = _resolve_device_mode(vcfg.get("device", "auto"))
    dtype = _dtype_from_str(vcfg.get("dtype", "bf16"))
    attn_impl = vcfg.get("attn_impl", None)
    local_files_only = bool(vcfg.get("local_files_only", False))

    is_local_dir = Path(model_path).exists() and Path(model_path).is_dir()
    if is_local_dir:
        _health_check_local_model_dir(Path(model_path))

    if target_device == "cpu" and dtype != torch.float32:
        LOGGER.info(f"[VLM] Forcing dtype=float32 on CPU (was {dtype}).")
        dtype = torch.float32

    LOGGER.info(
        f"Loading VLM: {model_path} (dtype={dtype}, device={device_req}, "
        f"resolved_device={target_device}, device_map={device_map}, local_files_only={local_files_only})"
    )

    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    if attn_impl and target_device == "cuda":
        load_kwargs["attn_implementation"] = attn_impl
    elif attn_impl:
        LOGGER.info(f"[VLM] Ignoring attn_impl={attn_impl!r} on non-CUDA device.")
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load AutoProcessor from {model_path}: {e}")

    if device_map is None:
        model.to(target_device)
    model.eval()

    seed = vcfg.get("gen", {}).get("seed", None)
    if seed is not None:
        set_random_seed(int(seed), deterministic_torch=True, quiet=True)

    return model, processor







def _resolve_prompt(cfg: Dict[str, Any]) -> tuple[str, str, bool]:
    vcfg = cfg.get("vllama3", {}) or {}
    pcfg = vcfg.get("prompt", {}) or {}

    lang = (pcfg.get("lang", "") or "").lower()
    strict = bool(pcfg.get("strict", False))

    prompt_text = str(pcfg.get("template", "") or pcfg.get("text", ""))
    if not prompt_text:
        
        templates = pcfg.get("templates", {}) or {}
        if lang in templates:
            prompt_text = str(templates[lang])
        elif templates:
            
            prompt_text = str(templates.get("en") or templates.get("zh") or next(iter(templates.values())))

    if not prompt_text:
        prompt_text = "Summarize the main event in one concise sentence (verb + key nouns)."

    
    if not strict:
        if lang in ("en", "english"):
            if _contains_chinese(prompt_text):
                pass
            elif "English" not in prompt_text and "Answer in English" not in prompt_text:
                prompt_text = f"{prompt_text} Answer in English."

    return prompt_text, lang, strict







def _build_messages(frames: List[Image.Image], prompt_text: str) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = [{"type": "image"} for _ in frames]
    contents.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": contents}]

def _move_inputs_to_device(inputs: Dict[str, Any], device: str, dtype: torch.dtype):
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            if k in ("pixel_values", "video_pixel_values"):
                inputs[k] = v.to(device, dtype=dtype, non_blocking=True)
            elif k in ("input_ids", "attention_mask", "position_ids"):
                inputs[k] = v.to(device, non_blocking=True)
            elif ("grid" in k) or ("merge" in k):
                inputs[k] = v.to(device, non_blocking=True)
            else:
                inputs[k] = v.to(device, non_blocking=True)
        elif isinstance(v, (list, tuple)):
            if ("grid" in k) or ("merge" in k):
                try:
                    inputs[k] = torch.tensor(v, device=device)
                except Exception:
                    pass
    return inputs

@torch.inference_mode()
def infer_segment_text(
    model,
    processor,
    frames: List[Image.Image],
    prompt_text: str,
    gen_cfg: Dict[str, Any],
    device_choice: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    force_generate: bool = False,
) -> str:
    do_sample = bool(gen_cfg.get("do_sample", False))
    
    if hasattr(model, "chat") and not force_generate:
        try:
            chat_kwargs = {
                "do_sample": do_sample,
                "max_new_tokens": int(gen_cfg.get("max_new_tokens", 64)),
            }
            if do_sample:
                chat_kwargs["temperature"] = float(gen_cfg.get("temperature", 0.4))
                chat_kwargs["top_p"] = float(gen_cfg.get("top_p", 0.8))
            resp, _ = model.chat(
                processor,
                images=frames,
                question=prompt_text,
                history=None,
                **chat_kwargs,
            )
            return _postprocess_caption(resp)
        except Exception as e:
            LOGGER.warning(f"chat() failed, fallback to template+generate(). err={e}")

    
    tokenizer = None
    _processor = processor
    if isinstance(processor, dict) and "tokenizer" in processor:
        tokenizer = processor["tokenizer"]
        _processor = None

    if _processor is not None and hasattr(_processor, "apply_chat_template"):
        messages = _build_messages(frames, prompt_text)
        prompt_bos = _processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = _processor(text=prompt_bos, images=frames, return_tensors="pt")
        inputs = _move_inputs_to_device(inputs, device_choice, dtype)

        input_ids = inputs.get("input_ids", None)
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2 or input_ids.shape[0] < 1:
            raise RuntimeError("generate() path requires a non-empty batched input_ids tensor.")
        prompt_len = int(input_ids.shape[-1])
        if prompt_len <= 0:
            raise RuntimeError("generate() path requires prompt_len > 0.")

        allowed = (
            "max_new_tokens","do_sample","temperature","top_p","top_k",
            "repetition_penalty","no_repeat_ngram_size","length_penalty",
            "num_beams","early_stopping","min_new_tokens"
        )
        gen_kwargs = {k: gen_cfg[k] for k in allowed if k in gen_cfg}
        
        if "max_new_tokens" in gen_kwargs: gen_kwargs["max_new_tokens"] = int(gen_kwargs["max_new_tokens"])
        if "min_new_tokens" in gen_kwargs: gen_kwargs["min_new_tokens"] = int(gen_kwargs["min_new_tokens"])
        if "no_repeat_ngram_size" in gen_kwargs: gen_kwargs["no_repeat_ngram_size"] = int(gen_kwargs["no_repeat_ngram_size"])
        if "num_beams" in gen_kwargs: gen_kwargs["num_beams"] = int(gen_kwargs["num_beams"])
        if "length_penalty" in gen_kwargs: gen_kwargs["length_penalty"] = float(gen_kwargs["length_penalty"])
        if "repetition_penalty" in gen_kwargs: gen_kwargs["repetition_penalty"] = float(gen_kwargs["repetition_penalty"])
        if "temperature" in gen_kwargs: gen_kwargs["temperature"] = float(gen_kwargs["temperature"])
        if "top_p" in gen_kwargs: gen_kwargs["top_p"] = float(gen_kwargs["top_p"])
        if not bool(gen_kwargs.get("do_sample", False)):
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
            gen_kwargs.pop("top_k", None)

        out = model.generate(**inputs, use_cache=True, **gen_kwargs)
        if not isinstance(out, torch.Tensor) or out.ndim < 2 or out.shape[0] < 1:
            raise RuntimeError("generate() returned an invalid output tensor.")
        tok = getattr(_processor, "tokenizer", None)
        seq = out[0]

        
        if seq.shape[-1] >= prompt_len:
            new_tokens = seq[prompt_len:]
            text = tok.decode(new_tokens, skip_special_tokens=True) if tok is not None else ""
            text = _strip_prompt_prefix(text, prompt_bos)
            return _postprocess_caption(text)

        
        if seq.shape[-1] > 0:
            text = tok.decode(seq, skip_special_tokens=True) if tok is not None else ""
            return _postprocess_caption(text)

        raise RuntimeError("generate() returned an empty sequence.")

    
    raise RuntimeError("Processor/tokenizer path unsupported for generate(); no chat template and no local tokenizer available.")







def _infer_side(events_path: str) -> str:
    parts = [part.lower() for part in Path(events_path).parts]
    if "ref" in parts:
        return "ref"
    if "gen" in parts:
        return "gen"
    return "ref"

def _infer_sample_id(events_path: str) -> str:
    name = Path(events_path).name
    if name.endswith(".events.json"):
        return name[:-len(".events.json")]
    return Path(events_path).stem

def _derive_out_path(cfg: Dict[str, Any], side: str, sample_id: str) -> Path:
    vcfg = cfg.get("vllama3", {}) or {}
    ecfg = vcfg.get("export", {}) or {}
    out_dir = _expand_placeholders(str(ecfg.get("out_dir", "${paths.data_root}/vlm")), cfg)
    pat = ecfg.get("fname_ref" if side == "ref" else "fname_gen", "{sample_id}.vlm.json")
    rel = pat.format(sample_id=sample_id)
    out_path = Path(out_dir) / rel
    ensure_dir(out_path.parent)
    return out_path

def run(video_path: str, events_json_path: str, out_json_path: Optional[str], cfg_path: str) -> Dict[str, Any]:
    cfg = read_yaml(cfg_path)
    vcfg = cfg.get("vllama3", {}) or {}

    model, processor = load_vlmodel(cfg)

    
    dc = vcfg.get("video_decode", {}) or {}
    backend = str(dc.get("backend", "decord"))
    fps_req = float(dc.get("fps", 1.0))
    max_frames = int(dc.get("max_frames", 180))
    short_edge = int(dc.get("short_edge", 384))

    
    prompt_text, lang, strict = _resolve_prompt(cfg)
    gen_cfg = vcfg.get("gen", {}) or {}

    _, device_choice, _ = _resolve_device_mode(vcfg.get("device", "auto"))
    dtype = _dtype_from_str(vcfg.get("dtype", "bf16"))
    if device_choice == "cpu" and dtype != torch.float32:
        dtype = torch.float32
    force_generate = bool(gen_cfg.get("force_generate", False))

    events = read_json(events_json_path)
    if not isinstance(events, list):
        raise ValueError(f"events json must be a list: {events_json_path}")
    events = [_validate_event_record(seg, idx) for idx, seg in enumerate(events)]

    side = _infer_side(events_json_path)
    sample_id = _infer_sample_id(events_json_path)

    if out_json_path:
        out_path = Path(out_json_path)
        ensure_dir(out_path.parent)
    else:
        out_path = _derive_out_path(cfg, side, sample_id)

    
    LOGGER.info(f"[Prompt] strict={strict} lang={lang} -> {prompt_text}")

    t0 = time.time()
    out_events: List[Dict[str, Any]] = []
    for idx, seg in enumerate(events):
        s_abs = float(seg.get("s_abs", 0.0))
        e_abs = float(seg.get("e_abs", 0.0))
        seg_id = str(seg.get("id", f"e{idx:03d}"))

        try:
            frames = sample_frames(
                video_path=video_path,
                s_abs=s_abs,
                e_abs=e_abs,
                backend=backend,
                fps_req=fps_req,
                max_frames=max_frames,
                short_edge=short_edge,
            )

            caption = infer_segment_text(
                model=model,
                processor=processor,
                frames=frames,
                prompt_text=prompt_text,
                gen_cfg=gen_cfg,
                device_choice=device_choice,
                dtype=dtype,
                force_generate=force_generate,
            )
        except Exception as e:
            LOGGER.error(f"[{idx+1}/{len(events)}] seg={seg_id} failed: {e}")
            LOGGER.debug(traceback.format_exc())
            raise RuntimeError(f"Segment inference failed for seg={seg_id}") from e

        seg_out = dict(seg)
        seg_out["text"] = _postprocess_caption(caption)
        if not seg_out["text"]:
            raise RuntimeError(f"Empty caption after postprocess for seg={seg_id}")
        out_events.append(seg_out)
        LOGGER.info(f"[{idx+1}/{len(events)}] seg={seg_id} frames={len(frames)} -> '{seg_out['text']}'")

    write_json(out_events, out_path, indent=2)
    LOGGER.info(f"Done: {out_path} (segments={len(out_events)}) in {time.time()-t0:.2f}s")
    return {"n_segments": len(out_events), "out": str(out_path)}

def parse_args():
    ap = argparse.ArgumentParser(description="VideoLLaMA3 segment captioning for event intervals.")
    ap.add_argument("--video", type=str, required=True, help="Path to source video for the current sample/pair.")
    ap.add_argument("--events", type=str, required=True, help="Path to event intervals JSON, usually outputs/event/cache/events/{ref|gen}/<id>.events.json")
    ap.add_argument("--config", type=str, required=True, help="Path to model_vlm.yaml")
    ap.add_argument("--out", type=str, default=None, help="Optional override for output VLM JSON path, usually outputs/event/cache/vlm/{ref|gen}/<id>.vlm.json")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        video_path=args.video,
        events_json_path=args.events,
        out_json_path=args.out,
        cfg_path=args.config,
    )
