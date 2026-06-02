# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import os, re, json, math, argparse, warnings, inspect, glob
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image

import torch
from transformers import AutoModel, AutoTokenizer

p = argparse.ArgumentParser()

p.add_argument('--video-file', help='Full video file (mp4/mkv/mov/avi/m4v/webm)')
p.add_argument('--vqa-json',   help='VQA JSON (text and ID only; compatible with multiple field names)')
p.add_argument('--assert-json',help='Assertion JSON (text and ID only; compatible with multiple field names)')
p.add_argument('--out-file',   help='Output scored JSON file path')

p.add_argument('--list-file', help='Video list file: one video absolute/relative path per line')
p.add_argument('--video-dir', help='Video directory (auto-scan common video extensions)')
p.add_argument('--vqa-dir',   help='VQA JSON directory (matches <stem>_VQA.json)')
p.add_argument('--assert-dir',help='Assertion JSON directory (matches <stem>_assertion(.json|s.json))')
p.add_argument('--out-dir',   help='Batch output root directory (per video generates <out-dir>/<stem>/<stem>_scored.json)')

p.add_argument('--local-path', default='', help='Local model directory, e.g. /data/hf_home/models/openbmb__MiniCPM-V-4_5')
p.add_argument('--model-id',   default='openbmb/MiniCPM-V-4_5')
p.add_argument('--revision',   default='main')
p.add_argument('--cache-dir',  default='')
p.add_argument('--local-files-only', action='store_true')

p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
p.add_argument('--dtype',  default='bf16', choices=['bf16','fp16','fp32'])
p.add_argument('--disable-flash-sdp', dest='disable_flash_sdp', action='store_true')
p.add_argument('--force-math-sdp',    dest='force_math_sdp',    action='store_true')

p.add_argument('--fps',           type=int,   default=3,   help='Global frame extraction target fps')
p.add_argument('--cap-frames',    type=int,   default=300, help='Max total frames sent to model (global cap)')
p.add_argument('--resize-short',  type=int,   default=448, help='Resize short edge to this size (0 = no resize)')
p.add_argument('--decode-backend', default='auto', choices=['auto','cv2','decord'])

p.add_argument('--max-new-tokens', type=int, default=256)
p.add_argument('--temperature',    type=float, default=0.0)
p.add_argument('--enable-thinking', action='store_true')
p.add_argument('--verbose', action='store_true')

p.add_argument('--dump-raw', dest='dump_raw', action='store_true', help='Save raw model scoring output fragments to result JSON')
p.add_argument('--no-dump-raw', dest='dump_raw', action='store_false')
p.set_defaults(dump_raw=True)

p.add_argument('--multi-turn', dest='multi_turn', action='store_true',
               help='Evaluate multiple items for the same video in a shared conversation history (default on)')
p.add_argument('--no-multi-turn', dest='multi_turn', action='store_false',
               help='Disable multi-turn, evaluate each item independently')
p.set_defaults(multi_turn=True)

args = p.parse_args()

TIME_SCALE = 0.1
MAX_NUM_FRAMES = 450
VIDEO_EXTS = {'.mp4', '.mkv', '.mov', '.avi', '.m4v', '.webm'}

_SIGNAL_WEIGHT: Dict[str, float] = {
  "interpenetration": 2.7,
  "collision": 2.2,
  "post_collision_change": 2.5,
  "center_of_mass_over_base": 2.3,
  "shadow_consistency": 2.8,
  "size_inconsistency": 2.9,
  "reflection_consistency": 2.6,
  "biological_motion": 3.0,
  "material_optics": 2.9,
  "deformation": 3.2,
  "fracture": 2.2,
  "support": 2.6,
  "balance": 2.7,
  "trajectory_dir": 3.0,
  "accel_sign": 2.6,
  "rigid_transform": 2.9,
  "tilt_angle_threshold": 2.7,
  "flame_proximity": 2.5,
  "cast_shadow_contact": 2.9,
  "friction_sliding": 2.0,
  "wheel_rolling": 2.0,
  "hue_histogram_jump": 2.4,
  "hot_surface_contact": 2.5,
  "slosh_response": 2.5,
  "buoyancy_float": 2.5,
  "specular_highlight_consistency": 2.8,
  "smoke_direction_consistency": 2.8,
  "frame_discontinuity": 3.3,
  "continuity": 2.8,
  "compressibility": 2.1,
  "leakage": 1.9,
  "grasp_stability": 1.8,
  "tool_contact": 1.8,
  "spillage": 1.8,
  "viscosity_effect": 2.5,
  "pour_direction": 1.8,
  "slip": 1.7,
  "step_order": 1.6,
  "handoff_between_hands": 1.6,
  "state_change": 2.9,
  "tool_usage_correct": 1.9,
  "goal_placement": 1.6,
  "result_appearance": 1.8,
  "action_phase": 1.6,
  "supportable": 1.6,
  "placeable": 1.5,
  "pourable": 1.5,
  "openable": 1.5,
  "containment": 1.9,
  "prohibited_intersection": 2.8,
  "falling_hazard": 2.2,
  "finger_clearance": 2.5,
  "blade_orientation_away": 1.9,
  "_default": 1.2
}
_ATTEN = [1.00, 0.20, 0.20, 0.2, 0.2]
def _atten(k: int) -> float: return _ATTEN[k] if k < len(_ATTEN) else 0.20

_TAG_RE = re.compile(r'\[(r[0-9a-zA-Z_\-]+)\]')

def _strip_think(text: str) -> str:
    if not isinstance(text, str): return text
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S|re.I).strip()

def _strip_code_fences_keep_inner(text: str) -> str:
    if not isinstance(text, str): text = str(text)
    lines = text.replace('\r', '\n').split('\n')
    out_lines, fence = [], False
    for ln in lines:
        if re.match(r'^\s*```', ln):
            fence = not fence
            continue
        out_lines.append(ln)
    return '\n'.join(out_lines)

def _clean_text_before_parse(text: str) -> str:
    return _strip_code_fences_keep_inner(_strip_think(text))

def _dtype_of(device: str, choice: str):
    if device == 'cpu': return torch.float32
    return {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[choice]

def _dtype_kw(d):
    sig = inspect.signature(AutoModel.from_pretrained)
    return {"dtype": d} if "dtype" in sig.parameters else {"torch_dtype": d}

def _ensure_dir_for_file(file_path: str):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

def _ensure_dir(path: str):
    os.makedirs(path or ".", exist_ok=True)

def read_video_meta_and_backend(video_path: str, preference: str = 'auto'):
    cap = None; vr = None
    backend = None; fps = None; total = None; duration = None
    if preference in ('auto', 'cv2'):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            duration = total / max(fps, 1e-6) if total > 0 else 0.0
            if total > 0:
                backend = 'cv2'
                return backend, (cap, None), fps, total, duration
        except Exception:
            pass
    if preference in ('auto', 'decord'):
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            try: fps = float(vr.get_avg_fps())
            except Exception: fps = 25.0
            duration = total / max(fps, 1e-6)
            backend = 'decord'
            return backend, (None, vr), fps, total, duration
        except Exception:
            pass
    raise RuntimeError(f"No available video backend for {video_path} (tried cv2, decord)")

def read_frames(backend: str, handles, idx_all: np.ndarray) -> List[Image.Image]:
    frames: List[Image.Image] = []
    if backend == 'cv2':
        cap, _ = handles
        import cv2
        if len(idx_all) == 0: return frames
        try: cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx_all[0]))
        except Exception: pass
        cur = int(idx_all[0]); want = set(int(i) for i in idx_all.tolist())
        while True:
            ok, frame = cap.read()
            if not ok: break
            if cur in want:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                if len(frames) >= len(idx_all): break
            cur += 1
    else:
        _, vr = handles
        if len(idx_all) == 0: return frames
        arr = vr.get_batch(idx_all.tolist()).asnumpy()
        frames = [Image.fromarray(x.astype('uint8')).convert('RGB') for x in arr]
    return frames

def _temporal_ids(ts: np.ndarray) -> List[List[int]]:
    grid = np.round(ts / TIME_SCALE).astype(np.int32)
    n = len(grid)
    packing = min(3, max(1, int(math.ceil(n / MAX_NUM_FRAMES))))
    chunks = []
    for k in range(packing):
        part = grid[k::packing]
        chunks.append(part.tolist())
    return chunks

def encode_whole_video(meta, choose_fps: int, cap_frames: int, resize_short: int):
    backend, handles, fps, total, duration = meta
    s, e = 0.0, duration
    clip_len = max(1e-6, e - s)
    want = int(min(cap_frames, round(clip_len * min(choose_fps, int(round(fps))))))
    start_idx = 0
    end_idx   = max(0, total - 1)
    if end_idx <= start_idx:
        end_idx = min(total - 1, start_idx + max(1, want))
    idx_all = np.linspace(start_idx, end_idx, max(1, want)).astype(np.int64)
    ts = idx_all / max(fps, 1e-6)
    frames = read_frames(backend, handles, idx_all)
    if resize_short and resize_short > 0:
        nf = []
        for img in frames:
            w, h = img.size
            short = min(w, h)
            if short > resize_short:
                if w < h:
                    new_w = resize_short; new_h = int(h * (resize_short / w))
                else:
                    new_h = resize_short; new_w = int(w * (resize_short / h))
                img = img.resize((new_w, new_h), Image.BILINEAR)
            nf.append(img)
        frames = nf
    tids = _temporal_ids(ts)
    return frames, tids, (s, e), len(idx_all)

def load_model(where: str, device: str, dtype, cache_dir: str, local_only: bool, revision: str, is_dir: bool):
    common = dict(trust_remote_code=True, attn_implementation='sdpa', **_dtype_kw(dtype))
    if cache_dir: common['cache_dir'] = cache_dir
    if local_only: common['local_files_only'] = True
    if is_dir:
        m = AutoModel.from_pretrained(where, **common).eval()
        t = AutoTokenizer.from_pretrained(where, trust_remote_code=True, local_files_only=local_only)
    else:
        m = AutoModel.from_pretrained(where, revision=revision, **common).eval()
        t = AutoTokenizer.from_pretrained(where, revision=revision, trust_remote_code=True,
                                          **({} if not cache_dir else {'cache_dir': cache_dir}))
    if device == 'cuda' and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            if args.disable_flash_sdp: torch.backends.cuda.enable_flash_sdp(False)
            if args.force_math_sdp:
                torch.backends.cuda.enable_math_sdp(True); torch.backends.cuda.enable_mem_efficient_sdp(False)
            else:
                torch.backends.cuda.enable_math_sdp(True); torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception: pass
        m = m.to('cuda')
    return m, t

def _read_json_any(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _norm_text_item(x: Dict[str, Any], *, default_prefix: str, kind: str) -> Dict[str, Any]:
    if not isinstance(x, dict):
        x = {"text": str(x)}
    _id = str(x.get("id") or x.get("idx") or x.get("index") or "").strip()
    if not _id:
        _id = f"{default_prefix}{x.get('_i', 0)+1}"
    text = (x.get("text") or x.get("name") or x.get("question") or x.get("vqa") or
            x.get("vqa_text") or x.get("asstertion_text") or x.get("assertion") or
            x.get("prompt") or "").strip()
    sigs = (x.get("required_singal") or x.get("signals") or x.get("required_signals")
            or x.get("signal") or x.get("signal_words") or [])
    if isinstance(sigs, str):
        tmp = re.split(r'[,\;\s]+', sigs.strip())
        sigs = [s for s in tmp if s]
    if not isinstance(sigs, list): sigs = []
    sigs = [str(s).strip() for s in sigs if str(s).strip()]
    return {"id": _id, "text": text, "signals": (sigs if sigs else ["_default"]), "type": kind}

def _extract_array(payload: Any) -> List[Any]:
    if isinstance(payload, list): return payload
    if isinstance(payload, dict):
        for k in ["items", "data", "list", "vqa", "assertions", "results"]:
            if isinstance(payload.get(k), list):
                return payload[k]
    return []

def _normalize_inputs(vqa_json_path: str, assert_json_path: str) -> List[Dict[str, Any]]:
    vqa_raw = _read_json_any(vqa_json_path)
    asr_raw = _read_json_any(assert_json_path)
    vqa_arr = _extract_array(vqa_raw) if not isinstance(vqa_raw, list) else vqa_raw
    asr_arr = _extract_array(asr_raw) if not isinstance(asr_raw, list) else asr_raw
    vqa_norm, asr_norm = [], []
    for i, x in enumerate(vqa_arr, 1):
        xx = dict(x or {}); xx["_i"] = i
        vqa_norm.append(_norm_text_item(xx, default_prefix="q", kind="VQA"))
    for i, x in enumerate(asr_arr, 1):
        xx = dict(x or {}); xx["_i"] = i
        asr_norm.append(_norm_text_item(xx, default_prefix="a", kind="assertion"))
    merged = vqa_norm + asr_norm
    return [m for m in merged if m.get("text")]

_PROMPT_RATE_60 = """
You will evaluate the QUALITY of a single item (question or assertion) about the full video.

Scoring rules (0–60, integer only):
- The more it tests higher-level video understanding about world commonsense, physical laws, and causal consistency (e.g., no sudden limb twisting, consistent lighting/shadow direction, stable camera/viewpoint, no object interpenetration, physically plausible motion, etc.—physics, causality, safety, affordances, temporal consistency), the higher the score.
- The closer it aligns with real-world commonsense and the more concretely it can be verified/falsified on the video, the higher the score.
- If it is vague, low-level (mere color/shape naming), weakly tied to the scene, or cannot be answered based on the reference video, give a lower score.
- If the assertion or question does not match/align with the reference video, give a lower score.
- Output ONLY one integer in [0,60]. No extra text.

Item:
- id: {ID}
- text: {TEXT}
""".strip()

_INT_RE = re.compile(r'(?<!\d)([0-5]?\d|60)(?!\d)')

def _clean(text: str) -> str:
    return _clean_text_before_parse(text)

def _call_model_for_score_single(frames: List[Image.Image], tids: List[List[int]], model, tok,
                                 *, item_id: str, text: str) -> Tuple[int, str]:
    prompt = _PROMPT_RATE_60.format(ID=item_id, TEXT=text)
    msgs = [{'role': 'user', 'content': frames + [prompt]}]
    out = model.chat(
        msgs=msgs, tokenizer=tok, temporal_ids=tids, use_image_id=False, max_slice_nums=1,
        do_sample=(args.temperature > 0), temperature=(args.temperature if args.temperature > 0 else None),
        enable_thinking=args.enable_thinking, max_new_tokens=args.max_new_tokens
    )
    raw = _clean(out if isinstance(out, str) else str(out))
    m = _INT_RE.search(raw.strip())
    if m:
        val = int(m.group(1))
        return max(0, min(60, val)), (raw if args.dump_raw else "")
    return _heuristic_fallback_score(text), (raw if args.dump_raw else "")

def _call_model_for_score_multi(frames: List[Image.Image], tids: List[List[int]], model, tok,
                                *, chat_history: List[Dict[str, Any]],
                                item_id: str, text: str) -> Tuple[int, str]:
    prompt = _PROMPT_RATE_60.format(ID=item_id, TEXT=text)
    msgs = list(chat_history) + [{'role': 'user', 'content': frames + [prompt]}]
    out = model.chat(
        msgs=msgs, tokenizer=tok, temporal_ids=tids, use_image_id=False, max_slice_nums=1,
        do_sample=(args.temperature > 0), temperature=(args.temperature if args.temperature > 0 else None),
        enable_thinking=args.enable_thinking, max_new_tokens=args.max_new_tokens
    )
    raw = _clean(out if isinstance(out, str) else str(out))
    m = _INT_RE.search(raw.strip())
    score = int(m.group(1)) if m else _heuristic_fallback_score(text)
    chat_history.append({'role': 'user', 'content': f"[rate-60] id={item_id}\n{text}"})
    chat_history.append({'role': 'assistant', 'content': str(raw).strip()})
    return max(0, min(60, score)), (raw if args.dump_raw else "")

def _heuristic_fallback_score(text: str) -> int:
    heur = 0; t = (text or "").lower()
    hi_kw = ["during","should","consistent","frame","shadow","occlusion","gravity","collide","support",
             "balance","trajectory","speed","abrupt","discontinuity","causal","cause","effect","safety",
             "affordance","reflection","lighting","specular","deform","orientation","pose","temporal",
             "across frames","world"]
    heur += sum(1 for k in hi_kw if k in t) * 3
    heur += 8 if ("should" in t or "must" in t) else 0
    L = len(t.split())
    if 8 <= L <= 28: heur += 12
    elif 5 <= L < 8 or 28 < L <= 40: heur += 7
    return int(max(0, min(60, heur)))

def _normalize_tokens(s: str) -> List[str]:
    s = s.lower(); s = re.sub(r'[^a-z0-9\s\-_/]', ' ', s)
    return [t for t in s.split() if t]

def _is_semantic_duplicate(a: str, b: str, jaccard_thresh: float = 0.8) -> bool:
    A, B = set(_normalize_tokens(a)), set(_normalize_tokens(b))
    if not A or not B: return a.strip().lower() == b.strip().lower()
    inter = len(A & B); union = max(1, len(A | B))
    return (inter / union) >= jaccard_thresh

def _apply_duplicate_penalty(items_sorted: List[Dict[str, Any]], *, penalty: float = 0.6):
    seen_texts: List[str] = []
    for it in items_sorted:
        txt = it["text"]
        dup = any(_is_semantic_duplicate(txt, prev) for prev in seen_texts)
        if dup:
            it["content_score_60"] = int(round(it["content_score_60"] * penalty))
            it["duplicate_penalized"] = True
        else:
            it["duplicate_penalized"] = False
            seen_texts.append(txt)

def _sig_prefix(sig: str) -> str:
    s = (sig or '').strip().lower()
    s = re.split(r'[\(\s]', s)[0]
    s = s.replace('/', '_').replace('-', '_').replace(' ', '_').replace('.', '_')
    s = re.sub(r'[^a-z0-9_]', '_', s)
    return s or '_default'

_FULL_SCORE = 40.0
_SLOPE = 10.0 / 1.2           # ≈ 8.3333333333
_INTERCEPT = 40.0 - _SLOPE*3.2 # ≈ 13.3333333333

_SECONDARY_FACTOR = 0.2

def _map_base_to_score(base: float) -> float:
    """Map raw weight base (e.g. 3.2/3.0/2.5/2.0/...) to a single signal score 0~40."""
    try:
        x = float(base)
    except Exception:
        x = 0.0
    val = _SLOPE * x + _INTERCEPT
    if val < 0.0: val = 0.0
    if val > _FULL_SCORE: val = _FULL_SCORE
    return val

def _signal_score_40(signals: List[str]) -> Tuple[float, Dict[str, float]]:

    if not isinstance(signals, list) or not signals:
        signals = ["_default"]

    mapped: List[Tuple[str, float]] = []
    for s in signals:
        key = _sig_prefix(s)
        base = float(_SIGNAL_WEIGHT.get(key, _SIGNAL_WEIGHT.get("_default", 1.2)))
        s_i = _map_base_to_score(base)
        mapped.append((s, s_i))

    mapped.sort(key=lambda x: x[1], reverse=True)

    total = 0.0
    detail: Dict[str, float] = {}
    for i, (raw_s, s_i) in enumerate(mapped):
        if i == 0:
            contrib = s_i
        else:
            contrib = _SECONDARY_FACTOR * s_i
        total += contrib
        detail[raw_s] = round(contrib, 4)

    if total > _FULL_SCORE:
        total = _FULL_SCORE

    return round(total, 2), detail


def run_single(video_file: str, vqa_json: str, assert_json: str, out_file: str,
               *, model, tok):
    meta = read_video_meta_and_backend(video_file, preference=args.decode_backend)
    backend, handles, fps, total, duration = meta
    if args.verbose:
        print(f"[Video] {os.path.basename(video_file)} | {duration:.2f}s | fps={fps:.2f} | frames={total} | backend={backend}")
    frames, tids, (cs, ce), _ = encode_whole_video(meta, args.fps, min(args.cap_frames, MAX_NUM_FRAMES), args.resize_short)

    items = _normalize_inputs(vqa_json, assert_json)
    if args.verbose:
        print(f"[Input] items loaded = {len(items)} (VQA + assertions)")

    scored: List[Dict[str, Any]] = []
    chat_history: List[Dict[str, Any]] = []
    if args.multi_turn and args.verbose:
        print("[Mode] multi-turn enabled: shared conversation history (text only, no repeated frames)")

    for it in items:
        if args.multi_turn:
            sc, raw = _call_model_for_score_multi(frames, tids, model, tok,
                                                  chat_history=chat_history,
                                                  item_id=it["id"], text=it["text"])
        else:
            sc, raw = _call_model_for_score_single(frames, tids, model, tok,
                                                   item_id=it["id"], text=it["text"])
        new_it = dict(it)
        new_it["content_score_60"] = int(sc)
        if args.dump_raw:
            new_it["content_score_raw"] = (raw or "")[:1000]
        scored.append(new_it)

    _apply_duplicate_penalty(scored, penalty=0.6)

    for it in scored:
        s40, detail = _signal_score_40(it.get("signals") or ["_default"])
        it["signal_score_40"] = round(float(s40), 2)
        it["signal_detail"] = {k: round(v, 4) for k, v in detail.items()}

    bank_ids: List[str] = []
    for it in scored:
        total = float(it["content_score_60"]) + float(it["signal_score_40"])
        it["total_score_100"] = round(total, 2)
        it["enter_bank"] = bool(total >= 80.0)
        if it["enter_bank"]:
            bank_ids.append(it["id"])

    out = {
        "summary": {
            "items": len(scored),
            "enter_bank": len(bank_ids),
            "clip_range_sec": [round(cs, 2), round(ce, 2)],
            "mode": "multi-turn" if args.multi_turn else "single-turn"
        },
        "items": [{
            "id": it["id"],
            "type": it["type"],                 # "VQA" / "assertion"
            "name": it["text"],
            "signals": it.get("signals", []),
            "content_score_60": it["content_score_60"],
            "signal_score_40": it["signal_score_40"],
            "total_score_100": it["total_score_100"],
            "duplicate_penalized": it.get("duplicate_penalized", False),
            "signal_detail": it.get("signal_detail", {}),
            **({"content_score_raw": it.get("content_score_raw", "")} if args.dump_raw else {})
        } for it in scored],
        "bank_ids": bank_ids
    }

    _ensure_dir_for_file(out_file)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if args.verbose:
        print(f"[OK] -> {out_file} | bank={len(bank_ids)}/{len(scored)}")

def _stem(path: str) -> str:
    b = os.path.basename(path)
    return os.path.splitext(b)[0]

def _guess_pair_paths(stem: str, vqa_dir: str, assert_dir: str) -> Optional[Tuple[str, str]]:
    vqa_path = os.path.join(vqa_dir, f"{stem}_VQA.json") if vqa_dir else None
    cand = []
    if assert_dir:
        cand += glob.glob(os.path.join(assert_dir, f"{stem}_assertion.json"))
        cand += glob.glob(os.path.join(assert_dir, f"{stem}_assertions.json"))
    asr_path = cand[0] if cand else None
    if vqa_path and os.path.isfile(vqa_path) and asr_path and os.path.isfile(asr_path):
        return vqa_path, asr_path
    return None

def _iter_videos_from_dir(video_dir: str) -> List[str]:
    outs = []
    for root, _, files in os.walk(video_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in VIDEO_EXTS:
                outs.append(os.path.join(root, fn))
    outs.sort()
    return outs

def main():
    single_mode = bool(args.video_file and args.vqa_json and args.assert_json and args.out_file)
    batch_mode  = bool((args.list_file or args.video_dir) and args.vqa_dir and args.assert_dir and args.out_dir)

    if not single_mode and not batch_mode:
        raise SystemExit(
            "Usage error:\n"
            "- Single file: must provide --video-file --vqa-json --assert-json --out-file\n"
            "- Batch: must provide (--list-file or --video-dir) and --vqa-dir --assert-dir --out-dir"
        )

    device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    dtype  = _dtype_of(device, args.dtype)
    where  = args.local_path if args.local_path else args.model_id
    is_dir = bool(args.local_path and os.path.isdir(args.local_path))
    model, tok = load_model(where, device, dtype, args.cache_dir, args.local_files_only, args.revision, is_dir)

    if single_mode:
        run_single(args.video_file, args.vqa_json, args.assert_json, args.out_file, model=model, tok=tok)
        return

    videos: List[str] = []
    if args.list_file:
        with open(args.list_file, 'r', encoding='utf-8') as f:
            videos = [ln.strip() for ln in f if ln.strip()]
    elif args.video_dir:
        videos = _iter_videos_from_dir(args.video_dir)

    if args.verbose:
        print(f"[Batch] videos = {len(videos)}")

    for vf in videos:
        if not os.path.isfile(vf):
            print(f"[Skip] video not found: {vf}")
            continue
        stem = _stem(vf)
        pair = _guess_pair_paths(stem, args.vqa_dir, args.assert_dir)
        if not pair:
            print(f"[Skip] missing VQA/_assertion for stem={stem}")
            continue
        vqa_json, assert_json = pair
        _ensure_dir(args.out_dir)
        out_file = os.path.join(args.out_dir, f"{stem}_scored.json")

        if args.verbose:
            print(f"[Run] {stem} -> {out_file}")
        run_single(vf, vqa_json, assert_json, out_file, model=model, tok=tok)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6')
        main()
