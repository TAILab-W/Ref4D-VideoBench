# -*- coding: utf-8 -*-

import os, re, json, math, argparse, warnings, inspect, traceback
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image

import torch
from transformers import AutoModel, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--json-dir',  default='', help='Rule JSON directory (filenames usually like <stem>_rules.json)')
p.add_argument('--video-dir', default='', help='Video directory (same stem as JSON, different extension)')
p.add_argument('--out-dir',   default='', help='Output directory (writes <video_stem>_assertions.json)')

p.add_argument('--local-path', default='', help='Local model directory, e.g. /data/hf_home/models/openbmb__MiniCPM-V-4_5')
p.add_argument('--model-id',   default='openbmb/MiniCPM-V-4_5')
p.add_argument('--revision',   default='main')
p.add_argument('--cache-dir',  default='')
p.add_argument('--local-files-only', action='store_true')

p.add_argument('--device', default='cuda', choices=['cuda','cpu'])
p.add_argument('--dtype',  default='bf16', choices=['bf16','fp16','fp32'])
p.add_argument('--disable-flash-sdp', dest='disable_flash_sdp', action='store_true')
p.add_argument('--force-math-sdp',    dest='force_math_sdp',    action='store_true')

p.add_argument('--fps',           type=int,   default=3,   help='Global frame extraction target fps')
p.add_argument('--cap-frames',    type=int,   default=300, help='Max total frames sent to model (global cap)')
p.add_argument('--resize-short',  type=int,   default=448, help='Resize short edge to this size (0 = no resize)')
p.add_argument('--decode-backend', default='auto', choices=['auto','cv2','decord'])

p.add_argument('--max-new-tokens', type=int, default=512)
p.add_argument('--temperature',    type=float, default=0.0)
p.add_argument('--enable-thinking', action='store_true')
p.add_argument('--verbose', action='store_true')

p.add_argument('--max-rules-in-prompt', type=int, default=12)

p.add_argument('--multi-turn', dest='multi_turn', action='store_true', help='Use conversation history during fallback/supplement so later assertions are influenced by earlier context')
p.add_argument('--single-turn', dest='multi_turn', action='store_false',help='Disable conversation history, use single-turn fallback (legacy behavior)')
p.set_defaults(multi_turn=True)

p.add_argument('--dump-raw', action='store_true', help='Save and print each raw model output')
p.add_argument('--raw-dir',  default='', help='Raw output save directory (default out-dir/debug/<video_stem>/)')

args = p.parse_args()

TIME_SCALE = 0.1
MAX_NUM_FRAMES = 450
MAX_NUM_PACKING = 3

SCHEMA_STR = """
[
  {
    "id": "a1",
    "asstertion_text": "Write a complete, human-readable rule in English. Prefer an 'During..., ...' or '...should...' sentence.",
    "required_singal": ["<from signal list>"]
  }
]
""".strip()

VIDEO_EXTS = {'.mp4', '.mkv', '.mov', '.avi', '.m4v', '.webm'}

def _strip_think(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S|re.I).strip()

def _strip_code_fences_keep_inner(text: str) -> str:
    """
    Remove code fence markers (```xxx and ```), keep inner text.
    Also discard pure fence lines (only ``` or ```lang).
    """
    if not isinstance(text, str):
        text = str(text)
    lines = text.replace('\r', '\n').split('\n')
    out_lines = []
    fence = False
    for ln in lines:
        if re.match(r'^\s*```', ln):
            fence = not fence
            continue
        out_lines.append(ln)
    return '\n'.join(out_lines)

def _dtype_of(device: str, choice: str):
    if device == 'cpu':
        return torch.float32
    return {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[choice]

def _dtype_kw(d):
    sig = inspect.signature(AutoModel.from_pretrained)
    return {"dtype": d} if "dtype" in sig.parameters else {"torch_dtype": d}

def _read_json(fp: str) -> Dict[str, Any]:
    with open(fp, 'r', encoding='utf-8') as f:
        return json.load(f)

def _ensure_dir_dir(dir_path: str):
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

def _ensure_dir_for_file(file_path: str):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

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
            try:
                fps = float(vr.get_avg_fps())
            except Exception:
                fps = 25.0
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
        if len(idx_all) == 0:
            return frames
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx_all[0]))
        except Exception:
            pass
        cur = int(idx_all[0])
        want = set(int(i) for i in idx_all.tolist())
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if cur in want:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                if len(frames) >= len(idx_all):
                    break
            cur += 1
    else:
        _, vr = handles
        if len(idx_all) == 0:
            return frames
        arr = vr.get_batch(idx_all.tolist()).asnumpy()
        frames = [Image.fromarray(x.astype('uint8')).convert('RGB') for x in arr]
    return frames

def _temporal_ids(ts: np.ndarray) -> List[List[int]]:
    grid = np.round(ts / TIME_SCALE).astype(np.int32)
    n = len(grid)
    packing = min(MAX_NUM_PACKING, max(1, int(math.ceil(n / MAX_NUM_FRAMES))))
    chunks = []
    for k in range(packing):
        part = grid[k::packing]
        chunks.append(part.tolist())
    return chunks

def encode_whole_video(meta, choose_fps: int, cap_frames: int,
                       resize_short: int) -> Tuple[List[Image.Image], List[List[int]], Tuple[float, float], int]:
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
                    new_w = resize_short
                    new_h = int(h * (resize_short / w))
                else:
                    new_h = resize_short
                    new_w = int(w * (resize_short / h))
                img = img.resize((new_w, new_h), Image.BILINEAR)
            nf.append(img)
        frames = nf

    tids = _temporal_ids(ts)
    return frames, tids, (s, e), len(idx_all)

def load_model(where: str, device: str, dtype, cache_dir: str, local_only: bool, revision: str, is_dir: bool):
    common = dict(trust_remote_code=True, attn_implementation='sdpa', **_dtype_kw(dtype))
    if cache_dir:
        common['cache_dir'] = cache_dir
    if local_only:
        common['local_files_only'] = True

    if is_dir:
        m = AutoModel.from_pretrained(where, **common).eval()
        t = AutoTokenizer.from_pretrained(where, trust_remote_code=True, local_files_only=local_only)
    else:
        m = AutoModel.from_pretrained(where, revision=revision, **common).eval()
        t = AutoTokenizer.from_pretrained(
            where, revision=revision, trust_remote_code=True,
            **({} if not cache_dir else {'cache_dir': cache_dir})
        )

    if device == 'cuda' and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            if args.disable_flash_sdp:
                torch.backends.cuda.enable_flash_sdp(False)
            if args.force_math_sdp:
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            else:
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass
        m = m.to('cuda')

    return m, t

_SIGNAL_WEIGHT = {
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

def _prefix(sig: str) -> str:
    s = (sig or '').strip().lower()
    s = re.split(r'[\(\s]', s)[0]
    s = s.replace('/', '_').replace('-', '_').replace(' ', '_').replace('.', '_')
    s = re.sub(r'[^a-z0-9_]', '_', s)
    return s or '_default'

def _rule_quality(rule: Dict[str, Any]) -> float:
    sigs = rule.get('required_signals') or []
    if not isinstance(sigs, list):
        return 0.0
    score = 0.0
    for s in sigs:
        key = _prefix(s)
        score += _SIGNAL_WEIGHT.get(key, _SIGNAL_WEIGHT['_default'])
    score += 0.15 * max(0, len(sigs) - 1)
    kinds = set(_prefix(s).split('_')[0] for s in sigs if isinstance(s, str))
    score += 0.1 * max(0, len(kinds) - 1)
    return score

def _select_rules(rule_json: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    all_rules = [r for r in (rule_json.get('rules') or []) if isinstance(r, dict)]
    for r in all_rules:
        r['_quality'] = _rule_quality(r)
    all_rules.sort(key=lambda r: r.get('_quality', 0.0), reverse=True)
    return all_rules[:k]

BASE_INSTRUCTION = """
You must generate assertions based on the given reference video. These assertions must be TRUE for that reference video and directly verifiable from the visuals.

[Background]
Combine the high-level rules below (ranked from higher to lower importance) with the entire reference video to produce assertions for later judging whether an AIGC-generated video conforms to world knowledge and everyday common sense (e.g., object size inconsistency, violation of gravity, object interpenetration, lens/frame discontinuities, unreasonable scene colors and lighting, unreasonable material brittleness/deformation, etc.). Each assertion must:
refer only to entities/events that visibly appear in the video;
state an objective, observable fact that is TRUE in this reference video;
be specific and falsifiable by visual inspection (frame-by-frame if necessary).

[Clip] {CLIP_RANGE} (global)

[Available Rules (id + rule_text; higher ranks preferred)]
{RULES_RANKED_BLOCK}
# Note: each line is in the form "rID: <rule_text>". In your outputs, cite one or more rule ids at the start of each assertion line using [rID] tags.

[Strict Output Requirements]
Output at most FIVE lines.
Each line must start with one or more rule id tags， followed by the assertion text: 
[rID][rID2] assertion_text
Do NOT include extra explanations, titles, numbering, blank lines, code fences, or <think> tags.

[Assertion Quality Requirements]
Must be TRUE in this reference video and have visible grounding.
Be specific and easy to verify; avoid opinions, hypotheticals, and modal verbs (e.g., “should/might/could”).
Prefer higher-ranked rules or rule combinations; avoid repetition; keep each line ≤ 28 English words.
Use clear English, affirmative sentences in the simple present tense.

[Examples]
[rID]"Throughout the video, <entity> remains <state>."
[rID][rID2]"During..., ... should ..."
""".strip()

def _compact_rules_for_prompt(rule_json: Dict[str, Any]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    rules = rule_json.get("rules", []) or []
    id_map: Dict[str, Dict[str, Any]] = {}
    for r in rules:
        rid = str(r.get("id") or "").strip()
        if not rid:
            continue
        id_map[rid] = {
            "id": rid,
            "rule_text": r.get("rule_text"),
            "required_signals": r.get("required_signals") or []
        }
    k = min(args.max_rules_in_prompt, len(id_map))
    chosen = _select_rules({"rules": list(id_map.values())}, k)
    lines = [f"- {r.get('id')}: {r.get('rule_text')}" for r in chosen]
    block = "\n".join(lines)
    return block, id_map

def build_user_prompt_from_rules(rule_json: Dict[str, Any],
                                 clip_info: Tuple[float, float]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    s, e = clip_info
    rules_block, id_map = _compact_rules_for_prompt(rule_json)
    prompt = BASE_INSTRUCTION.format(
        CLIP_RANGE=f"{s:.1f}-{e:.1f}s",
        RULES_RANKED_BLOCK=rules_block
    )
    return prompt, id_map

def _default_raw_dir_for_video(out_dir: str, video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(out_dir, "debug", stem)

def _save_raw(text: str, path: str, head_print: int = 2000, tag: str = "RAW OUT"):
    try:
        _ensure_dir_for_file(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text if isinstance(text, str) else str(text))
    except Exception:
        pass
    print(f"[{tag}] {text[:head_print]}{'...' if isinstance(text, str) and len(text)>head_print else ''}", flush=True)

_TAG_RE = re.compile(r'\[(r[0-9a-zA-Z_\-]+)\]')

def _normalize_tokens(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s\-_/]', ' ', s)
    toks = [t for t in s.split() if t]
    return toks

def _is_semantic_duplicate(a: str, b: str, jaccard_thresh: float = 0.8) -> bool:
    A, B = set(_normalize_tokens(a)), set(_normalize_tokens(b))
    if not A or not B:
        return a.strip().lower() == b.strip().lower()
    inter = len(A & B)
    union = max(1, len(A | B))
    j = inter / union
    return j >= jaccard_thresh

def _clean_text_before_parse(text: str) -> str:
    t = _strip_think(text)
    t = _strip_code_fences_keep_inner(t)
    return t

def _parse_tagged_lines(text: str) -> List[Tuple[List[str], str]]:
    """
    Parse model raw text into lines:
      Return [(rule_ids, assertion_text), ...]
    Lines may have prefix numbering like "1. ", "1、", "① ", etc.; all ignored.
    """
    if not isinstance(text, str):
        text = str(text)
    text = _clean_text_before_parse(text)

    lines = [ln.strip() for ln in text.replace('\r', '\n').split('\n') if ln.strip()]
    out: List[Tuple[List[str], str]] = []
    for ln in lines:
        ln = re.sub(r'^\s*(\d+[\.\)])\s*', '', ln)
        # Strictly reject JSON header prompts and other noise
        if ln.strip().lower().startswith(('json', 'output:', 'answer:', 'assertions:', 'result:')):
            continue
        tags = _TAG_RE.findall(ln)
        if not tags:
            continue
        body = _TAG_RE.sub('', ln).strip()
        body = re.sub(r'^[\-\:：、.\s]+', '', body).strip()
        if not body:
            continue
        uniq_ids: List[str] = []
        seen = set()
        for t in tags:
            if t not in seen:
                uniq_ids.append(t)
                seen.add(t)
        out.append((uniq_ids, body))
    return out

def _assemble_results(tagged: List[Tuple[List[str], str]],
                      rules_by_id: Dict[str, Dict[str, Any]],
                      *,
                      start_index: int = 1,
                      dedup: bool = True) -> List[Dict[str, Any]]:
    """
    No longer auto-pad to 5 items; only return actually retrieved items (max 5).
    """
    result: List[Dict[str, Any]] = []
    texts_so_far: List[str] = []

    for ids, body in tagged:
        body_clean = body.strip()
        if not body_clean:
            continue
        if dedup:
            dup = False
            for prev in texts_so_far:
                if _is_semantic_duplicate(prev, body_clean):
                    dup = True
                    break
            if dup:
                continue
        signals: List[str] = []
        sig_seen = set()
        for rid in ids:
            r = rules_by_id.get(rid)
            if not r:
                continue
            for s in (r.get("required_signals") or []):
                s_str = str(s)
                if s_str not in sig_seen:
                    signals.append(s_str)
                    sig_seen.add(s_str)
        result.append({
            "id": "",
            "asstertion_text": body_clean,
            "required_singal": signals
        })
        texts_so_far.append(body_clean)

        if len(result) >= 5:
            break

    for i, it in enumerate(result, start_index):
        it["id"] = f"a{i}"

    return result[:5]

def _basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _drop_rules_suffix(name: str) -> str:
    if name.lower().endswith('_rules'):
        return name[: -len('_rules')]
    return name

def _video_candidates(video_dir: str) -> Dict[str, str]:
    cand: Dict[str, str] = {}
    for fn in os.listdir(video_dir):
        full = os.path.join(video_dir, fn)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext not in VIDEO_EXTS:
            continue
        stem = _basename_no_ext(fn)
        key = stem.lower()
        if key not in cand or len(fn) < len(os.path.basename(cand[key])):
            cand[key] = full
    return cand

def _best_match_video(video_dir: str, rule_stem: str) -> Tuple[Optional[str], Optional[str]]:
    cand = _video_candidates(video_dir)
    target = rule_stem.lower()
    if target in cand:
        vp = cand[target]
        return vp, _basename_no_ext(vp)
    prefix_hits = [(k, v) for k, v in cand.items() if k.startswith(target)]
    if prefix_hits:
        prefix_hits.sort(key=lambda kv: (len(kv[0]), len(os.path.basename(kv[1]))))
        vp = prefix_hits[0][1]
        return vp, _basename_no_ext(vp)
    contain_hits = [(k, v) for k, v in cand.items() if target in k]
    if contain_hits:
        contain_hits.sort(key=lambda kv: (len(kv[0]), len(os.path.basename(kv[1]))))
        vp = contain_hits[0][1]
        return vp, _basename_no_ext(vp)
    return None, None

def _process_one(rule_json_path: str, video_path: str, out_json_path: str,
                 model, tok, args):
    backend, handles, fps, total, duration = read_video_meta_and_backend(
        video_path, preference=args.decode_backend
    )
    if args.verbose:
        print(f"[Video] {os.path.basename(video_path)} | {duration:.2f}s | fps={fps:.2f} | frames={total} | backend={backend}")

    rulej = _read_json(rule_json_path)

    frames, tids, (cs, ce), _ = encode_whole_video(
        (backend, handles, fps, total, duration),
        args.fps, min(args.cap_frames, MAX_NUM_FRAMES),
        args.resize_short
    )

    user_prompt, rules_by_id_all = build_user_prompt_from_rules(rulej, (cs, ce))

    raw_dir = args.raw_dir or _default_raw_dir_for_video(args.out_dir, video_path)
    if args.dump_raw:
        _ensure_dir_dir(raw_dir)

    def _make_user_msg(prompt: str, *, include_frames: bool = True) -> Dict[str, Any]:
        return {'role': 'user', 'content': (frames + [prompt]) if include_frames else [prompt]}

    def _call_model(msgs: List[Dict[str, Any]], *, tag: str, round_idx: int = 1,
                    max_new_tokens: Optional[int] = None, use_image_id: bool = False) -> str:
        out = model.chat(
            msgs=msgs,
            tokenizer=tok,
            temporal_ids=tids,
            use_image_id=use_image_id,
            max_slice_nums=1,
            do_sample=False,
            temperature=None,
            enable_thinking=False,
            max_new_tokens=(max_new_tokens if max_new_tokens is not None else args.max_new_tokens)
        )
        raw = out if isinstance(out, str) else str(out)
        if args.dump_raw:
            fn = "raw_first.txt" if tag == "first" and round_idx == 1 else f"raw_round_{round_idx:02d}.txt"
            _save_raw(raw, os.path.join(raw_dir, fn), tag=f"RAW OUT {tag.upper()}#{round_idx}")
        return _clean_text_before_parse(raw)

    multi_turn = getattr(args, "multi_turn", True)

    msgs: List[Dict[str, Any]] = []
    msgs.append(_make_user_msg(user_prompt, include_frames=True))
    text_first = _call_model(msgs, tag="first", round_idx=1, use_image_id=False)
    tagged = _parse_tagged_lines(text_first)

    msgs.append({'role': 'assistant', 'content': text_first})

    RETRY_LIMIT = 5
    round_idx = 1

    def _build_continue_prompt(current_tagged: List[Tuple[List[str], str]], want: int) -> str:
        exist_texts = [t for _, t in current_tagged]
        exist_block = "\n".join(f"- {t}" for t in exist_texts) if exist_texts else "(none)"
        return (
            "Continue. Based on the previous assistant output above and the SAME global video, "
            "output ONLY NEW assertions. Each line must start with rule tags like [r1][r5] "
            "followed by the assertion text. Avoid overlap, paraphrases, or minor rewording "
            "of earlier assertions. Prefer higher-ranked rules and complementary combinations "
            "not yet covered.\n\n"
            "[Already generated assertions (DO NOT repeat)]:\n" + exist_block +
            f"\n\nOutput up to {want} new lines. No JSON, no explanations, no code fences."
        )

    while len(tagged) < 5 and round_idx < RETRY_LIMIT:
        round_idx += 1
        remain = 5 - len(tagged)

        if multi_turn:
            cont_prompt = _build_continue_prompt(tagged, remain)
            msgs.append(_make_user_msg(cont_prompt, include_frames=False))
            text_more = _call_model(
                msgs, tag="dialogue", round_idx=round_idx,
                max_new_tokens=args.max_new_tokens + 128,
                use_image_id=True
            )
            msgs.append({'role': 'assistant', 'content': text_more})
        else:
            exist_texts = [t for _, t in tagged]
            exist_block = "\n".join(f"- {t}" for t in exist_texts) if exist_texts else "(none)"
            refill_prompt = (
                user_prompt
                + "\n\n[Already generated assertions (DO NOT repeat; avoid paraphrases)]:\n"
                + exist_block
                + f"\n\nNow output ONLY NEW lines, each starting with rule tags like [r1][r5] "
                  "then the assertion text. Do NOT include JSON, explanations, or code fences. "
                  f"Output up to {remain} lines."
            )
            msgs_single = [_make_user_msg(refill_prompt, include_frames=True)]
            text_more = _call_model(
                msgs_single, tag="retry", round_idx=round_idx,
                max_new_tokens=args.max_new_tokens + 128,
                use_image_id=False
            )

        add_tagged = _parse_tagged_lines(text_more)

        def _already_has(body: str) -> bool:
            for _, b in tagged:
                if _is_semantic_duplicate(b, body):
                    return True
            return False

        for ids, body in add_tagged:
            if not _already_has(body):
                tagged.append((ids, body))

        if args.verbose:
            print(f"[Retry/Dialogue] round={round_idx}, now_have={len(tagged)}", flush=True)

    result = _assemble_results(tagged, rules_by_id_all, start_index=1, dedup=True)

    _ensure_dir_for_file(out_json_path)
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if args.verbose:
        print(f"[OK] -> {out_json_path} | items={len(result)}")

def main():
    if not args.json_dir or not args.video_dir or not args.out_dir:
        raise ValueError("Must provide --json-dir, --video-dir, --out-dir")

    _ensure_dir_dir(args.out_dir)

    device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    dtype = _dtype_of(device, args.dtype)

    where = args.local_path if args.local_path else args.model_id
    is_dir = bool(args.local_path and os.path.isdir(args.local_path))
    model, tok = load_model(where, device, dtype, args.cache_dir, args.local_files_only, args.revision, is_dir)

    json_files = sorted(
        f for f in (os.path.join(args.json_dir, p) for p in os.listdir(args.json_dir))
        if os.path.isfile(f) and f.lower().endswith('.json')
    )
    if args.verbose:
        print(f"[Batch] json files found = {len(json_files)}")

    miss_video, done, fail = 0, 0, 0
    for jpath in json_files:
        base = _basename_no_ext(jpath)
        stem_rule = _drop_rules_suffix(base)

        vpath, vstem = _best_match_video(args.video_dir, stem_rule)
        if not vpath:
            miss_video += 1
            print(f"[Skip] No video matched for rule '{base}' (stem='{stem_rule}') under {args.video_dir}")
            continue

        out_json = os.path.join(args.out_dir, f"{vstem}_assertions.json")
        try:
            _process_one(jpath, vpath, out_json, model, tok, args)
            done += 1
        except Exception as e:
            fail += 1
            print(f"[Error] {vstem} <- {base}: {e}")
            if args.verbose:
                traceback.print_exc()

    print(f"[Summary] success={done}, missing_video={miss_video}, failed={fail}")

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6')
        main()