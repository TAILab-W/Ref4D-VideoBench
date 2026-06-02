
import os, re, json, math, argparse, warnings, inspect, traceback
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

import torch
from transformers import AutoModel, AutoTokenizer

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--debug-dump', default='', help='Debug: dump the raw A/B/C/V/F/FB input text for this run into this directory')

    p.add_argument('--video', default='', help='Path to a single video')
    p.add_argument('--out',   default='', help='Output JSON path for a single video')

    p.add_argument('--batch-from', default='', help='Directory, .json, or .jsonl task file for batch mode')
    p.add_argument('--out-dir',    default='', help='Output directory for batch mode; required')

    p.add_argument('--local-path', default='', help='Local model directory, such as /path/to/MiniCPM-V-4_5 or a quantized variant directory')
    p.add_argument('--model-id',   default='openbmb/MiniCPM-V-4_5')
    p.add_argument('--revision',   default='main')
    p.add_argument('--cache-dir',  default='')
    p.add_argument('--local-files-only', action='store_true')

    p.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    p.add_argument('--dtype', default='bf16', choices=['bf16','fp16','fp32'])
    p.add_argument('--disable-flash-sdp', action='store_true', help='Disable Flash SDP for better stability')
    p.add_argument('--force-math-sdp', action='store_true', help='Force math SDP; most stable but slower')

    p.add_argument('--fps', type=int, default=6)
    p.add_argument('--cap-frames', type=int, default=240)
    p.add_argument('--resize-short', type=int, default=448)
    p.add_argument('--max-packing', type=int, default=3)
    p.add_argument('--decode-backend', default='auto', choices=['auto','cv2','decord'])

    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--min-max-new-tokens', type=int, default=96)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--enable-thinking', action='store_true')
    p.add_argument('--verbose', action='store_true')

    p.add_argument('--min-span-sec', type=float, default=0.1, help='Minimum valid time span in seconds; shorter spans will be filtered')

    p.add_argument('--min-fps', type=int, default=2)
    return p

args = None

TIME_SCALE = 0.1
MAX_NUM_FRAMES = 180
MAX_NUM_PACKING = 3
VIDEO_EXTS = {'.mp4','.mkv','.mov','.avi','.m4v','.webm','.flv','.ts','.mpg','.mpeg','.wmv'}
MIN_REL_SPAN = 0.15  
PLURAL_MAP = {
    "person":"people","man":"men","woman":"women","child":"children",
    "cow":"cows","sheep":"sheep","goat":"goats","deer":"deer",
    "bird":"birds","fish":"fish","duck":"ducks","chicken":"chickens","horse":"horses"
}

def _display_plural(name:str, count:int)->str:
    base = _canon_name(name)
    if count >= 10:
        return PLURAL_MAP.get(base, base+"s")
    return base

def _norm_token(s:str) -> str:
    return (s or "").strip().lower().replace(" ", "-")

def _canon_name(n:str) -> str:
    return _norm_token(n)

def _map_to_nearest_scale(values, scale):
    tree=cKDTree(np.asarray(scale)[:,None]); _, idx=tree.query(np.asarray(values)[:,None])
    return np.asarray(scale)[idx]

def _group_array(arr, size):
    return [arr[i:i+size] for i in range(0,len(arr),size)]

def _dtype_of(device,choice):
    if device=='cpu': return torch.float32
    return {'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}.get(choice, torch.bfloat16)

def _guarded_json(s:str):
    if not isinstance(s, str): return None, 'not_str'
    s = s.strip()
    if not s: return None, 'empty'
    try:
        return json.loads(s), None
    except Exception as e:
        m=re.search(r'\{.*\}', s, flags=re.S)
        if m:
            try: return json.loads(m.group(0)), None
            except Exception as e2: return None, f'parse_fail_after_brace:{e2}'
        return None, f'parse_fail:{e}'

def _strip_think(text:str):
    if not isinstance(text,str): return text
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S|re.I)
    text = re.sub(r'^\s*<think>.*$', '', text, flags=re.S|re.I)
    return text.strip()

def _empty_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def _has_nonempty_evidence(json_path:str, min_bytes:int=64) -> bool:
    try:
        if (not os.path.isfile(json_path)) or os.path.getsize(json_path) < min_bytes:
            return False
        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return False
        fine = obj.get("fine")
        if not isinstance(fine, dict):
            return False
        ents = fine.get("entities", [])
        return isinstance(ents, list) and len(ents) > 0
    except Exception:
        return False

def _attr_values_as_iter(vs):
    if vs is None:
        return []
    if isinstance(vs, bool):
        return []
    if isinstance(vs, str):
        return [vs]
    if isinstance(vs, (int, float)):
        return [str(vs)]
    if isinstance(vs, (list, tuple, set)):
        return list(vs)
    return []

def _clean_open_attrs(attrs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    allow_nonalpha_keys = {"number-or-id", "printed-text", "brand-or-logo"}
    for k, vs in (attrs or {}).items():
        k2 = _norm_token(k)
        keep = []
        for v in _attr_values_as_iter(vs):
            if not isinstance(v, str):
                continue
            t = _norm_token(v)
            if not t:
                continue
            if (k2 not in allow_nonalpha_keys) and (not re.search(r'[a-z]', t)):
                continue
            if re.search(r'[a-z]', t) and len(re.sub(r'[^a-z]', '', t)) == 1 and len(t) <= 2:
                continue
            keep.append(t)
        if keep:
            out[k2] = sorted(set(keep))
    return out

def _mk_signature_from_attrs(name:str, attrs:Dict[str,List[str]]) -> str:
    cues=[]
    for k in ['color','pattern','printed-text','number-or-id','brand-or-logo','texture','species-or-breed']:
        cues += (attrs.get(k) or [])[:2]
    for k in ['position','orientation','facing-direction']:
        if (attrs.get(k) or []):
            cues.append(attrs[k][0])
    cues = [_norm_token(x) for x in cues if isinstance(x,str) and x.strip()]
    if not cues:
        return _norm_token(name)
    cues = sorted(set(cues))[:4]
    return "-".join(cues)

def _coerce_spans(x):
    if isinstance(x, (int, float, str)) or x is None:
        return []
    if isinstance(x, (list, tuple)):
        if len(x) == 2 and all(isinstance(t, (int, float)) for t in x):
            return [[float(x[0]), float(x[1])]]
        out = []
        for it in x:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                a, b = it[0], it[1]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    out.append([float(a), float(b)])
        return out
    return []

def _merge_spans(spans: List[List[float]], tol: float = 0.2) -> List[List[float]]:
    if not spans:
        return []
    segs = []
    for a, b in spans:
        try:
            a = float(a); b = float(b)
        except Exception:
            continue
        if b <= a:
            continue
        segs.append([a, b])
    if not segs:
        return []
    segs.sort(key=lambda x: x[0])
    merged = [segs[0]]
    for a, b in segs[1:]:
        if a <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged

def _span_total(spans: List[List[float]]) -> float:
    return sum(max(0.0, float(b)-float(a)) for a,b in (spans or []))

def _span_intersection(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    inter=[]
    for a1,b1 in (A or []):
        for a2,b2 in (B or []):
            s=max(a1,a2); e=min(b1,b2)
            if e>s: inter.append([s,e])
    return _merge_spans(inter)

def _nearest_midpair(A: List[List[float]], B: List[List[float]]):
    best_gap=1e9; best=None
    for a1,b1 in (A or []):
        for a2,b2 in (B or []):
            if b1<=a2:
                gap=a2-b1; pair=(b1,a2)
            elif b2<=a1:
                gap=a1-b2; pair=(b2,a1)
            else:
                return None
            if gap<best_gap:
                best=pair; best_gap=gap
    return best

def _rel_span_fallback(sub_sp: List[List[float]], obj_sp: List[List[float]],
                       start_s: float, end_s: float, min_rel: float) -> List[List[float]]:
    inter = _span_intersection(sub_sp, obj_sp)
    if inter:
        kept = [[round(max(start_s,a),1), round(min(end_s,b),1)] for a,b in inter if (b-a)>=min_rel]
        if kept: return kept
        a,b = inter[0]
        mid=(a+b)/2.0
    else:
        pair=_nearest_midpair(sub_sp, obj_sp)
        if not pair: return []
        mid=(pair[0]+pair[1])/2.0
    half=min_rel/2.0
    a=max(start_s, mid-half); b=min(end_s, mid+half)
    if b>a:
        return [[round(a,1), round(b,1)]]
    return []

def _coerce_task_object(it, src_desc:str):
    if not isinstance(it, dict):
        raise ValueError(f"{src_desc}: each task must be an object with keys 'video' and 'out'")
    vin = it.get('video')
    vout = it.get('out')
    if not isinstance(vin, str) or not vin.strip():
        raise ValueError(f"{src_desc}: task field 'video' must be a non-empty string")
    if not isinstance(vout, str) or not vout.strip():
        raise ValueError(f"{src_desc}: task field 'out' must be a non-empty string")
    return vin, vout

def _discover_inputs(batch_from:str, out_dir:str):
    items=[]
    if not batch_from:
        return items
    if os.path.isdir(batch_from):
        for root, _, fns in os.walk(batch_from):
            for fn in sorted(fns):
                fp=os.path.join(root, fn)
                if os.path.isfile(fp) and os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                    rel=os.path.relpath(fp, batch_from)
                    base=os.path.splitext(rel)[0] + '.json'
                    items.append((fp, os.path.join(out_dir, base)))
        return items

    ext=os.path.splitext(batch_from)[1].lower()
    if ext == '.json':
        with open(batch_from,'r',encoding='utf-8') as f:
            obj=json.load(f)

        task_objs = None
        if isinstance(obj, list):
            task_objs = obj
        elif isinstance(obj, dict) and 'videos' in obj and isinstance(obj['videos'], list):
            task_objs = obj['videos']
        else:
            raise ValueError("JSON task file must be a list of task objects or an object with key 'videos' containing a list of task objects")

        for idx, it in enumerate(task_objs, start=1):
            items.append(_coerce_task_object(it, f"{batch_from}[{idx}]"))        
        return items

    if ext == '.jsonl':
        with open(batch_from,'r',encoding='utf-8') as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                items.append(_coerce_task_object(obj, f"{batch_from}:{lineno}"))        
        return items

    raise ValueError(f"Unsupported --batch-from input: {batch_from}. Expected a directory, .json, or .jsonl task file.")

def _sanitize_out(vin:str, vout:str, out_dir:str) -> str:
    expect = os.path.splitext(os.path.basename(vin))[0] + '.json'
    parent = os.path.dirname(vout) or out_dir or "."
    return os.path.join(parent, expect)

def read_video_meta_and_backend(video_path:str, preference:str='auto'):
    cap = None
    vr  = None
    backend=None; fps=None; total=None; duration=None
    if preference in ('auto','cv2'):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            duration = total/max(fps,1e-6) if total>0 else 0.0
            if total>0:
                backend='cv2'
                return backend, (cap, None), fps, total, duration
        except Exception:
            pass
    if preference in ('auto','decord'):
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total=len(vr)
            try: fps=float(vr.get_avg_fps())
            except Exception: fps=25.0
            duration=total/max(fps,1e-6)
            backend='decord'
            return backend, (None, vr), fps, total, duration
        except Exception:
            pass
    raise RuntimeError("No available video backend (tried cv2, decord)")

def read_frames(backend, handles, idx_all):
    frames=[]
    if backend=='cv2':
        cap,_ = handles
        import cv2
        try: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception: pass
        i=0; want=set(idx_all.tolist())
        while True:
            ok, frame = cap.read()
            if not ok: break
            if i in want:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                if len(frames)>=len(idx_all): break
            i+=1
    else:
        _,vr = handles
        arr=vr.get_batch(idx_all.tolist()).asnumpy()
        frames=[Image.fromarray(x.astype('uint8')).convert('RGB') for x in arr]
    return frames

def encode_clip(meta, start_s:float, end_s:float, choose_fps:int, cap_frames:int, resize_short:int, max_packing:int):
    backend, handles, fps, total, duration = meta
    s=max(0.0,start_s); e=min(duration,end_s)
    if e<=s: e=min(duration,s+1.0)
    clip_len=e-s
    if choose_fps * int(clip_len) <= MAX_NUM_FRAMES:
        packing_nums=1
        choose_frames=round(min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, clip_len))
    else:
        packing_nums=math.ceil(clip_len*choose_fps/MAX_NUM_FRAMES)
        if packing_nums<=max_packing:
            choose_frames=round(clip_len * choose_fps)
        else:
            choose_frames=round(MAX_NUM_FRAMES * max_packing)
            packing_nums=max_packing
    choose_frames=int(min(choose_frames, max(1, cap_frames)))
    start_idx=int(round(s*fps)); end_idx=max(0, int(round(e*fps))-1)
    end_idx=min(total-1, end_idx)
    idx_all=np.linspace(start_idx, end_idx, choose_frames).astype(np.int64)

    ts=idx_all / max(fps,1e-6)
    scale=np.arange(0, duration+1e-6, TIME_SCALE)
    tids=(_map_to_nearest_scale(ts, scale)/TIME_SCALE).astype(np.int32)
    tids=_group_array(tids, packing_nums)

    frames = read_frames(backend, handles, idx_all)
    if resize_short and resize_short>0:
        nf=[]
        for img in frames:
            w,h=img.size; short=min(w,h)
            if short>resize_short:
                if w<h:
                    new_w=resize_short; new_h=int(h*(resize_short/w))
                else:
                    new_h=resize_short; new_w=int(w*(resize_short/h))
                img=img.resize((new_w,new_h), Image.BILINEAR)
            nf.append(img)
        frames=nf
    return frames, tids, (s,e), len(idx_all)

def build_prompts(min_span:float=0.3):
    facets = (
        "color, pattern, texture, material, size, age, sex, state, pose, action, "
        "orientation, facing-direction, position, object-part, tool-or-instrument, equipment, "
        "species-or-breed, vehicle-type, food-type, brand-or-logo, printed-text, number-or-id, "
        "art-medium, style, weather, lighting, scene, camera-view"
    )

    PA = (
        "You will see a set of frames covering absolute time [START, END] seconds.\n\n"
        "List the objects visible in this clip using class name + count + stable time spans (0.1s precision).\n"
        "Requirements: use concise English singular class names (for example person/dog/cat/bird/cow/car/horse). "
        "If the number of objects of the same class is clearly greater than 8, use the English plural form "
        "(for example ants/people) and write count as the string \">8\". For multi-shot videos, count conservatively "
        "at the whole-video level. Spans must stay within [START,END]. Merge adjacent spans when the gap is <= 0.2s, "
        "and ignore noise shorter than 0.1s.\n\n"
        "Output exactly one JSON object:\n"
        "{\n"
        '  "objects":[\n'
        '    {"name":"cow","count":2,"spans":[[ABS_S,ABS_E],...]},\n'
        '    {"name":"person","count":1,"spans":[[ABS_S,ABS_E]]},\n'
        '    {"name":"ants","count":">8","spans":[[ABS_S,ABS_E]]}\n'
        "  ]\n"
        "}"
    )

    PB = (
        "For each concrete visible instance, output attributes and time spans. Enumerate visible instances as completely as possible "
        "(not limited by the counts from A).\n"
        "If an attribute is visible, include it. Use lowercase hyphenated values such as black-white or left-facing. "
        "Do not guess invisible attributes. To distinguish instances, provide a short 'signature' with at least two stable cues "
        "(for example color/pattern/number/text/side).\n"
        f"Suggested slots (open vocabulary): {facets}\n\n"
        "Output exactly one JSON object:\n"
        "{\n"
        '  "entities":[\n'
        '    {\n'
        '      "id":"e1","name":"cow","signature":"black-white-left-ear-tag-37",\n'
        '      "attributes":{"color":["black","white"],"action":["grazing"],"position":["left"],"number-or-id":["37"]},\n'
        '      "spans":[[ABS_S,ABS_E],...]\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    PC = (
        "Based on the entities from B, output clearly visible relation triplets with time spans and confidence.\n"
        "Allowed predicates (open examples): left-of/right-of/above/below/front-of/behind/over/under/inside/overlapping/next-to/"
        "holding/carrying/touching/looking-at/feeding/chasing/following/riding/pulling/pushing/"
        "passing/crossing/entering/exiting/throwing/catching/drinking/eating\n\n"
        "Output exactly one JSON object:\n"
        '{\n  "relations":[{"subject":"e1","predicate":"left-of","object":"e2","spans":[[ABS_S,ABS_E]],"confidence":0.8}]\n}'
        "\nRequirements: subject/object must be ids from B.entities; spans must stay within [START,END] with 0.1s precision; "
        "merge adjacent spans when the gap is <= 0.2s."
    )

    PV = (
        "Consistency correction (output final JSON only). Input contains A/B/C.\n"
        "1) Remove uncertain objects and relations; repair invalid ids.\n"
        "2) Merge overlapping or adjacent spans (gap <= 0.2s); delete spans shorter than __MIN_SPAN__s; clip all spans to [START,END].\n"
        "3) Output only:\n"
        "{\n"
        '  "entities":[{"id":"e1","name":"cow","signature":"...","attributes":{...},"spans":[[ABS_S,ABS_E],... ]}, ...],\n'
        '  "relations":[{"subject":"e1","predicate":"next-to","object":"e2","spans":[[ABS_S,ABS_E]],"confidence":0.8}, ...]\n'
        "}"
    ).replace("__MIN_SPAN__", f"{min_span:.1f}")

    PF = (
        "Fine-grained completion (output final JSON only). Without changing spans, add possibly missing fine-grained attributes "
        "using an open vocabulary. Pay attention to logos/signs/text, numbers or ids, animal patterns or ear tags, clothing or protective gear, "
        "vehicle subtype and state, scene, lighting, and camera view. Only deduplicate and complete."
    )

    FB = (
        "You are a careful video understanding expert. Frames cover [START, END] sec.\n"
        "Return ONE JSON only. No extra text.\n\n"
        "Extract visible object instances and relations. Prefer concise English singular names.\n"
        "Animals you might see (if visible): cow/cattle/ox/bull/calf, horse, sheep, goat, dog, cat, bird, chicken, pig, deer.\n"
        "Attributes (open set): color, pattern, texture, material, size, age, sex, state, pose, action, orientation, facing-direction, position,\n"
        "species-or-breed, printed-text, number-or-id, brand-or-logo, scene, weather, lighting, camera-view.\n"
        "Each entity must include a short 'signature' (>=2 stable cues like colors/pattern/mark/ear-tag/side left/right).\n\n"
        "Schema:\n"
        "{\n"
        '  "entities":[\n'
        '    {"id":"e1","name":"cow","signature":"black-white-left-ear-tag-37",\n'
        '     "attributes":{"color":["black","white"],"action":["grazing"],"position":["left"],"number-or-id":["37"]},\n'
        '     "spans":[[ABS_S,ABS_E]]}\n'
        "  ],\n"
        '  "relations":[{"subject":"e1","predicate":"left-of","object":"e2","spans":[[ABS_S,ABS_E]],"confidence":0.7}]\n'
        "}\n"
        f"ONLY output the JSON; times within [START,END], 0.1s precision; merge adjacent (<=0.2s); drop spans shorter than {min_span:.1f}s."
    )

    return PA, PB, PC, PV, PF, FB

def _filter_spans(spans, start_s, end_s, min_span:float, merge_tol:float=0.2):
    segs=[]
    for a,b in (spans or []):
        try:
            a=float(a); b=float(b)
            a=max(a, start_s); b=min(b, end_s)
            if b>a: segs.append([a,b])
        except:
            continue
    if not segs:
        return []
    segs.sort(key=lambda x: x[0])
    merged=[segs[0]]
    for a,b in segs[1:]:
        if a <= merged[-1][1] + merge_tol:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a,b])
    out = [[round(a,1), round(b,1)] for a,b in merged if (b-a) >= min_span]
    if not out:
        thr = max(0.1, min_span*0.5)
        out = [[round(a,1), round(b,1)] for a,b in merged if (b-a) >= thr]
    return out

def _dtype_kw(d):
    sig=inspect.signature(AutoModel.from_pretrained)
    return {"dtype": d} if "dtype" in sig.parameters else {"torch_dtype": d}

def load_model(where:str, device:str, dtype, cache_dir:str, local_only:bool, revision:str, is_dir:bool):
    common=dict(trust_remote_code=True, attn_implementation='sdpa', **_dtype_kw(dtype))
    if cache_dir: common['cache_dir']=cache_dir
    if local_only: common['local_files_only']=True
    if is_dir:
        m=AutoModel.from_pretrained(where, **common).eval()
        t=AutoTokenizer.from_pretrained(where, trust_remote_code=True, local_files_only=local_only)
    else:
        m=AutoModel.from_pretrained(where, revision=revision, **common).eval()
        t=AutoTokenizer.from_pretrained(where, revision=revision, trust_remote_code=True, **({} if not cache_dir else {'cache_dir': cache_dir}))
    if device=='cuda' and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        if args.disable_flash_sdp or args.force_math_sdp:
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                if args.force_math_sdp:
                    torch.backends.cuda.enable_math_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                else:
                    torch.backends.cuda.enable_math_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
            except Exception:
                pass
        m=m.to('cuda')
    return m,t

def run_clip(model, tok, frames, tids, start_s, end_s,
             max_new_tokens:int, temperature:float, enable_thinking:bool,
             fps:int, cap_frames:int, max_packing:int, resize_short:int,
             prompts:Tuple[str,str,str,str,str,str], min_span_sec:float):

    PA, PB, PC, PV, PF, FB = prompts

    def _chat(prompt:str, thinking:bool):
        p = prompt.replace('[START]', f'{start_s:.1f}').replace('[END]', f'{end_s:.1f}')
        msgs=[{'role':'user','content': frames + [p]}]
        out = model.chat(
            msgs=msgs, tokenizer=tok,
            temporal_ids=tids, use_image_id=False, max_slice_nums=1,
            do_sample=(temperature>0), temperature=(temperature if temperature>0 else None),
            enable_thinking=thinking, max_new_tokens=max_new_tokens
        )
        return _strip_think(out)

    STRICT_SUFFIX = "\nStrict requirement: output exactly one JSON object, with no explanation, comments, or extra characters. If uncertain, output the empty structure shown above."

    def _list_of_dicts(x):
        if isinstance(x, list):
            return [it for it in x if isinstance(it, dict)]
        return []

    def _ask_and_parse(prompt: str, empty_schema: dict, allow_thinking: bool, tag: str = ""):
        txt1 = _chat(prompt, thinking=allow_thinking)
        if args.debug_dump:
            os.makedirs(args.debug_dump, exist_ok=True)
            with open(os.path.join(args.debug_dump, f"clip_{start_s:.1f}_{end_s:.1f}_{tag}_1.txt"), "w", encoding="utf-8") as f:
                f.write(str(txt1))
        obj, _ = _guarded_json(txt1)
        if isinstance(obj, dict):
            return obj
        txt2 = _chat(prompt + STRICT_SUFFIX, thinking=False)
        if args.debug_dump:
            with open(os.path.join(args.debug_dump, f"clip_{start_s:.1f}_{end_s:.1f}_{tag}_2.txt"), "w", encoding="utf-8") as f:
                f.write(str(txt2))
        obj2, _ = _guarded_json(txt2)
        if isinstance(obj2, dict):
            return obj2
        return empty_schema

    cur_tokens=max_new_tokens; cur_temp=temperature
    while True:
        try:
            A = _ask_and_parse(PA, {"objects": []}, allow_thinking=enable_thinking, tag="A")

            hint_from_A = {}
            try:
                for it in (A.get("objects") or []):
                    if not isinstance(it, dict):  
                        continue
                    cname = _canon_name(it.get("name",""))
                    cnt_raw = it.get("count", 0)
                    try:
                        cnt = int(float(cnt_raw))
                    except Exception:
                        cnt = 0
                    if cname:
                        hint_from_A[cname] = max(hint_from_A.get(cname, 0), max(0, cnt))
            except Exception:
                hint_from_A = {}

            ref=json.dumps({"objects":A.get("objects",[])}, ensure_ascii=False)

            B = _ask_and_parse(PB+"\n\n"+ref, {"entities":[]}, allow_thinking=enable_thinking, tag="B")
            ents=json.dumps({"entities":B.get("entities",[])}, ensure_ascii=False)

            C = _ask_and_parse(PC+"\n\n"+ents, {"relations":[]}, allow_thinking=enable_thinking, tag="C")

            payload=json.dumps({"A":A,"B":B,"C":C}, ensure_ascii=False)
            V = _ask_and_parse(PV+"\n\n"+payload,
                               {"entities":B.get("entities",[]), "relations":C.get("relations",[])},
                               allow_thinking=enable_thinking, tag="V")

            V2 = _ask_and_parse(
                PF+"\n\n"+json.dumps({"entities":V.get("entities",[]), "relations":V.get("relations",[])}, ensure_ascii=False),
                {"entities":V.get("entities",[]), "relations":V.get("relations",[])},
                allow_thinking=enable_thinking, tag="F"
            )

            ents_out=[]
            for e in _list_of_dicts(V2.get("entities", [])):
                spans=_filter_spans(_coerce_spans(e.get("spans")), start_s, end_s, min_span_sec)
                if not spans:
                    continue

                raw_attrs_obj = e.get("attributes") if isinstance(e.get("attributes"), dict) else {}
                raw_attrs = { _norm_token(k): [ _norm_token(x) for x in _attr_values_as_iter(vs) if isinstance(x,str) ]
                              for k,vs in raw_attrs_obj.items() }
                attrs = _clean_open_attrs(raw_attrs)

                name_raw = e.get("name","")
                name = _canon_name(name_raw if isinstance(name_raw, str) else "")

                sig_raw = e.get("signature","")
                sig = _norm_token(sig_raw if isinstance(sig_raw, str) else "") or _mk_signature_from_attrs(name, attrs)
                if sig:
                    attrs.setdefault("signature", [sig])

                ents_out.append({
                    "id": str(e.get("id","")),
                    "name": name,
                    "attributes": attrs,
                    "spans": spans
                })

            rels_out=[]
            for r in _list_of_dicts(V2.get("relations", [])):
                spans=_filter_spans(_coerce_spans(r.get("spans")), start_s, end_s, min_span_sec)
                if not spans:
                    continue
                conf_raw = r.get("confidence", 0.0)
                try:
                    conf_val = float(conf_raw)
                except Exception:
                    conf_val = 0.0
                rels_out.append({
                    "subject": str(r.get("subject","")),
                    "predicate": _norm_token(r.get("predicate","")),
                    "object": str(r.get("object","")),
                    "spans": spans,
                    "confidence": conf_val
                })

            if not ents_out:
                FB_obj = _ask_and_parse(prompts[-1], {"entities":[],"relations":[]}, allow_thinking=enable_thinking, tag="FB")

                for e in _list_of_dicts(FB_obj.get("entities", [])):
                    spans=_filter_spans(_coerce_spans(e.get("spans")), start_s, end_s, min_span_sec)
                    if not spans:
                        continue
                    raw_attrs_obj = e.get("attributes") if isinstance(e.get("attributes"), dict) else {}
                    raw_attrs = { _norm_token(k): [ _norm_token(x) for x in _attr_values_as_iter(vs) if isinstance(x,str) ]
                                  for k,vs in raw_attrs_obj.items() }
                    attrs = _clean_open_attrs(raw_attrs)
                    name_raw = e.get("name","")
                    name = _canon_name(name_raw if isinstance(name_raw, str) else "")
                    sig_raw = e.get("signature","")
                    sig = _norm_token(sig_raw if isinstance(sig_raw, str) else "") or _mk_signature_from_attrs(name, attrs)
                    if sig:
                        attrs.setdefault("signature", [sig])
                    ents_out.append({
                        "id": str(e.get("id","")),
                        "name": name,
                        "attributes": attrs,
                        "spans": spans
                    })

                for r in _list_of_dicts(FB_obj.get("relations", [])):
                    spans=_filter_spans(_coerce_spans(r.get("spans")), start_s, end_s, min_span_sec)
                    if not spans:
                        continue
                    conf_raw = r.get("confidence", 0.0)
                    try: conf_val = float(conf_raw)
                    except: conf_val = 0.0
                    rels_out.append({
                        "subject": str(r.get("subject","")),
                        "predicate": _norm_token(r.get("predicate","")),
                        "object": str(r.get("object","")),
                        "spans": spans,
                        "confidence": conf_val
                    })

                if not ents_out:
                    rescue=[]
                    for it in (A.get("objects") or []):
                        if not isinstance(it, dict):  
                            continue
                        cname = _canon_name(it.get("name",""))
                        try: cnt = int(float(it.get("count", 0)))
                        except: cnt = 0
                        for j in range(max(0, cnt)):
                            rescue.append({
                                "id": f"r{j+1}",
                                "name": cname,
                                "attributes": {"signature":[f"{cname}-coarse"]},
                                "spans": [[round(start_s,1), round(end_s,1)]]
                            })
                    return rescue, [], hint_from_A

            return ents_out, rels_out, hint_from_A

        except torch.cuda.OutOfMemoryError as e:
            print(f"[OOM] {e}")
            _empty_cuda()
            if cur_tokens>args.min_max_new_tokens:
                cur_tokens=max(args.min_max_new_tokens, int(cur_tokens*0.75))
                print(f"[OOM] reduce max_new_tokens -> {cur_tokens}")
            elif cur_temp>0.0:
                cur_temp=0.0
                print(f"[OOM] set temperature -> 0")
            else:
                raise
        except RuntimeError:
            raise

def process_one_video(video_path:str, out_path:str, model, tok, prompts):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        if _has_nonempty_evidence(out_path):
            print(f"[SKIP] exists -> {out_path}")
            return
    except Exception:
        pass

    meta = read_video_meta_and_backend(video_path, preference=args.decode_backend)
    backend, handles, fps, total, duration = meta
    print(f"[Video] {os.path.basename(video_path)} | backend={backend} | duration={duration:.2f}s | fps={fps:.2f} | frames={total}")
    print(f"[Cfg]   min_span_sec={args.min_span_sec}")
    print(f"[Plan]  full-video input")

    fps_cur    = args.fps
    cap_cur    = args.cap_frames
    tokens_cur = args.max_new_tokens
    temp_cur   = args.temperature
    pack_cur   = MAX_NUM_PACKING
    resize_cur = args.resize_short

    fine_entities = []
    all_rels = []

    while True:
        try:
            s, e = 0.0, duration
            frames, tids, (cs,ce), kframes = encode_clip(meta, s, e, fps_cur, cap_cur, resize_cur, pack_cur)
            ents, rels, hint_from_A = run_clip(
                model, tok, frames, tids, cs, ce, tokens_cur, temp_cur, args.enable_thinking,
                fps_cur, cap_cur, pack_cur, resize_cur, prompts, args.min_span_sec
            )

            local2global = {}
            clip_entities = []
            clip_relations = []

            local_index = {"id2info":{}, "class2ids":{}, "order_ids":[]}

            for idx, eobj in enumerate((ents or []), start=1):
                if not isinstance(eobj, dict):
                    continue

                name = _canon_name(eobj.get("name",""))
                if not name:
                    if hint_from_A and len([k for k,v in hint_from_A.items() if v > 0]) == 1:
                        name = next(k for k,v in hint_from_A.items() if v > 0)
                    elif hint_from_A and sum(hint_from_A.values()) == 1:
                        name = max(hint_from_A, key=hint_from_A.get)
                    else:
                        continue

                spans = [[float(a), float(b)] for a,b in (eobj.get("spans") or []) if isinstance(a,(int,float)) and isinstance(b,(int,float))]
                spans = _merge_spans(spans)
                attrs = eobj.get("attributes", {}) or {}
                if not isinstance(attrs, dict):
                    attrs = {}

                gid = f"o{idx}"
                local_id = str(eobj.get("id","")).strip() or f"e{idx}"
                local2global[local_id] = gid

                if local_id not in local_index["id2info"]:
                    local_index["id2info"][local_id] = {"name": name, "spans": spans}
                    local_index["order_ids"].append(local_id)
                    local_index["class2ids"].setdefault(name, []).append(local_id)

                ent_rec = {
                    "id": gid,
                    "name": name,
                    "attributes": _clean_open_attrs(attrs),
                    "spans": [[round(cs,1), round(ce,1)]] if not spans else [[round(a,1), round(b,1)] for a,b in spans]
                }
                clip_entities.append(ent_rec)
                fine_entities.append(ent_rec)

            def _resolve_rel_endpoint(token, rel_spans, idx):
                t = str(token if token is not None else "").strip()
                if not t:
                    return ""
                if t in idx["id2info"]:
                    return t
                m = re.match(r'^([a-zA-Z][\w-]*)[ #_\-]*([0-9]+)$', t)
                if m:
                    cls = _canon_name(m.group(1))
                    n = int(m.group(2))
                    ids = idx["class2ids"].get(cls, [])
                    if 1 <= n <= len(ids):
                        return ids[n-1]
                cls = _canon_name(t)
                ids = idx["class2ids"].get(cls, [])
                if len(ids) == 1:
                    return ids[0]
                if len(ids) > 1:
                    if rel_spans:
                        best, best_sc = "", -1.0
                        for lid in ids:
                            inter = _span_intersection(idx["id2info"][lid]["spans"], rel_spans)
                            sc = _span_total(inter)
                            if sc > best_sc:
                                best, best_sc = lid, sc
                        if best_sc > 0:
                            return best
                    return max(ids, key=lambda z: _span_total(idx["id2info"][z]["spans"]))
                m2 = re.search(r'([0-9]+)$', t)
                if m2:
                    n = int(m2.group(1))
                    if 1 <= n <= len(idx["order_ids"]):
                        return idx["order_ids"][n-1]
                return ""

            rel_saved = rel_drop_id = rel_drop_span = 0
            for r in (rels or []):
                if not isinstance(r, dict):
                    continue
                raw_rspan = _filter_spans(_coerce_spans(r.get("spans")), cs, ce, args.min_span_sec)
                sid_l = _resolve_rel_endpoint(r.get("subject",""), raw_rspan, local_index)
                oid_l = _resolve_rel_endpoint(r.get("object",""), raw_rspan, local_index)
                if not sid_l or not oid_l:
                    rel_drop_id += 1
                    continue

                r_spans = raw_rspan
                if not r_spans:
                    sub_sp = local_index["id2info"][sid_l]["spans"]
                    obj_sp = local_index["id2info"][oid_l]["spans"]
                    r_spans = _rel_span_fallback(sub_sp, obj_sp, cs, ce, MIN_REL_SPAN)
                    if not r_spans:
                        rel_drop_span += 1
                        continue

                sid = local2global.get(sid_l, "")
                oid = local2global.get(oid_l, "")
                if not sid or not oid:
                    rel_drop_id += 1
                    continue

                conf_raw = r.get("confidence", 0.0)
                try:
                    conf_val = float(conf_raw)
                except Exception:
                    conf_val = 0.0
                pred = _norm_token(r.get("predicate",""))

                rel_rec = {
                    "subject": sid,
                    "predicate": pred,
                    "object": oid,
                    "spans": [[round(a,1), round(b,1)] for a,b in _merge_spans(r_spans)],
                    "confidence": conf_val
                }
                clip_relations.append(rel_rec)
                all_rels.append(rel_rec)
                rel_saved += 1

            if args.verbose:
                print(f"[REL] full-video: saved={rel_saved} drop_id={rel_drop_id} drop_span={rel_drop_span}")
            if args.verbose:
                print(f"[OK] full-video {cs:.1f}-{ce:.1f}s frames={kframes} ent={len(clip_entities)} rel={len(clip_relations)}")
            break

        except torch.cuda.OutOfMemoryError:
            if fps_cur > args.min_fps:
                fps_cur = max(args.min_fps, fps_cur - 1); print(f"[OOM] retry fps={fps_cur}")
            elif cap_cur > 96:
                cap_cur = max(96, int(cap_cur * 0.75)); print(f"[OOM] retry cap_frames={cap_cur}")
            elif tokens_cur > args.min_max_new_tokens:
                tokens_cur = max(args.min_max_new_tokens, int(tokens_cur * 0.75)); print(f"[OOM] retry max_new_tokens={tokens_cur}")
            elif resize_cur > 320:
                resize_cur = max(320, int(resize_cur * 0.85)); print(f"[OOM] retry resize_short={resize_cur}")
            elif pack_cur > 1:
                pack_cur = max(1, pack_cur - 1); print(f"[OOM] retry max_packing={pack_cur}")
            else:
                raise
            _empty_cuda()
        except Exception as ex:
            print(f"[WARN] full-video failed: video={video_path} out={out_path} exc={type(ex).__name__}: {ex}")
            for line in traceback.format_exc().rstrip().splitlines():
                print(f"[TRACE] {line}")
            break

    obj_counts = {}
    for ent in fine_entities:
        obj_counts[ent["name"]] = obj_counts.get(ent["name"], 0) + 1
    display_counts = { _display_plural(k, v): v for k,v in obj_counts.items() }

    out = {
        "meta":{
            "video_basename": os.path.basename(video_path),
            "fps_per_window": args.fps, "resize_short": args.resize_short,
            "model": args.model_id or "local",
            "backend": backend, "dtype": args.dtype,
            "min_span_sec": args.min_span_sec
        },
        "fine":{ "entities": fine_entities, "relations": all_rels },
        "views":{
            "objects_count_display": display_counts,
        }
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved -> {out_path}")

def main():
    global args, MAX_NUM_PACKING, MIN_REL_SPAN
    args = build_arg_parser().parse_args()
    MAX_NUM_PACKING = max(1, min(args.max_packing, 6))
    MIN_REL_SPAN = max(0.05, float(args.min_span_sec) * 0.5)

    single_mode = bool(args.video)
    batch_mode  = bool(args.batch_from)

    if not single_mode and not batch_mode:
        raise SystemExit("Provide --video/--out for single-file mode or --batch-from/--out-dir for batch mode")

    if single_mode and not args.out:
        raise SystemExit("--out cannot be empty in single-file mode")

    if batch_mode:
        if not args.out_dir:
            raise SystemExit("--out-dir cannot be empty in batch mode")
        os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu'
    dtype  = _dtype_of(device, args.dtype)

    where = args.local_path if args.local_path else args.model_id
    is_dir = bool(args.local_path and os.path.isdir(args.local_path))
    model, tok = load_model(where, device, dtype, args.cache_dir, args.local_files_only, args.revision, is_dir)

    prompts = build_prompts(args.min_span_sec)

    if single_mode:
        if not os.path.exists(args.video):
            raise FileNotFoundError(args.video)
        
        expect = os.path.splitext(os.path.basename(args.video))[0] + '.json'
        out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
        out_final = os.path.join(out_dir, expect)
        os.makedirs(out_dir, exist_ok=True)
        process_one_video(args.video, out_final, model, tok, prompts)
        return

    tasks = _discover_inputs(args.batch_from, args.out_dir)
    if not tasks:
        raise SystemExit(f"No videos found in batch mode: {args.batch_from}")
    
    tasks = [(vin, _sanitize_out(vin, vout, args.out_dir)) for vin, vout in tasks]

    print(f"[Batch] total videos: {len(tasks)}")
    for i,(vin, vout) in enumerate(tasks, start=1):
        try:
            print(f"\n=== [{i}/{len(tasks)}] {vin} ===")
            if not os.path.exists(vin):
                raise FileNotFoundError(vin)
            os.makedirs(os.path.dirname(vout), exist_ok=True)
            process_one_video(vin, vout, model, tok, prompts)
        except Exception as e:
            print(f"[ERROR] {vin} failed: {e}")

if __name__=="__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6')
        main()
