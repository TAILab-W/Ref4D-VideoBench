#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-video prompt generator with PySceneDetect integration.

Default repository paths:
- Video root: data/refvideo
- Evidence root: data/metadata/semantic_event_evidence
- Prompt output JSONL: data/metadata/ref4d_prompts.jsonl
- MiniCPM model root: checkpoints/minicpm-v-4_5
"""

import os
import json
import argparse
import warnings
import re
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
VIDEO_ROOT = DATA_DIR / "refvideo"
EVIDENCE_ROOT = METADATA_DIR / "semantic_event_evidence"
PROMPT_ROOT = METADATA_DIR / "_prompt_tmp"
PROMPT_JSONL_PATH = METADATA_DIR / "ref4d_prompts.jsonl"
MODEL_ROOT = PROJECT_ROOT / "checkpoints" / "minicpm-v-4_5"

for _dir in (VIDEO_ROOT, METADATA_DIR, EVIDENCE_ROOT, PROMPT_ROOT, MODEL_ROOT):
    _dir.mkdir(parents=True, exist_ok=True)

try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
    print("[INFO] PySceneDetect imported successfully; automatic shot-change detection is available.")
except ImportError:
    print("[ERROR] PySceneDetect is not installed. Run: pip install scenedetect")
    SCENEDETECT_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

def _dtype_of(device, choice):
    if device == 'cpu': 
        return torch.float32
    return {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}.get(choice, torch.bfloat16)

def _strip_think(text: str):
    if not isinstance(text, str): 
        return text
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S | re.I)
    text = re.sub(r'^\s*<think>.*$', '', text, flags=re.S | re.I)
    return text.strip()

# ---------- Imports and dependencies ----------
def is_english_text(text: str) -> bool:
    """Is english text helper."""
    import re
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    return not chinese_pattern.search(text)

def validate_multishot_format(text: str) -> bool:
    """Validate multishot format helper."""
    import re
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    shot_pattern = re.compile(r'^shot\d+:\s*.+', re.IGNORECASE)
    
    if len(lines) == 0:
        return False
    
    for line in lines:
        if not shot_pattern.match(line):
            return False
    
    return True

def validate_prompt_output(text: str, is_multishot: bool) -> tuple[bool, str]:
    """Validate prompt output helper."""
    if not text or not text.strip():
        return False, "Output is empty"
    
    if not is_english_text(text):
        return False, "Output contains Chinese characters"
    
    if is_multishot:
        if not validate_multishot_format(text):
            return False, "Invalid multi-shot format; expected prefixes such as shot1: and shot2:"
    
    return True, "Validation passed"


def _sanitize_evidence_for_prompt(data, *, max_list_items: int = 80):
    """Drop dense numeric vectors before sending reference evidence to the VLM."""
    vector_keys = {
        "emb",
        "embed",
        "embeds",
        "embedding",
        "embeddings",
        "event_embedding",
        "feature",
        "features",
        "vector",
        "vectors",
    }
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if str(key).lower() in vector_keys:
                continue
            cleaned[key] = _sanitize_evidence_for_prompt(value, max_list_items=max_list_items)
        return cleaned
    if isinstance(data, list):
        if all(isinstance(x, (int, float)) for x in data):
            return None
        cleaned_items = [_sanitize_evidence_for_prompt(x, max_list_items=max_list_items) for x in data[:max_list_items]]
        return [x for x in cleaned_items if x is not None]
    return data

# ---------- Validation helpers ----------
def read_video_with_cv2(video_path: str):
    """Read video with cv2 helper."""
    if not CV2_AVAILABLE:
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        duration = total_frames / max(fps, 1e-6) if total_frames > 0 else 0.0
        cap.release()
        
        if total_frames > 0:
            return {'backend': 'cv2', 'fps': fps, 'total_frames': total_frames, 'duration': duration}
    except Exception as e:
        print(f"[WARNING] CV2 read failed: {e}")
    
    return None

def read_video_with_decord(video_path: str):
    """Read video with decord helper."""
    if not DECORD_AVAILABLE:
        return None
    
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        try:
            fps = float(vr.get_avg_fps())
        except:
            fps = 25.0
        duration = total_frames / max(fps, 1e-6)
        
        return {'backend': 'decord', 'fps': fps, 'total_frames': total_frames, 'duration': duration}
    except Exception as e:
        print(f"[WARNING] Decord read failed: {e}")
    
    return None

def get_video_info(video_path: str):
    """Get video info helper."""
    info = read_video_with_cv2(video_path)
    if info:
        return info
    
    info = read_video_with_decord(video_path)
    if info:
        return info
    
    raise RuntimeError("Unable to read video information; check the video file or install cv2/decord.")

def extract_frames_from_video(video_path: str, start_time: float, end_time: float, target_frames: int = 8):
    """Extract frames from video helper."""
    frames = []
    
    if CV2_AVAILABLE:
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            frame_indices = np.linspace(start_frame, end_frame, target_frames, dtype=int)
            
            current_frame = 0
            frame_idx = 0
            
            while cap.isOpened() and frame_idx < len(frame_indices):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if current_frame == frame_indices[frame_idx]:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                    frame_idx += 1
                
                current_frame += 1
            
            cap.release()
            return frames
        except Exception as e:
            print(f"[WARNING] CV2 frame extraction failed: {e}")
    
    if DECORD_AVAILABLE:
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            
            start_frame = int(start_time * fps)
            end_frame = min(int(end_time * fps), len(vr) - 1)
            
            frame_indices = np.linspace(start_frame, end_frame, target_frames, dtype=int)
            frame_array = vr.get_batch(frame_indices.tolist()).asnumpy()
            
            for frame in frame_array:
                pil_image = Image.fromarray(frame.astype('uint8')).convert('RGB')
                frames.append(pil_image)
            
            return frames
        except Exception as e:
            print(f"[ERROR] Decord frame extraction failed: {e}")
    
    raise RuntimeError("Unable to extract video frames")

# ---------- Video decoding helpers ----------
def detect_shots_with_scenedetect(video_path: str, threshold: float = 35.0, min_shot_length: float = 3.0):
    """Detect shots with scenedetect helper."""
    
    if not SCENEDETECT_AVAILABLE:
        print("[ERROR] PySceneDetect is unavailable")
        return None
    
    try:
        print(f"[INFO] Detecting shot changes with PySceneDetect...")
        print(f"[INFO] Detection parameters: threshold={threshold}, min_shot_length={min_shot_length}")
        
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        video_manager.set_downscale_factor()
        
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        video_manager.release()
        
        print(f"[INFO] Detected {len(scene_list)} scenes")
        
        if len(scene_list) <= 1:
            print(f"[INFO] No shot changes detected")
            return None
        
        valid_shots = []
        for i, (start_time, end_time) in enumerate(scene_list):
            duration = end_time.get_seconds() - start_time.get_seconds()
            if duration >= min_shot_length:
                shot_info = {
                    'shot_id': i + 1,
                    'start_time': start_time.get_seconds(),
                    'end_time': end_time.get_seconds(),
                    'duration': duration
                }
                valid_shots.append(shot_info)
                print(f"[Shot {i+1}] {start_time.get_seconds():.1f}s - {end_time.get_seconds():.1f}s ({duration:.1f}s)")
        
        if len(valid_shots) <= 1:
            print(f"[INFO] Not enough valid shots")
            return None
        
        print(f"[INFO] Detected {len(valid_shots)} valid shots")
        return valid_shots
        
    except Exception as e:
        print(f"[ERROR] Shot detection failed: {e}")
        return None

# ---------- Shot detection ----------
def load_model(model_path: str, device: str, dtype):
    """Load model helper."""
    print(f"[INFO] Loading model: {model_path}")
    
    try:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if device == 'cuda' and torch.cuda.is_available():
            model = model.to('cuda')
        
        print(f"[INFO] Model loaded")
        return model, tokenizer
        
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        raise


# ---------- Model loading ----------
def generate_single_shot_description(model, tokenizer, frames, video_json_data):
    """Generate single shot description helper."""
    
    import json
    json_reference = ""
    if video_json_data:
        prompt_json_data = _sanitize_evidence_for_prompt(video_json_data)
        json_reference = json.dumps(prompt_json_data, indent=2, ensure_ascii=False)
    
    prompt = f"""
Observe the video content and generate a natural English video description prompt.

## Reference JSON Data:
{json_reference if json_reference else "No reference data available"}

## Task:
Generate a comprehensive English video description that covers all attribute categories mentioned in the JSON while adding appropriate visual details you observe.

## Requirements:
- **Complete JSON Coverage**: Include information about EVERY attribute category found in the JSON (objects, colors, actions, events, positions, lighting, materials, spatial relations, etc.)
- **Visual Accuracy**: Use your visual observation for the actual values - correct JSON values if they don't match what you see
- **Enhanced Details**: Add appropriate visual details you observe that complement the JSON information (lighting conditions, background elements, textures, movements, etc.)
- **Optimal Length**: Create 3-4 natural, fluent English sentences that provide comprehensive coverage
- **Balanced Description**: Ensure JSON attributes are fully covered while enriching with observational details
- **Natural Flow**: Seamlessly integrate all elements into coherent, descriptive sentences
- **ENGLISH ONLY**: Output must be entirely in English, no Chinese characters allowed

**Output format**: Provide ONLY the final video description (3-4 sentences), no analysis process or explanations.

Generate the comprehensive video description:
""".strip()
    
    messages = [{'role': 'user', 'content': frames + [prompt]}]
    
    try:
        response = model.chat(
            msgs=messages,
            tokenizer=tokenizer,
            use_image_id=False,
            max_slice_nums=1,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=512
        )
        return _strip_think(response)
    except Exception as e:
        print(f"[ERROR] Description generation failed: {e}")
        return f"Generation failed: {str(e)}"

def generate_shot_description(model, tokenizer, frames, shot_info, shot_json_data=None):
    """Generate shot description helper."""
    
    shot_id = shot_info['shot_id']
    start_time = shot_info['start_time']
    end_time = shot_info['end_time']
    
    import json
    json_reference = ""
    shot_specific_json = {}
    
    if shot_json_data:
        shot_specific_json = _sanitize_evidence_for_prompt(shot_json_data.copy())
        
        def _event_overlaps_shot(event):
            if not isinstance(event, dict):
                return False
            s_abs = event.get('s_abs', event.get('start_time', event.get('start', 0)))
            e_abs = event.get('e_abs', event.get('end_time', event.get('end', float('inf'))))
            return (
                start_time <= s_abs < end_time or
                start_time <= e_abs <= end_time or
                (s_abs <= start_time and e_abs >= end_time)
            )

        def _filter_event_payload(payload):
            if isinstance(payload, list):
                return [e for e in payload if _event_overlaps_shot(e)]
            if isinstance(payload, dict) and isinstance(payload.get('events'), list):
                filtered_payload = payload.copy()
                filtered_payload['events'] = [e for e in payload['events'] if _event_overlaps_shot(e)]
                return filtered_payload
            return payload

        if 'event_evidence' in shot_specific_json:
            shot_specific_json['event_evidence'] = _filter_event_payload(shot_specific_json['event_evidence'])
        
        json_reference = json.dumps(shot_specific_json, indent=2, ensure_ascii=False)
    
    prompt = f"""
Observe this shot's video frames and generate a natural English description.

Shot {shot_id} ({start_time:.1f}s-{end_time:.1f}s)

## Reference JSON Data for this time segment:
{json_reference if json_reference else "No reference data available"}

## Task:
Generate a comprehensive shot description that covers all relevant attribute categories from the JSON while adding appropriate visual details you observe.

## Requirements:
- **Complete JSON Coverage**: Include information about EVERY attribute category found in the relevant JSON data (objects, colors, actions, events, positions, lighting, etc.)
- **Visual Accuracy**: Use your visual observation for the actual values - correct JSON values if they don't match what you see
- **Enhanced Details**: Add appropriate visual details you observe that complement the JSON information for this time segment
- **Comprehensive Description**: Create one detailed, natural English sentence that covers all elements
- **Value Correction**: Correct JSON values if they don't match what you observe, but keep all attribute categories
- **EXACT FORMAT**: Must start with "shot{shot_id}:" (lowercase "shot")
- **ENGLISH ONLY**: Output must be entirely in English, no Chinese characters allowed

**Output format**: Provide ONLY the final "shot{shot_id}: [comprehensive description]" format, no analysis or explanations.

Generate the shot description:
""".strip()
    
    messages = [{'role': 'user', 'content': frames + [prompt]}]
    
    try:
        response = model.chat(
            msgs=messages,
            tokenizer=tokenizer,
            use_image_id=False,
            max_slice_nums=1,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=384
        )
        return _strip_think(response)
    except Exception as e:
        print(f"[ERROR] Description generation failed for shot {shot_id}: {e}")
        return f"shot{shot_id}: Generation failed - {str(e)}"

def generate_multi_shot_description(model, tokenizer, shots_data, video_json_data=None):
    """Generate multi shot description helper."""
    
    result_lines = []
    for shot in shots_data:
        description = shot['description']
        if not description.lower().startswith(f"shot{shot['shot_id']}:"):
            clean_desc = description.replace(f"Shot {shot['shot_id']}:", "").strip()
            clean_desc = description.replace(f"shot{shot['shot_id']}:", "").strip()
            result_lines.append(f"shot{shot['shot_id']}: {clean_desc}")
        else:
            result_lines.append(description)
    
    return "\n".join(result_lines)

def generate_description_with_retry(generation_func, model, tokenizer, *args, is_multishot=False, max_retries=3):
    """Generate description with retry helper."""
    
    for attempt in range(max_retries):
        try:
            print(f"[INFO] Generating description (attempt {attempt + 1}/{max_retries})")
            
            description = generation_func(model, tokenizer, *args)
            
            is_valid, error_msg = validate_prompt_output(description, is_multishot)
            
            if is_valid:
                print(f"[OK] Description validation passed")
                return description
            else:
                print(f"[WARNING] Description validation failed: {error_msg}")
                if attempt < max_retries - 1:
                    print(f"[INFO] Retrying with attempt {attempt + 2}...")
                else:
                    print(f"[ERROR] Maximum retries reached; returning the last result")
                    return description
                    
        except Exception as e:
            print(f"[ERROR] Error during generation: {e}")
            if attempt < max_retries - 1:
                print(f"[INFO] Retrying with attempt {attempt + 2}...")
            else:
                print(f"[ERROR] Maximum retries reached")
                return f"Generation failed: {str(e)}"
    
    return "Generation failed: exceeded maximum retry count"

# ---------- Prompt generation ----------
def process_single_shot_video(video_path: str, json_path: str, model, tokenizer, output_dir: str):
    """Process single shot video helper."""
    print(f"[INFO] Processing single-shot video: {video_path}")
    print(f"[INFO] Using JSON file: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            video_json_data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON file: {e}")
        video_json_data = {}
    
    video_info = get_video_info(video_path)
    duration = video_info['duration']
    
    frames = extract_frames_from_video(video_path, 0, duration, target_frames=12)
    print(f"[INFO] Extracted {len(frames)} frames")
    
    description = generate_description_with_retry(
        generate_single_shot_description, 
        model, tokenizer, frames, video_json_data,
        is_multishot=False
    )
    
    return description

def process_multi_shot_video(video_path: str, json_path: str, model, tokenizer, output_dir: str, 
                           shot_threshold: float = 35.0, min_shot_length: float = 3.0):
    """Process multi shot video helper."""
    print(f"[INFO] Processing multi-shot video: {video_path}")
    print(f"[INFO] Using JSON file: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            video_json_data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON file: {e}")
        video_json_data = {}
    
    video_info = get_video_info(video_path)
    duration = video_info['duration']
    
    shots = detect_shots_with_scenedetect(video_path, shot_threshold, min_shot_length)
    
    if not shots:
        print(f"[WARNING] No shot changes detected; falling back to single-shot mode")
        return process_single_shot_video(video_path, json_path, model, tokenizer, output_dir)
    
    shots_data = []
    for shot in shots:
        print(f"[INFO] Processing shot {shot['shot_id']}: {shot['start_time']:.1f}s-{shot['end_time']:.1f}s")
        
        frames = extract_frames_from_video(
            video_path, 
            shot['start_time'], 
            shot['end_time'], 
            target_frames=min(8, max(4, int(shot['duration'] * 2)))
        )
        
        description = generate_description_with_retry(
            generate_shot_description,
            model, tokenizer, frames, shot, video_json_data,
            is_multishot=False
        )
        
        shots_data.append({
            'shot_id': shot['shot_id'],
            'start_time': shot['start_time'],
            'end_time': shot['end_time'],
            'duration': shot['duration'],
            'frames': frames,
            'description': description
        })
    
    overall_description = generate_multi_shot_description(model, tokenizer, shots_data, video_json_data)
    
    is_valid, error_msg = validate_prompt_output(overall_description, is_multishot=True)
    if not is_valid:
        print(f"[WARNING] Multi-shot overall description format validation failed: {error_msg}")
        result_lines = []
        for shot in shots_data:
            description = shot['description']
            shot_id = shot['shot_id']
            clean_desc = description.lower()
            if clean_desc.startswith(f"shot{shot_id}:"):
                result_lines.append(description)
            else:
                clean_desc = description.replace(f"Shot {shot_id}:", "").strip()
                clean_desc = description.replace(f"shot{shot_id}:", "").strip()
                if not clean_desc:
                    clean_desc = description.strip()
                result_lines.append(f"shot{shot_id}: {clean_desc}")
        overall_description = "\n".join(result_lines)
        print(f"[INFO] Manually repaired multi-shot format")
    
    return overall_description

# ---------- Video processing ----------
def _infer_sample_id_from_video(video_path: str) -> str:
    return Path(video_path).stem


def _load_prompt_metadata(metadata_jsonl: str) -> dict:
    if not metadata_jsonl:
        return {}
    p = Path(metadata_jsonl)
    if not p.exists():
        return {}
    out = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(obj.get("sample_id", "") or "")
            if sid:
                out[sid] = obj
    return out


def _infer_theme_from_video(video_path: str, sample_id: str, metadata_jsonl: str = "") -> str:
    meta = _load_prompt_metadata(metadata_jsonl).get(sample_id, {})
    theme = str(meta.get("theme", "") or "").strip()
    if theme:
        return theme

    p = Path(video_path).resolve()
    try:
        rel = p.relative_to(VIDEO_ROOT.resolve())
        if len(rel.parts) > 1:
            return rel.parts[0]
    except Exception:
        pass

    if p.parent.resolve() == VIDEO_ROOT.resolve():
        return ""
    return p.parent.name


def _append_prompt_record(jsonl_path: str, record: dict) -> None:
    out = Path(jsonl_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out.exists():
        with open(out, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    old = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if old.get("sample_id") == record.get("sample_id"):
                    continue
                rows.append(old)
    rows.append(record)
    tmp = out.with_name(f".{out.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out)


def main():
    parser = argparse.ArgumentParser(description='Video description prompt generator with JSON assistance')
    parser.add_argument('--video', required=True, help='Input video file path (typically under data/refvideo)')
    parser.add_argument('--json', required=True, help='Input evidence JSON file path (typically under data/metadata/semantic_event_evidence)')
    parser.add_argument('--out', default=str(PROMPT_ROOT), help='Temporary prompt text output directory')
    parser.add_argument('--output-jsonl', default=str(PROMPT_JSONL_PATH), help='Merged prompt JSONL output path')
    parser.add_argument('--video-type', required=True, choices=['single', 'multi'], 
                       help='Video type: single for single-shot videos, multi for multi-shot videos')
    parser.add_argument('--sample-id', default='', help='Explicit sample_id for the appended JSONL record')
    parser.add_argument('--theme', default='', help='Explicit theme for the appended JSONL record')
    parser.add_argument('--metadata-jsonl', default=str(PROMPT_JSONL_PATH), help='Prompt metadata JSONL used to infer theme when --theme is omitted')
    
    parser.add_argument('--model-path', default=str(MODEL_ROOT), help='MiniCPM-V model path (default: checkpoints/minicpm-v-4_5)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'], help='Data type')
    
    parser.add_argument('--shot-threshold', type=float, default=35.0, help='Shot-change detection threshold')
    parser.add_argument('--min-shot-length', type=float, default=3.0, help='Minimum shot length in seconds')
    
    args = parser.parse_args()
    
    if not SCENEDETECT_AVAILABLE and args.video_type == 'multi':
        print("[ERROR] Multi-shot mode requires PySceneDetect. Install it with: pip install scenedetect")
        return
    
    if not CV2_AVAILABLE and not DECORD_AVAILABLE:
        print("[ERROR] cv2 or decord is required for video processing")
        return
    
    if not os.path.exists(args.video):
        print(f"[ERROR] Video file does not exist: {args.video}")
        return
    
    if not os.path.exists(args.json):
        print(f"[ERROR] JSON file does not exist: {args.json}")
        return
    
    os.makedirs(args.out, exist_ok=True)
    
    device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    dtype = _dtype_of(device, args.dtype)
    
    model, tokenizer = load_model(args.model_path, device, dtype)
    
    if args.video_type == 'single':
        description_text = process_single_shot_video(args.video, args.json, model, tokenizer, args.out)
    else:
        description_text = process_multi_shot_video(
            args.video, args.json, model, tokenizer, args.out, 
            args.shot_threshold, args.min_shot_length
        )
    
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    txt_path = os.path.join(args.out, f"{video_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(description_text)
    print(f"[OK] Video prompt saved: {txt_path}")

    sample_id = args.sample_id.strip() or _infer_sample_id_from_video(args.video)
    theme = args.theme.strip() or _infer_theme_from_video(args.video, sample_id, args.metadata_jsonl)
    record = {
        "sample_id": sample_id,
        "prompt": description_text,
        "shot_type": args.video_type,
        "theme": theme,
    }
    _append_prompt_record(args.output_jsonl, record)
    print(f"[OK] Prompt appended to JSONL: {args.output_jsonl}")
    
    print(f"[INFO] Prompt length: {len(description_text)} characters")
    print(f"[INFO] Prompt preview: {description_text[:200]}...")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        main()
