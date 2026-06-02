#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch prompt generator that merges outputs into one JSONL file."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"

VIDEO_BASE_DIR = DATA_DIR / "refvideo"
EVIDENCE_BASE_DIR = METADATA_DIR / "semantic_event_evidence"
PROMPT_JSONL_PATH = METADATA_DIR / "ref4d_prompts.jsonl"
SOURCE_INDEX_PATH = METADATA_DIR / "ref4d_videobench_reference_sources.csv"
PROGRESS_FILE = METADATA_DIR / "prompt_progress.json"
TMP_OUTPUT_DIR = METADATA_DIR / "_prompt_tmp"
MODEL_ROOT = PROJECT_ROOT / "checkpoints" / "minicpm-v-4_5"

SINGLE_GENERATOR = SCRIPT_DIR / "video_prompt_generator_scenedetect.py"

for _dir in (VIDEO_BASE_DIR, EVIDENCE_BASE_DIR, METADATA_DIR, TMP_OUTPUT_DIR, MODEL_ROOT):
    _dir.mkdir(parents=True, exist_ok=True)


@dataclass
class VideoItem:
    video_path: str
    theme: str
    filename: str
    video_type: str
    json_path: Optional[str] = None
    json_found: bool = False
    sample_id: Optional[str] = None


def infer_sample_id(filename: str) -> str:
    """Infer sample_id from the current Ref4D video filename."""
    return Path(filename).stem


def _normalize_video_type(value: Optional[str]) -> Optional[str]:
    val = str(value or "").strip().lower()
    return val if val in {"single", "multi"} else None


def _merge_sample_meta(
    sample_meta: Dict[str, Dict[str, str]],
    sample_id: str,
    *,
    shot_type: Optional[str] = None,
    theme: Optional[str] = None,
) -> None:
    if not sample_id:
        return
    rec = sample_meta.setdefault(sample_id, {})
    norm_type = _normalize_video_type(shot_type)
    if norm_type:
        rec["shot_type"] = norm_type
    if theme:
        rec["theme"] = str(theme)


def load_sample_metadata(metadata_jsonl: str, source_index: str) -> Dict[str, Dict[str, str]]:
    """Load sample_id -> {shot_type, theme} from released metadata assets."""
    sample_meta: Dict[str, Dict[str, str]] = {}

    prompt_path = Path(metadata_jsonl) if metadata_jsonl else None
    if prompt_path and prompt_path.is_file():
        for line in prompt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            _merge_sample_meta(
                sample_meta,
                str(obj.get("sample_id", "") or ""),
                shot_type=obj.get("shot_type"),
                theme=obj.get("theme"),
            )

    source_path = Path(source_index) if source_index else None
    if source_path and source_path.is_file():
        with source_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                _merge_sample_meta(
                    sample_meta,
                    str(row.get("sample_id", "") or ""),
                    shot_type=row.get("shot_type"),
                    theme=row.get("theme"),
                )

    return sample_meta


def _video_type_for_sample(sample_id: str, sample_meta: Dict[str, Dict[str, str]], default_video_type: str) -> str:
    return sample_meta.get(sample_id, {}).get("shot_type") or default_video_type


def _theme_for_sample(sample_id: str, sample_meta: Dict[str, Dict[str, str]], fallback: str) -> str:
    return sample_meta.get(sample_id, {}).get("theme") or fallback


def find_video_files(
    base_dir: str,
    sample_meta: Dict[str, Dict[str, str]],
    default_video_type: str = "single",
) -> List[VideoItem]:
    """Scan videos and infer shot type from released metadata when available."""
    items: List[VideoItem] = []
    root = Path(base_dir)

    for p in root.glob("*.mp4"):
        sample_id = infer_sample_id(p.name)
        items.append(
            VideoItem(
                video_path=str(p),
                theme=_theme_for_sample(sample_id, sample_meta, root.name),
                filename=p.name,
                video_type=_video_type_for_sample(sample_id, sample_meta, default_video_type),
                sample_id=sample_id,
            )
        )

    for theme_dir in root.iterdir():
        if not theme_dir.is_dir():
            continue

        mp4_files = list(theme_dir.glob("**/*.mp4"))
        if not mp4_files:
            mp4_files = list(theme_dir.glob("*.mp4"))

        for p in mp4_files:
            name = p.name
            sample_id = infer_sample_id(name)

            items.append(
                VideoItem(
                    video_path=str(p),
                    theme=_theme_for_sample(sample_id, sample_meta, theme_dir.name),
                    filename=name,
                    video_type=_video_type_for_sample(sample_id, sample_meta, default_video_type),
                    sample_id=sample_id,
                )
            )

    return items


def build_evidence_index(json_base_dir: str) -> Dict[str, str]:
    """Build filename -> absolute path index for supported evidence JSON files."""
    index: Dict[str, str] = {}
    for pattern in ("**/*_semantic_event.json", "**/*.json"):
        for p in Path(json_base_dir).glob(pattern):
            index.setdefault(p.name, str(p))
    return index


def find_corresponding_json(item: VideoItem, json_base_dir: str, index: Dict[str, str]) -> bool:
    """Resolve evidence file for a video."""
    stem = item.sample_id or Path(item.filename).stem
    candidates = [
        f"{stem}_semantic_event.json",
    ]

    for json_name in candidates:
        c1 = Path(json_base_dir) / item.theme / json_name
        c2 = Path(json_base_dir) / json_name

        if c1.exists():
            item.json_path = str(c1)
            item.json_found = True
            return True
        if c2.exists():
            item.json_path = str(c2)
            item.json_found = True
            return True
        if json_name in index:
            item.json_path = index[json_name]
            item.json_found = True
            return True

    item.json_found = False
    return False


class ProgressTracker:
    """Simple progress tracker for resume mode."""

    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.data = self._load()

    def _new(self) -> dict:
        return {
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "processed_files": {},
            "statistics": {"success": 0, "failed": 0, "skipped": 0},
        }

    def _load(self) -> dict:
        p = Path(self.progress_file)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARNING] Failed to load progress file: {e}")
        return self._new()

    def _save(self) -> None:
        Path(self.progress_file).write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def is_processed(self, video_path: str) -> bool:
        return video_path in self.data["processed_files"]

    def mark(self, video_path: str, status: str, payload: dict) -> None:
        self.data["processed_files"][video_path] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            **payload,
        }
        self.data["statistics"][status] += 1
        self.data["last_update"] = datetime.now().isoformat()
        self._save()

    def stats(self) -> dict:
        return self.data["statistics"]


def load_existing_sample_ids(output_jsonl: str) -> Set[str]:
    """Load existing sample_id values from output JSONL to avoid duplicates."""
    ids: Set[str] = set()
    p = Path(output_jsonl)
    if not p.exists():
        return ids

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            sid = obj.get("sample_id")
            if sid:
                ids.add(str(sid))
        except json.JSONDecodeError:
            continue
    return ids


def append_prompt_jsonl(output_jsonl: str, record: dict) -> None:
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_video_processing(
    item: VideoItem,
    model_path: str,
    tmp_output_dir: str,
    device: str,
    dtype: str,
    shot_threshold: float,
    min_shot_length: float,
) -> tuple[bool, float, Optional[str], Optional[str]]:
    """Run single-video generator and return (ok, sec, err, prompt_text)."""
    tmp_dir = Path(tmp_output_dir) / item.theme
    tmp_dir.mkdir(parents=True, exist_ok=True)
    passthrough_jsonl = tmp_dir / "_single_script_passthrough.jsonl"

    cmd = [
        sys.executable,
        str(SINGLE_GENERATOR),
        "--video",
        item.video_path,
        "--json",
        str(item.json_path),
        "--out",
        str(tmp_dir),
        "--video-type",
        item.video_type,
        "--sample-id",
        str(item.sample_id),
        "--theme",
        item.theme,
        "--model-path",
        model_path,
        "--output-jsonl",
        str(passthrough_jsonl),
        "--device",
        device,
        "--dtype",
        dtype,
    ]
    if item.video_type == "multi":
        cmd.extend(["--shot-threshold", str(shot_threshold), "--min-shot-length", str(min_shot_length)])

    print(f"[PROCESSING] {item.theme}/{item.filename}")
    print(f"[COMMAND] {' '.join(cmd)}")

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return False, 0.0, "timeout (>900s)", None
    except Exception as e:
        return False, 0.0, str(e), None

    elapsed = time.time() - start
    if result.returncode != 0:
        err = f"returncode={result.returncode}"
        print(f"[ERROR] {item.filename} failed: {err}")
        print(result.stdout)
        print(result.stderr)
        return False, elapsed, err, None

    txt_path = tmp_dir / f"{Path(item.filename).stem}.txt"
    if not txt_path.exists():
        return False, elapsed, f"missing output txt: {txt_path}", None

    prompt_text = txt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        return False, elapsed, "empty prompt", None
    return True, elapsed, None, prompt_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch prompt generator merged to one JSONL file")
    parser.add_argument("--video-base-dir", default=str(VIDEO_BASE_DIR), help="Video input root")
    parser.add_argument("--json-base-dir", default=str(EVIDENCE_BASE_DIR), help="Evidence JSON root")
    parser.add_argument("--output-jsonl", default=str(PROMPT_JSONL_PATH), help="Output merged JSONL file")
    parser.add_argument("--model-path", default=str(MODEL_ROOT), help="MiniCPM-V model path")
    parser.add_argument("--progress-file", default=str(PROGRESS_FILE), help="Progress file for resume mode")
    parser.add_argument("--tmp-output-dir", default=str(TMP_OUTPUT_DIR), help="Temporary txt output directory")
    parser.add_argument(
        "--metadata-jsonl",
        default=str(PROMPT_JSONL_PATH),
        help="Prompt metadata JSONL used to resolve sample_id -> shot_type/theme",
    )
    parser.add_argument(
        "--source-index",
        default=str(SOURCE_INDEX_PATH),
        help="Reference source CSV used to resolve sample_id -> shot_type/theme",
    )
    parser.add_argument("--theme", nargs="*", help="Process selected themes only")
    parser.add_argument("--resume", action="store_true", help="Resume from progress file")
    parser.add_argument("--dry-run", action="store_true", help="List files and exit")
    parser.add_argument("--skip-confirmation", action="store_true", help="Skip interactive confirmation")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Model runtime device")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Model dtype")
    parser.add_argument(
        "--default-video-type",
        default="single",
        choices=["single", "multi"],
        help="Video type used only when metadata does not provide shot_type",
    )
    parser.add_argument("--shot-threshold", type=float, default=35.0, help="PySceneDetect threshold")
    parser.add_argument("--min-shot-length", type=float, default=3.0, help="Minimum shot length (seconds)")
    args = parser.parse_args()

    print("=" * 80)
    print("Batch Prompt Builder (Merged JSONL)")
    print("=" * 80)
    print(f"[INFO] Video root:    {args.video_base_dir}")
    print(f"[INFO] Evidence root: {args.json_base_dir}")
    print(f"[INFO] Output JSONL:  {args.output_jsonl}")
    print(f"[INFO] Model path:    {args.model_path}")
    print(f"[INFO] Metadata JSONL:{args.metadata_jsonl}")
    print(f"[INFO] Source index:  {args.source_index}")
    print(f"[INFO] Progress file: {args.progress_file}")

    if not Path(args.video_base_dir).exists():
        print(f"[ERROR] Video directory does not exist: {args.video_base_dir}")
        return
    if not Path(args.json_base_dir).exists():
        print(f"[ERROR] Evidence directory does not exist: {args.json_base_dir}")
        return

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tmp_output_dir).mkdir(parents=True, exist_ok=True)
    tracker = ProgressTracker(args.progress_file)
    existing_ids = load_existing_sample_ids(args.output_jsonl)
    sample_meta = load_sample_metadata(args.metadata_jsonl, args.source_index)
    print(f"[INFO] Metadata samples: {len(sample_meta)}")

    videos = find_video_files(
        args.video_base_dir,
        sample_meta=sample_meta,
        default_video_type=args.default_video_type,
    )
    if not videos:
        print("[ERROR] No video files found.")
        return

    if args.theme:
        wanted = set(args.theme)
        videos = [v for v in videos if v.theme in wanted]

    evidence_index = build_evidence_index(args.json_base_dir)
    for v in videos:
        find_corresponding_json(v, args.json_base_dir, evidence_index)

    missing_json = [v for v in videos if not v.json_found]
    processable = [v for v in videos if v.json_found]

    print(f"[INFO] Total videos:        {len(videos)}")
    print(f"[INFO] JSON matched:        {len(processable)}")
    print(f"[INFO] JSON missing:        {len(missing_json)}")
    print(f"[INFO] Existing sample_ids: {len(existing_ids)}")
    type_counts = {"single": 0, "multi": 0}
    for v in processable:
        if v.video_type in type_counts:
            type_counts[v.video_type] += 1
    print(f"[INFO] Shot types:          single={type_counts['single']}, multi={type_counts['multi']}")

    if missing_json:
        print("[WARNING] Missing evidence for the following files:")
        for v in missing_json[:20]:
            print(f"  - {v.theme}/{v.filename}")
        if len(missing_json) > 20:
            print(f"  ... and {len(missing_json) - 20} more")

    if args.resume:
        before = len(processable)
        processable = [v for v in processable if not tracker.is_processed(v.video_path)]
        print(f"[INFO] Resume filter removed {before - len(processable)} already-processed files.")

    before = len(processable)
    processable = [v for v in processable if (v.sample_id not in existing_ids)]
    print(f"[INFO] Duplicate sample_id filter removed {before - len(processable)} files.")

    if not processable:
        print("[INFO] Nothing to process.")
        return

    print(f"[INFO] Ready to process {len(processable)} files.")
    if args.dry_run:
        for v in processable[:50]:
            print(f"  - {v.sample_id} | {v.video_type} | {v.theme} | {v.filename}")
        if len(processable) > 50:
            print(f"  ... and {len(processable) - 50} more")
        return

    if not args.skip_confirmation:
        resp = input("Continue processing? (y/N): ").strip().lower()
        if resp != "y":
            print("[INFO] Cancelled.")
            return

    success = 0
    failed = 0
    skipped = 0

    for i, v in enumerate(processable, 1):
        print(f"\n[{i}/{len(processable)}] {v.theme}/{v.filename} ({v.sample_id})")
        ok, elapsed, err, prompt = run_video_processing(
            item=v,
            model_path=args.model_path,
            tmp_output_dir=args.tmp_output_dir,
            device=args.device,
            dtype=args.dtype,
            shot_threshold=args.shot_threshold,
            min_shot_length=args.min_shot_length,
        )

        if not ok:
            failed += 1
            tracker.mark(
                v.video_path,
                "failed",
                {"theme": v.theme, "filename": v.filename, "video_type": v.video_type, "error": err},
            )
            print(f"[ERROR] Failed in {elapsed:.1f}s: {err}")
            continue

        if not prompt:
            skipped += 1
            tracker.mark(
                v.video_path,
                "skipped",
                {"theme": v.theme, "filename": v.filename, "video_type": v.video_type, "reason": "empty prompt"},
            )
            print("[WARNING] Empty prompt skipped.")
            continue

        record = {
            "sample_id": v.sample_id,
            "prompt": prompt,
            "shot_type": v.video_type,
            "theme": v.theme,
        }
        append_prompt_jsonl(args.output_jsonl, record)
        existing_ids.add(v.sample_id or "")

        success += 1
        tracker.mark(
            v.video_path,
            "success",
            {
                "theme": v.theme,
                "filename": v.filename,
                "video_type": v.video_type,
                "sample_id": v.sample_id,
                "processing_time": elapsed,
            },
        )
        print(f"[OK] Appended to JSONL in {elapsed:.1f}s")

    print("\n" + "=" * 80)
    print("Batch finished")
    print("=" * 80)
    print(f"Success: {success}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    print(f"Output JSONL:  {args.output_jsonl}")
    print(f"Progress file: {args.progress_file}")
    print(f"Temp output:   {args.tmp_output_dir}")


if __name__ == "__main__":
    main()
