
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from tqdm.auto import tqdm
except Exception:  
    tqdm = None

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".flv", ".ts", ".mpg", ".mpeg", ".wmv"}
_OK_RE = "[OK] saved -> "
_ERR_RE = "[ERROR] "

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build reference semantic evidence from ref4d_meta.jsonl")

    p.add_argument("--repo-root", default="", help="Repository root. If omitted, infer automatically.")
    p.add_argument("--meta-path", default="data/metadata/ref4d_meta.jsonl", help="Path to ref4d_meta.jsonl (repo-relative if not absolute).")
    p.add_argument("--out-dir", default="data/metadata/semantic_evidence", help="Output directory for reference semantic evidence (repo-relative if not absolute).")
    p.add_argument(
        "--extractor-path",
        default="ref4d_eval/semantic/evidence_extract/evi_extract.py",
        help="Path to the semantic evidence extractor script (repo-relative if not absolute).",
    )
    p.add_argument("--force", action="store_true", help="Rebuild even if a valid evidence file already exists.")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N samples from meta; 0 means no limit.")

    p.add_argument("--debug-dump", default="", help="Optional debug dump directory passed to evi_extract.py")
    p.add_argument("--local-path", default="", help="Local MiniCPM-V model directory passed to evi_extract.py")
    p.add_argument("--model-id", default="openbmb/MiniCPM-V-4_5", help="Model id passed to evi_extract.py when --local-path is empty")
    p.add_argument("--revision", default="main", help="Model revision passed to evi_extract.py")
    p.add_argument("--cache-dir", default="", help="Cache directory passed to evi_extract.py")
    p.add_argument("--local-files-only", action="store_true", help="Pass --local-files-only to evi_extract.py")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device passed to evi_extract.py")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Dtype passed to evi_extract.py")
    p.add_argument("--disable-flash-sdp", action="store_true", help="Pass --disable-flash-sdp to evi_extract.py")
    p.add_argument("--force-math-sdp", action="store_true", help="Pass --force-math-sdp to evi_extract.py")
    p.add_argument("--fps", type=int, default=6, help="Sampling fps passed to evi_extract.py")
    p.add_argument("--cap-frames", type=int, default=240, help="cap_frames passed to evi_extract.py")
    p.add_argument("--resize-short", type=int, default=448, help="resize_short passed to evi_extract.py")
    p.add_argument("--max-packing", type=int, default=3, help="max_packing passed to evi_extract.py")
    p.add_argument("--decode-backend", default="auto", choices=["auto", "cv2", "decord"], help="Decode backend passed to evi_extract.py")
    p.add_argument("--max-new-tokens", type=int, default=512, help="max_new_tokens passed to evi_extract.py")
    p.add_argument("--min-max-new-tokens", type=int, default=96, help="min_max_new_tokens passed to evi_extract.py")
    p.add_argument("--temperature", type=float, default=0.0, help="Temperature passed to evi_extract.py")
    p.add_argument("--enable-thinking", action="store_true", help="Pass --enable-thinking to evi_extract.py")
    p.add_argument("--verbose", action="store_true", help="Pass --verbose to evi_extract.py")
    p.add_argument("--min-span-sec", type=float, default=0.1, help="min_span_sec passed to evi_extract.py")
    p.add_argument("--min-fps", type=int, default=2, help="min_fps passed to evi_extract.py")

    return p

def _infer_repo_root(cli_repo_root: str) -> Path:
    if cli_repo_root:
        return Path(cli_repo_root).expanduser().resolve()

    candidates: List[Path] = []
    this_file = Path(__file__).resolve()
    candidates.extend([this_file.parent, *this_file.parents])
    candidates.extend([Path.cwd(), *Path.cwd().parents])

    seen = set()
    for cand in candidates:
        cand_resolved = cand.resolve()
        if cand_resolved in seen:
            continue
        seen.add(cand_resolved)
        if (cand_resolved / "pyproject.toml").exists() and (cand_resolved / "ref4d_eval").is_dir() and (cand_resolved / "data").is_dir():
            return cand_resolved

    raise FileNotFoundError("Unable to infer repository root. Please pass --repo-root explicitly.")

def _resolve_repo_path(repo_root: Path, path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p.resolve() if p.is_absolute() else (repo_root / p).resolve()

def _has_nonempty_evidence(path: Path, min_bytes: int = 64) -> bool:
    try:
        if (not path.is_file()) or path.stat().st_size < min_bytes:
            return False
        with path.open("r", encoding="utf-8") as f:
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

def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

def _load_meta(meta_path: Path, limit: int) -> List[Tuple[str, str, int]]:
    records: List[Tuple[str, str, int]] = []
    seen_sample_ids = set()

    with meta_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{meta_path}:{lineno}: each line must be a JSON object")

            sample_id = obj.get("sample_id")
            ref_video = obj.get("ref_video")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"{meta_path}:{lineno}: missing or invalid 'sample_id'")
            if not isinstance(ref_video, str) or not ref_video.strip():
                raise ValueError(f"{meta_path}:{lineno}: missing or invalid 'ref_video'")
            if sample_id in seen_sample_ids:
                raise ValueError(f"{meta_path}:{lineno}: duplicate sample_id '{sample_id}'")

            seen_sample_ids.add(sample_id)
            records.append((sample_id.strip(), ref_video.strip(), lineno))

    if limit > 0:
        return records[:limit]
    return records

def _prepare_tasks(
    records: Sequence[Tuple[str, str, int]],
    repo_root: Path,
    out_dir: Path,
    force: bool,
) -> Tuple[List[Dict[str, str]], int, int, List[str]]:
    tasks: List[Dict[str, str]] = []
    skipped_valid = 0
    removed_invalid = 0
    precheck_failures: List[str] = []

    for sample_id, ref_video_rel, lineno in records:
        video_path = _resolve_repo_path(repo_root, ref_video_rel)
        out_path = (out_dir / f"{sample_id}.json").resolve()

        if video_path.suffix.lower() not in VIDEO_EXTS:
            precheck_failures.append(f"{sample_id} (meta line {lineno}): unsupported ref_video extension -> {video_path}")
            continue
        if not video_path.is_file():
            precheck_failures.append(f"{sample_id} (meta line {lineno}): ref_video not found -> {video_path}")
            continue
        if video_path.stem != sample_id:
            precheck_failures.append(
                f"{sample_id} (meta line {lineno}): basename(ref_video) must equal sample_id, got '{video_path.stem}'"
            )
            continue

        if force:
            if out_path.exists():
                _safe_unlink(out_path)
        else:
            if _has_nonempty_evidence(out_path):
                skipped_valid += 1
                continue
            if out_path.exists():
                _safe_unlink(out_path)
                removed_invalid += 1

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tasks.append({
            "sample_id": sample_id,
            "video": str(video_path),
            "out": str(out_path),
        })

    return tasks, skipped_valid, removed_invalid, precheck_failures

def _write_tasks_jsonl(tasks: Sequence[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps({"video": t["video"], "out": t["out"]}, ensure_ascii=False) + "\n")

def _build_extractor_cmd(args: argparse.Namespace, extractor_path: Path, tasks_path: Path, out_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        str(extractor_path),
        "--batch-from", str(tasks_path),
        "--out-dir", str(out_dir),
        "--device", args.device,
        "--dtype", args.dtype,
        "--fps", str(args.fps),
        "--cap-frames", str(args.cap_frames),
        "--resize-short", str(args.resize_short),
        "--max-packing", str(args.max_packing),
        "--decode-backend", args.decode_backend,
        "--max-new-tokens", str(args.max_new_tokens),
        "--min-max-new-tokens", str(args.min_max_new_tokens),
        "--temperature", str(args.temperature),
        "--min-span-sec", str(args.min_span_sec),
        "--min-fps", str(args.min_fps),
        "--revision", args.revision,
    ]

    if args.debug_dump:
        cmd += ["--debug-dump", args.debug_dump]
    if args.local_path:
        cmd += ["--local-path", args.local_path]
    else:
        cmd += ["--model-id", args.model_id]
    if args.cache_dir:
        cmd += ["--cache-dir", args.cache_dir]
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.disable_flash_sdp:
        cmd.append("--disable-flash-sdp")
    if args.force_math_sdp:
        cmd.append("--force-math-sdp")
    if args.enable_thinking:
        cmd.append("--enable-thinking")
    if args.verbose:
        cmd.append("--verbose")

    return cmd

def _run_extractor_with_progress(cmd: Sequence[str], cwd: Path, total: int) -> int:
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    pbar = tqdm(total=total, desc="semantic_ref", dynamic_ncols=True) if (tqdm is not None and total > 0) else None
    completed = 0

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")

        if _OK_RE in line or _ERR_RE in line:
            completed += 1
            if pbar is not None:
                pbar.update(1)
            else:
                print(f"[progress] {completed}/{total}")

        if _ERR_RE in line or ("[WARN]" in line) or ("[OOM]" in line):
            print(line)

    ret = proc.wait()
    if pbar is not None:
        if completed < total:
            pbar.update(total - completed)
        pbar.close()
    return ret

def main() -> None:
    args = build_arg_parser().parse_args()

    repo_root = _infer_repo_root(args.repo_root)
    meta_path = _resolve_repo_path(repo_root, args.meta_path)
    out_dir = _resolve_repo_path(repo_root, args.out_dir)
    extractor_path = _resolve_repo_path(repo_root, args.extractor_path)

    if not meta_path.is_file():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    if not extractor_path.is_file():
        raise FileNotFoundError(f"Extractor script not found: {extractor_path}")

    records = _load_meta(meta_path, args.limit)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, skipped_valid, removed_invalid, precheck_failures = _prepare_tasks(
        records=records,
        repo_root=repo_root,
        out_dir=out_dir,
        force=args.force,
    )

    print(f"[Meta] repo_root={repo_root}")
    print(f"[Meta] meta_path={meta_path}")
    print(f"[Meta] out_dir={out_dir}")
    print(f"[Meta] total_records={len(records)}")
    if skipped_valid > 0:
        print(f"[Meta] skipped_valid_existing={skipped_valid}")
    if removed_invalid > 0:
        print(f"[Meta] removed_invalid_existing={removed_invalid}")
    if precheck_failures:
        print(f"[Meta] precheck_failures={len(precheck_failures)}")
        for msg in precheck_failures:
            print(f"[FAIL/PRECHECK] {msg}")

    built_valid = 0
    built_failed = 0
    extractor_return_code = 0

    if tasks:
        with tempfile.TemporaryDirectory(prefix="build_semantic_ref_") as tmpdir:
            tasks_path = Path(tmpdir) / "tasks.jsonl"
            _write_tasks_jsonl(tasks, tasks_path)
            cmd = _build_extractor_cmd(args, extractor_path, tasks_path, out_dir)
            extractor_return_code = _run_extractor_with_progress(cmd, cwd=repo_root, total=len(tasks))

        failed_samples: List[str] = []
        for task in tasks:
            out_path = Path(task["out"])
            if _has_nonempty_evidence(out_path):
                built_valid += 1
            else:
                built_failed += 1
                failed_samples.append(task["sample_id"])

        print(f"[Result] built_valid={built_valid}")
        print(f"[Result] built_failed={built_failed}")
        if failed_samples:
            print("[Result] failed_samples=" + ", ".join(failed_samples))
    else:
        print("[Result] no build tasks to run")

    total_failed = len(precheck_failures) + built_failed
    total_success = built_valid
    total_skipped = skipped_valid

    print("\n===== semantic_ref summary =====")
    print(f"success={total_success}")
    print(f"failed={total_failed}")
    print(f"skipped_valid={total_skipped}")

    if total_failed > 0 or extractor_return_code != 0:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
