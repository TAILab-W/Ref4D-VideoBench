#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build World Knowledge Question Bank (Rules -> Assertions -> VQA -> Bank)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build World Knowledge QA Bank from ref4d_meta.jsonl")
    p.add_argument("--repo-root", default="", help="Project root directory. Auto-detected by default.")
    p.add_argument("--evidence-dir", default="data/metadata/semantic_event_evidence", help="Input: fused evidence directory")
    p.add_argument("--video-dir", default="data/refvideo", help="Reference video directory")
    p.add_argument("--out-dir", default="data/metadata/world_qa", help="Output: final question bank directory")
    p.add_argument("--cache-dir", default="outputs/world/cache/build", help="Intermediate cache directory")
    p.add_argument("--model-path", required=True, help="Local absolute path to MiniCPM-V-4_5 (required)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return p

def _infer_repo_root(cli_repo_root: str) -> Path:
    if cli_repo_root:
        return Path(cli_repo_root).expanduser().resolve()
    
    candidates = [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents)
    for cand in candidates:
        if (cand / "data").is_dir() and (cand / "ref4d_build").is_dir():
            return cand
    raise FileNotFoundError("Cannot find project root directory. Make sure you are under Ref4D-VideoBench.")

def _resolve_repo_path(repo_root: Path, path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p.resolve() if p.is_absolute() else (repo_root / p).resolve()

def run_step(cmd: List[str], step_name: str, cwd: Path):
    print(f"\n========== [World QA Build] Starting {step_name} ==========")
    proc = subprocess.Popen(cmd, cwd=str(cwd))
    ret = proc.wait()
    if ret != 0:
        print(f"[ERROR] {step_name} failed with exit code: {ret}")
        sys.exit(ret)

def main():
    args = build_arg_parser().parse_args()
    repo_root = _infer_repo_root(args.repo_root)
    
    evidence_dir = _resolve_repo_path(repo_root, args.evidence_dir)
    video_dir = _resolve_repo_path(repo_root, args.video_dir)
    out_dir = _resolve_repo_path(repo_root, args.out_dir)
    cache_dir = _resolve_repo_path(repo_root, args.cache_dir)
    
    rules_dir = cache_dir / "rules"
    assert_dir = cache_dir / "assertions"
    vqa_dir = cache_dir / "vqa"
    for d in [out_dir, rules_dir, assert_dir, vqa_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[Meta] repo_root={repo_root}")
    print(f"[Meta] model_path={args.model_path}")
    
    script_dir = repo_root / "ref4d_build" / "world_ref"

    base_kwargs = [
        "--local-path", args.model_path,
        "--device", args.device,
        "--dtype", args.dtype
    ]

    run_step([
        sys.executable, str(script_dir / "rule_many.py"),
        "--json-dir", str(evidence_dir),
        "--video-dir", str(video_dir),
        "--out-dir", str(rules_dir)
    ] + base_kwargs, "Rule Generation", repo_root)

    run_step([
        sys.executable, str(script_dir / "AS_0.py"),
        "--json-dir", str(rules_dir),
        "--video-dir", str(video_dir),
        "--out-dir", str(assert_dir)
    ] + base_kwargs, "Assertion Generation", repo_root)

    run_step([
        sys.executable, str(script_dir / "VQA_0.py"),
        "--json-dir", str(rules_dir),
        "--video-dir", str(video_dir),
        "--assert-dir", str(assert_dir),
        "--out-dir", str(vqa_dir)
    ] + base_kwargs, "VQA Generation", repo_root)

    run_step([
        sys.executable, str(script_dir / "question_bank.py"),
        "--video-dir", str(video_dir),
        "--vqa-dir", str(vqa_dir),
        "--assert-dir", str(assert_dir),
        "--out-dir", str(out_dir)
    ] + base_kwargs, "Bank Building", repo_root)

    print(f"\n[Success] World knowledge question bank built! Results saved to: {out_dir}")

if __name__ == "__main__":
    main()