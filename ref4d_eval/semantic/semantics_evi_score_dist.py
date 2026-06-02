
import argparse
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Tuple, Optional

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".flv", ".ts", ".mpg", ".mpeg", ".wmv"}

def is_video(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS

def list_videos(root: Path) -> List[Path]:
    if not root.exists():
        return []
    if root.is_file() and is_video(root):
        return [root]
    vids: List[Path] = []
    for p in root.rglob("*"):
        if is_video(p):
            vids.append(p)
    return sorted(vids)

def out_path_ref(in_file: Path, ref_out_root: Path) -> Path:
    return (ref_out_root / in_file.with_suffix(".json").name).resolve()

def out_path_gen(in_file: Path, model_in_root: Path, model_out_root: Path) -> Path:
    rel = in_file.relative_to(model_in_root).with_suffix(".json")
    return (model_out_root / rel).resolve()

def has_nonempty_evidence(p: Path, min_bytes: int = 64) -> bool:
    try:
        if (not p.is_file()) or p.stat().st_size < min_bytes:
            return False
        with p.open("r", encoding="utf-8") as f:
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

def clean_if_empty(json_path: Path) -> None:
    try:
        if json_path.is_file() and not has_nonempty_evidence(json_path):
            json_path.unlink(missing_ok=True)
    except Exception:
        pass

def detect_gpus(gpus_arg: str) -> List[str]:
    if gpus_arg and gpus_arg.lower() != "auto":
        return [x.strip() for x in gpus_arg.split(",") if x.strip()]

    try:
        import torch
        n = torch.cuda.device_count()
        if n and n > 0:
            return [str(i) for i in range(n)]
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["bash", "-lc", "nvidia-smi --query-gpu=index --format=csv,noheader"],
            text=True,
        )
        ids = [line.strip() for line in out.splitlines() if line.strip()]
        if ids:
            return ids
    except Exception:
        pass

    return []

def chunk_even(items: List, n: int) -> List[List]:
    n = max(1, n)
    buckets = [[] for _ in range(n)]
    for i, it in enumerate(items):
        buckets[i % n].append(it)
    return buckets

def write_taskfile_for_evi(tasks: List[Tuple[Path, Path]], tmp_dir: Path) -> Path:
    lf = tmp_dir / "tasks.jsonl"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for vin, vout in tasks:
        vout.parent.mkdir(parents=True, exist_ok=True)
        lines.append(json.dumps({"video": str(vin), "out": str(vout)}, ensure_ascii=False))
    lf.write_text("\n".join(lines), encoding="utf-8")
    return lf

def verify_bindings(tasks: List[Tuple[Path, Path]], quiet: bool = False) -> Tuple[int, int]:
    ok, bad = 0, 0
    for vin, vout in tasks:
        if not vout.is_file():
            bad += 1
            if not quiet:
                print(f"[VERIFY] MISSING: {vout}  <- {vin.name}")
            continue
        try:
            obj = json.loads(vout.read_text(encoding="utf-8"))
        except Exception:
            obj = {}
        vbase = vin.name
        meta = obj.get("meta") if isinstance(obj, dict) else None
        matched = False
        if isinstance(meta, dict):
            vb = meta.get("video_basename") or meta.get("video_name")
            vp = meta.get("video") or meta.get("video_path")
            if isinstance(vb, str) and vb == vbase:
                matched = True
            elif isinstance(vp, str) and Path(vp).name == vbase:
                matched = True
        if not matched:
            matched = (vout.name == Path(vbase).with_suffix(".json").name)
        if matched:
            ok += 1
        else:
            bad += 1
            if not quiet:
                print(f"[VERIFY] NAME MISMATCH: {vout.name}  (video {vbase})")
    return ok, bad

def build_ref_tasks(ref_video_dir: Path, ref_out_dir: Path, force: bool, limit: int = 0) -> List[Tuple[Path, Path]]:
    vids = list_videos(ref_video_dir)
    tasks: List[Tuple[Path, Path]] = []
    for v in vids:
        o = out_path_ref(v, ref_out_dir)
        if not force:
            clean_if_empty(o)
        if (not force) and has_nonempty_evidence(o):
            continue
        tasks.append((v, o))
    if limit > 0:
        tasks = tasks[:limit]
    return tasks

def build_gen_tasks_for_model(model_video_dir: Path, model_out_dir: Path, force: bool, limit: int = 0) -> List[Tuple[Path, Path]]:
    vids = list_videos(model_video_dir)
    tasks: List[Tuple[Path, Path]] = []
    for v in vids:
        o = out_path_gen(v, model_video_dir, model_out_dir)
        if not force:
            clean_if_empty(o)
        if (not force) and has_nonempty_evidence(o):
            continue
        tasks.append((v, o))
    if limit > 0:
        tasks = tasks[:limit]
    return tasks

_OK_RE = re.compile(r"\[OK\]\s*saved\s*->\s*(.+?\.json)\s*$")

def _spawn_worker(
    tag: str,
    evi_py: Path,
    model_local_path: Path,
    base_out_dir: Path,
    listfile: Path,
    gpu_id: Optional[str],
    extra_opts: List[str],
    live: bool,
    log_file: Optional[Path],
):
    cmd = [
        sys.executable,
        "-u",
        str(evi_py),
        "--batch-from",
        str(listfile),
        "--out-dir",
        str(base_out_dir),
        "--local-path",
        str(model_local_path),
        "--device",
        "cuda" if gpu_id is not None else "cpu",
        "--dtype",
        "bf16",
        "--decode-backend",
        "auto",
        "--fps",
        "6",
        "--cap-frames",
        "240",
        "--resize-short",
        "448",
        "--max-new-tokens",
        "512",
        "--temperature",
        "0.0",
        "--min-span-sec",
        "0.1",
    ]
    if extra_opts:
        cmd += extra_opts

    env = dict(os.environ)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    if live:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        return proc, None

    assert log_file is not None
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lf = log_file.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return proc, lf

def _reader_thread(proc: subprocess.Popen, gpu_idx: int, qout: "queue.Queue[Tuple[int, Optional[str]]]"):
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            qout.put((gpu_idx, line.rstrip("\n")))
    except Exception as e:
        qout.put((gpu_idx, f"[READER-ERR] {e}"))
    finally:
        qout.put((gpu_idx, None))

def _run_buckets(
    tag: str,
    evi_py: Path,
    model_local_path: Path,
    base_out_dir: Path,
    tasks_buckets: List[List[Tuple[Path, Path]]],
    gpus: List[str],
    extra_opts: List[str],
    live: bool,
    serialize_output: bool,
    quiet: bool,
):
    tmp_root = Path(tempfile.mkdtemp(prefix=f"semantics_{tag.replace(':', '_')}_"))
    log_root = tmp_root / "logs"
    procs: List[Optional[subprocess.Popen]] = []
    resources: List[Tuple[Optional[object], Optional[Path], int]] = []

    if live:
        try:
            from tqdm.auto import tqdm  
        except Exception:
            if not quiet:
                print("[WARN] tqdm not available; fallback to log files.")
            live = False

    if serialize_output:
        for i, part in enumerate(tasks_buckets):
            if not part:
                if not quiet:
                    print(f"[{tag}] (SERIAL) bucket{i}: no tasks.")
                continue
            gpu_id = gpus[i] if i < len(gpus) else None
            lf = write_taskfile_for_evi(part, tmp_root / f"gpu{i}")
            if not quiet:
                print(f"[{tag}] (SERIAL) bucket{i} -> {'GPU'+str(gpu_id) if gpu_id is not None else 'CPU'} | {len(part)} tasks")
            proc, logf = _spawn_worker(
                tag,
                evi_py,
                model_local_path,
                base_out_dir,
                lf,
                gpu_id,
                extra_opts,
                live=False,
                log_file=(log_root / f"gpu{i}" / f"{tag.replace(':', '_')}.log"),
            )
            code = proc.wait()
            if logf:
                logf.close()
            if code != 0:
                raise RuntimeError(f"[{tag}] serial bucket{i} failed with code {code}")
        if not quiet:
            print(f"[{tag}] done (serialized).")
        shutil.rmtree(tmp_root, ignore_errors=True)
        return

    for i, part in enumerate(tasks_buckets):
        gpu_id = gpus[i] if i < len(gpus) else None
        if not part:
            if not quiet:
                print(f"[{tag}] GPU{gpu_id if gpu_id is not None else 'CPU'}: no tasks.")
            procs.append(None)
            resources.append((None, None, 0))
            continue
        lf = write_taskfile_for_evi(part, tmp_root / f"gpu{i}")
        if live:
            proc, _ = _spawn_worker(
                tag, evi_py, model_local_path, base_out_dir, lf, gpu_id, extra_opts, live=True, log_file=None
            )
            procs.append(proc)
            resources.append((None, None, len(part)))
        else:
            logf_path = log_root / f"gpu{i}" / f"{tag.replace(':', '_')}.log"
            proc, logf = _spawn_worker(
                tag, evi_py, model_local_path, base_out_dir, lf, gpu_id, extra_opts, live=False, log_file=logf_path
            )
            if not quiet:
                print(f"[{tag}] GPU{gpu_id if gpu_id is not None else 'CPU'}: {len(part)} tasks -> {lf.name} | log={logf_path}")
            procs.append(proc)
            resources.append((logf, logf_path, len(part)))

    if not live:
        codes: List[int] = []
        for proc, (logf, _, _) in zip(procs, resources):
            if proc is None:
                continue
            code = proc.wait()
            if logf:
                logf.close()
            codes.append(code)
        if any(c != 0 for c in codes):
            raise RuntimeError(f"[{tag}] some workers failed: {codes}")
        if not quiet:
            print(f"[{tag}] done.")
        shutil.rmtree(tmp_root, ignore_errors=True)
        return

    from tqdm.auto import tqdm

    totals = [r[2] for r in resources]
    bars: List[Optional["tqdm"]] = []
    for i, total in enumerate(totals):
        if total <= 0:
            bars.append(None)
            continue
        gpu_id = gpus[i] if i < len(gpus) else None
        desc = f"{tag}|GPU{gpu_id if gpu_id is not None else 'CPU'}"
        bars.append(tqdm(total=total, desc=desc, position=i, leave=True, dynamic_ncols=True))

    qout: "queue.Queue[Tuple[int, Optional[str]]]" = queue.Queue()
    threads = []
    for i, proc in enumerate(procs):
        if proc is None or proc.stdout is None:
            continue
        th = threading.Thread(target=_reader_thread, args=(proc, i, qout), daemon=True)
        th.start()
        threads.append(th)

    finished = [False] * len(procs)
    alive = sum(1 for p in procs if p is not None)

    while alive > 0:
        try:
            gpu_idx, line = qout.get(timeout=0.5)
        except queue.Empty:
            for b in bars:
                if b:
                    b.refresh()
            done_now = 0
            for i, p in enumerate(procs):
                if p is None or finished[i]:
                    continue
                if p.poll() is not None:
                    finished[i] = True
                    done_now += 1
            alive -= done_now
            continue

        if line is None:
            continue

        m = _OK_RE.search(line)
        if m and bars[gpu_idx] is not None:
            bars[gpu_idx].update(1)
        if m and (not quiet):
            print(f"[{tag}|GPU{gpus[gpu_idx] if gpu_idx < len(gpus) else 'CPU'}] {m.group(0)}")

        if procs[gpu_idx] is not None and procs[gpu_idx].poll() is not None and not finished[gpu_idx]:
            finished[gpu_idx] = True
            alive -= 1

    for b in bars:
        if b:
            b.close()
    codes = [p.wait() for p in procs if p is not None]
    if any(c != 0 for c in codes):
        raise RuntimeError(f"[{tag}] some workers failed: {codes}")
    if not quiet:
        print(f"[{tag}] done.")
    shutil.rmtree(tmp_root, ignore_errors=True)

def extract_evidence_dist(
    evi_py: Path,
    model_local_path: Path,
    ref_video_dir: Optional[Path],
    ref_out_dir: Path,
    gen_video_root: Path,
    gen_out_root: Path,
    include_models: List[str],
    exclude_models: List[str],
    gpus: List[str],
    force: bool,
    limit: int,
    extra_evi_opts: List[str],
    verify: bool,
    live: bool,
    serialize_output: bool,
    quiet: bool,
):
    if (not gpus) and (not quiet):
        print("[WARN] No GPUs detected, will run on CPU (very slow).")

    if not quiet:
        print("\n=== [Stage 1/2] Extract REF evidence ===")
    ref_out_dir.mkdir(parents=True, exist_ok=True)

    if ref_video_dir is None:
        existing = list(ref_out_dir.glob("*.json"))
        if existing and (not quiet):
            print(f"[Ref] ref-video-dir is None; reuse existing {len(existing)} JSON files under {ref_out_dir}")
        elif (not existing) and (not quiet):
            print(f"[Ref] ref-video-dir is None and no existing evidence found under {ref_out_dir} (nothing to extract).")
        ref_tasks: List[Tuple[Path, Path]] = []
    elif not ref_video_dir.exists():
        if not quiet:
            print(f"[WARN] REF video dir not found: {ref_video_dir}; will only reuse existing evidence in {ref_out_dir}")
        ref_tasks = []
    else:
        ref_tasks = build_ref_tasks(ref_video_dir, ref_out_dir, force=force, limit=limit)

    if ref_tasks:
        buckets = chunk_even(ref_tasks, max(1, len(gpus)) if gpus else 1)
        _run_buckets(
            tag="ref",
            evi_py=evi_py,
            model_local_path=model_local_path,
            base_out_dir=ref_out_dir,
            tasks_buckets=buckets,
            gpus=gpus or [],
            extra_opts=extra_evi_opts,
            live=live,
            serialize_output=serialize_output,
            quiet=quiet,
        )
        if verify:
            ok, bad = verify_bindings(ref_tasks, quiet=quiet)
            if not quiet:
                print(f"[VERIFY/REF] ok={ok}, bad={bad}")
    else:
        if not quiet:
            print("[Skip] No REF tasks to run.")

    if not quiet:
        print("\n=== [Stage 2/2] Extract GEN evidence (multi-model) ===")
    if not gen_video_root.exists():
        if not quiet:
            print(f"[WARN] gen-video-root not found: {gen_video_root}")
        return

    model_dirs: List[Path] = []
    if any(is_video(p) for p in gen_video_root.iterdir() if p.is_file()):
        model_dirs = [gen_video_root]
    else:
        for d in sorted([p for p in gen_video_root.iterdir() if p.is_dir()]):
            name = d.name
            if include_models and (name not in include_models):
                continue
            if exclude_models and (name in exclude_models):
                continue
            if list_videos(d):
                model_dirs.append(d)

    if not model_dirs:
        if not quiet:
            print(f"[Skip] No model dirs found under: {gen_video_root}")
        return

    for md in model_dirs:
        mname = md.name
        if not quiet:
            print(f"\n--- [Model] {mname} ---")
        out_dir_m = (gen_out_root / mname).resolve()
        tasks = build_gen_tasks_for_model(md, out_dir_m, force=force, limit=limit)
        if not tasks:
            if not quiet:
                print(f"[Skip] Model {mname}: no tasks to run (all done).")
            continue
        buckets = chunk_even(tasks, max(1, len(gpus)) if gpus else 1)
        _run_buckets(
            tag=f"gen:{mname}",
            evi_py=evi_py,
            model_local_path=model_local_path,
            base_out_dir=out_dir_m,
            tasks_buckets=buckets,
            gpus=gpus or [],
            extra_opts=extra_evi_opts,
            live=live,
            serialize_output=serialize_output,
            quiet=quiet,
        )
        if verify:
            ok, bad = verify_bindings(tasks, quiet=quiet)
            if not quiet:
                print(f"[VERIFY/GEN:{mname}] ok={ok}, bad={bad}")

def score_softalign_per_model(
    batch_scoring_py: Path,
    yaml_path: Path,
    ref_evi_dir: Path,
    gen_evi_root: Path,
    include_models: List[str],
    exclude_models: List[str],
    out_dir: Optional[Path] = None,
    limit: int = 0,
    pass_through: Optional[List[str]] = None,
    quiet: bool = False,
) -> int:
    if not gen_evi_root.exists():
        if not quiet:
            print(f"[WARN] gen evidence root not found: {gen_evi_root}")
        return 0

    model_dirs: List[Path] = []
    has_subdir = any(d.is_dir() for d in gen_evi_root.iterdir()) if gen_evi_root.exists() else False
    if has_subdir:
        for d in sorted([p for p in gen_evi_root.iterdir() if p.is_dir()]):
            name = d.name
            if include_models and (name not in include_models):
                continue
            if exclude_models and (name in exclude_models):
                continue
            if list(d.rglob("*.json")):
                model_dirs.append(d)
    else:
        if list(gen_evi_root.rglob("*.json")):
            model_dirs = [gen_evi_root]

    if not model_dirs:
        if not quiet:
            print(f"[Skip] No model evidence dirs found under: {gen_evi_root}")
        return 0

    scored = 0
    merged_rows: List[dict] = []
    merged_fieldnames: List[str] = []
    for md in model_dirs:
        mname = md.name
        cmd = [
            sys.executable,
            str(batch_scoring_py),
            "--yaml",
            str(yaml_path),
            "--ref-dir",
            str(ref_evi_dir),
            "--gen-dir",
            str(md),
        ]
        if out_dir:
            cmd += ["--out-dir", str(out_dir)]
        if limit and limit > 0:
            cmd += ["--limit", str(limit)]
        if pass_through:
            cmd += pass_through
        if not quiet:
            print(f"\n[Score] {mname}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"[Score] failed for model {mname} with code {proc.returncode}")
        if out_dir:
            summary_csv = out_dir / "semantic_scores_summary.csv"
            if summary_csv.exists():
                with summary_csv.open(newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames and not merged_fieldnames:
                        merged_fieldnames = list(reader.fieldnames)
                    for row in reader:
                        merged_rows.append(row)
        scored += 1

    if out_dir and len(model_dirs) > 1 and merged_rows:
        summary_csv = out_dir / "semantic_scores_summary.csv"
        deduped: dict[tuple[str, str], dict] = {}
        for row in merged_rows:
            key = (row.get("modelname", ""), row.get("sample_id", ""))
            deduped[key] = row
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
            writer.writeheader()
            writer.writerows(deduped.values())
        if not quiet:
            print(f"[summary] merged {len(deduped)} rows -> {summary_csv}")
    return scored

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Multi-GPU / multi-model batch runner for semantic evidence extraction + scoring "
            "(new schema: hal / semantic_score / semantic_score_0_100)."
        )
    )
    ap.add_argument("--evi-extract-py", type=Path, required=True, help="MiniCPM semantic evidence extractor script path")
    ap.add_argument("--batch-scoring-py", type=Path, required=True, help="SoftAlign batch scoring script path")
    ap.add_argument("--softalign-yaml", type=Path, required=True, help="SoftAlign config YAML path")
    ap.add_argument("--ref-video-dir", type=Path, default=None, help="Reference video root (flat; optional)")
    ap.add_argument("--gen-video-root", type=Path, required=True, help="Generated video root (subdirs are model names, or a single-model dir)")
    ap.add_argument("--ref-out-dir", type=Path, required=True, help="Reference evidence output dir (flat)")
    ap.add_argument("--gen-out-root", type=Path, required=True, help="Generated evidence output root (subdir per model)")
    ap.add_argument("--model-local-path", type=Path, required=True, help="MiniCPM-V-4_5 local checkpoint dir")

    ap.add_argument("--gpus", default="auto", help="GPU list, e.g. '0,1,2,3'; default=auto")
    ap.add_argument("--include-models", default="", help="Only evaluate these models, comma-separated; empty=all")
    ap.add_argument("--exclude-models", default="", help="Exclude these models, comma-separated")

    ap.add_argument("--force", action="store_true", help="Force re-run extraction and scoring instead of reusing existing outputs")
    ap.add_argument("--limit", type=int, default=0, help="At most process the first N samples per split (debug use)")
    ap.add_argument("--steps", default="both", choices=["extract", "score", "both"], help="extract only / score only / full pipeline")
    ap.add_argument("--scores-out-dir", type=Path, default=None, help="Score output dir (if omitted, batch_scoring.py default is used)")

    ap.add_argument("--verify", action="store_true", help="Enable post-extraction binding verification")
    ap.add_argument("--live", action="store_true", help="Live terminal progress display (requires tqdm)")
    ap.add_argument("--serialize-output", action="store_true", help="Serialize GPU buckets to avoid any interleaved output")

    ap.add_argument("--extra-evi-opts", nargs=argparse.REMAINDER, help="Pass-through extra options for evi_extract.py")
    ap.add_argument("--quiet", action="store_true", help="Reduce logs; keep only essential messages")

    return ap.parse_args()

def main():
    args = parse_args()
    gpus = detect_gpus(args.gpus)
    include_models = [x for x in (args.include_models or "").split(",") if x.strip()]
    exclude_models = [x for x in (args.exclude_models or "").split(",") if x.strip()]
    quiet = bool(args.quiet)

    if args.steps in ("extract", "both"):
        extract_evidence_dist(
            evi_py=args.evi_extract_py.resolve(),
            model_local_path=args.model_local_path.resolve(),
            ref_video_dir=(args.ref_video_dir.resolve() if args.ref_video_dir is not None else None),
            ref_out_dir=args.ref_out_dir.resolve(),
            gen_video_root=args.gen_video_root.resolve(),
            gen_out_root=args.gen_out_root.resolve(),
            include_models=include_models,
            exclude_models=exclude_models,
            gpus=gpus,
            force=args.force,
            limit=args.limit,
            extra_evi_opts=(args.extra_evi_opts or []),
            verify=args.verify,
            live=args.live,
            serialize_output=args.serialize_output,
            quiet=quiet,
        )

    if args.steps in ("score", "both"):
        pass_through: List[str] = []
        if args.force:
            pass_through.append("--force")

        scored = score_softalign_per_model(
            batch_scoring_py=args.batch_scoring_py.resolve(),
            yaml_path=args.softalign_yaml.resolve(),
            ref_evi_dir=args.ref_out_dir.resolve(),
            gen_evi_root=args.gen_out_root.resolve(),
            include_models=include_models,
            exclude_models=exclude_models,
            out_dir=(args.scores_out_dir.resolve() if args.scores_out_dir else None),
            limit=args.limit,
            pass_through=pass_through or None,
            quiet=quiet,
        )
        if scored <= 0:
            raise SystemExit("[semantic] no generated evidence/model dirs were scored; check GEN_VIDEO_ROOT, GEN_OUT_ROOT, and MODELS/INCLUDE_MODELS")

if __name__ == "__main__":
    main()
