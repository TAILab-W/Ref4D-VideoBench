from __future__ import annotations

"""
Build canonical reference-side event caches from raw reference videos.

This entrypoint only builds reference-side assets:
  raw ref video -> scenes -> events -> vlm -> embeds -> merged ref evidence

Published outputs:
  - data/metadata/event_evidence/events_merged_ref/<sample_id>.newevents.json
  - data/metadata/event_evidence/embeds_merged_ref/<sample_id>.emb.merged.json

Intermediate workspace (kept for debugging / resuming):
  - outputs/event/cache/scenes/<sample_id>.scenes.json
  - outputs/event/cache/events/ref/<sample_id>.events.json
  - outputs/event/cache/vlm/ref/<sample_id>.vlm.json
  - outputs/event/cache/embeds/ref/<sample_id>.emb.json
  - outputs/event/cache/events_merged/ref/<sample_id>.newevents.json
  - outputs/event/cache/embeds/ref/<sample_id>.emb.merged.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None




PROJECT = Path(__file__).resolve().parents[2]
LAUNCHER: List[str] = [sys.executable]

TRANS_MODULE = "ref4d_eval.event.src.eventdetect.transnetv2_runner"
DDM_MODULE = "ref4d_eval.event.src.eventdetect.ddm_runner"
VLM_MODULE = "ref4d_eval.event.src.vlm.vllama3_infer"
EMBED_MODULE = "ref4d_eval.event.src.embed.e5_encoder"
MERGE_MODULE = "ref4d_eval.event.src.merge.merger"

VERBOSE = False






def _vprint(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **kwargs)


def _format_subprocess_failure(cmd: Sequence[str], result: subprocess.CompletedProcess) -> str:
    chunks: List[str] = []
    if isinstance(result.stdout, str) and result.stdout.strip():
        chunks.append(result.stdout.strip())
    if isinstance(result.stderr, str) and result.stderr.strip():
        chunks.append(result.stderr.strip())
    combined = "\n".join(chunks).strip()
    if combined:
        lines = combined.splitlines()
        tail = "\n".join(lines[-80:])
        return f"Command failed: {' '.join(cmd)}\n{tail}"
    return f"Command failed: {' '.join(cmd)}"






def _resolve_repo_relative_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _exists_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def p_ref_video(sample_id: str, ref_video_root: Path) -> Path:
    return ref_video_root / f"{sample_id}.mp4"


def p_scene(sample_id: str, work_root: Path) -> Path:
    return work_root / "scenes" / f"{sample_id}.scenes.json"


def p_events_ref(sample_id: str, work_root: Path) -> Path:
    return work_root / "events" / "ref" / f"{sample_id}.events.json"


def p_vlm_ref(sample_id: str, work_root: Path) -> Path:
    return work_root / "vlm" / "ref" / f"{sample_id}.vlm.json"


def p_emb_ref(sample_id: str, work_root: Path) -> Path:
    return work_root / "embeds" / "ref" / f"{sample_id}.emb.json"


def p_events_ref_merged_work(sample_id: str, work_root: Path) -> Path:
    return work_root / "events_merged" / "ref" / f"{sample_id}.newevents.json"


def p_emb_ref_merged_work(sample_id: str, work_root: Path) -> Path:
    return work_root / "embeds" / "ref" / f"{sample_id}.emb.merged.json"


def p_publish_event(sample_id: str, publish_root: Path) -> Path:
    return publish_root / "events_merged_ref" / f"{sample_id}.newevents.json"


def p_publish_embed(sample_id: str, publish_root: Path) -> Path:
    return publish_root / "embeds_merged_ref" / f"{sample_id}.emb.merged.json"






def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_scene_json(path: Path) -> None:
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
        raise ValueError(f"Invalid scenes json: {path}")
    prev_s: Optional[float] = None
    for idx, item in enumerate(data["scenes"]):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid scene[{idx}] in {path}")
        s = float(item[0])
        e = float(item[1])
        if s < 0 or not (e > s):
            raise ValueError(f"Invalid scene interval [{s}, {e}] in {path}")
        if prev_s is not None and s < prev_s:
            raise ValueError(f"Scene order decreased at index {idx} in {path}")
        prev_s = s


def _validate_event_json(path: Path) -> None:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Invalid events json: {path}")
    seen = set()
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"events[{idx}] must be a dict in {path}")
        eid = item.get("id") or item.get("eid") or item.get("event_id")
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"events[{idx}] missing non-empty id in {path}")
        if eid in seen:
            raise ValueError(f"duplicate event id '{eid}' in {path}")
        seen.add(eid)
        for key in ("s_abs", "e_abs", "s", "e"):
            if key not in item:
                raise ValueError(f"events[{idx}] missing '{key}' in {path}")
            float(item[key])
        s_abs = float(item["s_abs"])
        e_abs = float(item["e_abs"])
        s = float(item["s"])
        e = float(item["e"])
        if e_abs < s_abs:
            raise ValueError(f"events[{idx}] has e_abs < s_abs in {path}")
        if s < -1e-6 or e > 1.0 + 1e-6 or e < s - 1e-6:
            raise ValueError(f"events[{idx}] has invalid normalized interval in {path}")


def _validate_vlm_json(path: Path) -> None:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Invalid vlm json: {path}")
    seen = set()
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"vlm[{idx}] must be a dict in {path}")
        eid = item.get("id") or item.get("eid") or item.get("event_id")
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"vlm[{idx}] missing non-empty id in {path}")
        if eid in seen:
            raise ValueError(f"duplicate vlm id '{eid}' in {path}")
        seen.add(eid)
        for key in ("s_abs", "e_abs", "s", "e", "text"):
            if key not in item:
                raise ValueError(f"vlm[{idx}] missing '{key}' in {path}")
        if not isinstance(item["text"], str) or not item["text"].strip():
            raise ValueError(f"vlm[{idx}] has empty text in {path}")


def _validate_embed_json(path: Path) -> None:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Invalid embeds json: {path}")
    seen = set()
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"embeds[{idx}] must be a dict in {path}")
        eid = item.get("id") or item.get("eid") or item.get("event_id")
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"embeds[{idx}] missing non-empty id in {path}")
        if eid in seen:
            raise ValueError(f"duplicate embed id '{eid}' in {path}")
        seen.add(eid)
        emb = item.get("emb") or item.get("embedding") or item.get("vec")
        if not isinstance(emb, list) or len(emb) == 0:
            raise ValueError(f"embeds[{idx}] has empty embedding in {path}")
        if "text" not in item or not isinstance(item["text"], str) or not item["text"].strip():
            raise ValueError(f"embeds[{idx}] has empty text in {path}")
        for key in ("s", "e"):
            if key not in item:
                raise ValueError(f"embeds[{idx}] missing '{key}' in {path}")
            float(item[key])


def _validate_merged_events_json(path: Path) -> None:
    data = _read_json(path)
    if isinstance(data, dict) and "events" in data:
        events = data["events"]
    else:
        events = data
    if not isinstance(events, list):
        raise ValueError(f"Invalid merged events json: {path}")
    seen = set()
    for idx, item in enumerate(events):
        if not isinstance(item, dict):
            raise ValueError(f"merged events[{idx}] must be a dict in {path}")
        eid = item.get("id") or item.get("eid") or item.get("event_id")
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"merged events[{idx}] missing non-empty id in {path}")
        if eid in seen:
            raise ValueError(f"duplicate merged event id '{eid}' in {path}")
        seen.add(eid)
        for key in ("s", "e"):
            if key not in item:
                raise ValueError(f"merged events[{idx}] missing '{key}' in {path}")
            float(item[key])


def _validate_merged_embeds_json(path: Path) -> None:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Invalid merged embeds json: {path}")
    seen = set()
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"merged embeds[{idx}] must be a dict in {path}")
        eid = item.get("id") or item.get("eid") or item.get("event_id")
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"merged embeds[{idx}] missing non-empty id in {path}")
        if eid in seen:
            raise ValueError(f"duplicate merged embed id '{eid}' in {path}")
        seen.add(eid)
        emb = item.get("emb") or item.get("embedding") or item.get("vec")
        if not isinstance(emb, list) or len(emb) == 0:
            raise ValueError(f"merged embeds[{idx}] has empty embedding in {path}")


def _validate_merged_pair(events_path: Path, embeds_path: Path) -> None:
    events_data = _read_json(events_path)
    if isinstance(events_data, dict) and "events" in events_data:
        events_data = events_data["events"]
    if not isinstance(events_data, list):
        raise ValueError(f"Invalid merged events json: {events_path}")

    embeds_data = _read_json(embeds_path)
    if not isinstance(embeds_data, list):
        raise ValueError(f"Invalid merged embeds json: {embeds_path}")

    event_ids = set()
    for idx, item in enumerate(events_data):
        if not isinstance(item, dict):
            raise ValueError(f"merged events[{idx}] must be a dict in {events_path}")
        eid = item.get("id") or item.get("eid") or item.get("event_id")
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"merged events[{idx}] missing non-empty id in {events_path}")
        event_ids.add(eid)

    embed_ids = set()
    for idx, item in enumerate(embeds_data):
        if not isinstance(item, dict):
            raise ValueError(f"merged embeds[{idx}] must be a dict in {embeds_path}")
        eid = item.get("id") or item.get("eid") or item.get("event_id")
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"merged embeds[{idx}] missing non-empty id in {embeds_path}")
        embed_ids.add(eid)

    if event_ids != embed_ids:
        missing_embeds = sorted(event_ids - embed_ids)
        extra_embeds = sorted(embed_ids - event_ids)
        msg = []
        if missing_embeds:
            msg.append(f"missing embeds for ids {missing_embeds[:10]}")
        if extra_embeds:
            msg.append(f"extra embed ids {extra_embeds[:10]}")
        raise ValueError(
            f"Merged event/embed pair mismatch: {events_path} <-> {embeds_path}: {'; '.join(msg)}"
        )






def _run_module(project_root: Path, module: str, args: Sequence[str]) -> None:
    cmd = list(LAUNCHER) + ["-m", module] + [str(x) for x in args]

    env = os.environ.copy()
    if not VERBOSE:
        env.setdefault("PYTHONWARNINGS", "ignore")
        env.setdefault("TRANSFORMERS_VERBOSITY", "error")
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("TQDM_DISABLE", "1")

    if VERBOSE:
        print("[CMD]", " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=str(project_root), env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
        return

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(_format_subprocess_failure(cmd, result))


def _ensure_valid_stage(
    path: Path,
    validator,
    run_fn,
    *,
    force: bool,
    stage_name: str,
) -> None:
    if not force and _exists_nonempty(path):
        try:
            validator(path)
            _vprint(f"[{stage_name}] skip: {path.name}")
            return
        except Exception as exc:
            _vprint(f"[{stage_name}] existing output invalid, rebuilding: {path} -> {exc}")

    run_fn()
    if not _exists_nonempty(path):
        raise FileNotFoundError(f"{stage_name} output missing: {path}")
    validator(path)


def _ensure_valid_stage_pair(
    paths: Sequence[Tuple[Path, callable]],
    run_fn,
    *,
    force: bool,
    stage_name: str,
    pair_validator=None,
) -> None:
    if not force and all(_exists_nonempty(p) for p, _ in paths):
        try:
            for p, validator in paths:
                validator(p)
            if pair_validator is not None:
                pair_validator(*[p for p, _ in paths])
            names = ", ".join(p.name for p, _ in paths)
            _vprint(f"[{stage_name}] skip: {names}")
            return
        except Exception as exc:
            joined = ", ".join(str(p) for p, _ in paths)
            _vprint(f"[{stage_name}] existing outputs invalid, rebuilding: {joined} -> {exc}")

    run_fn()
    for p, validator in paths:
        if not _exists_nonempty(p):
            raise FileNotFoundError(f"{stage_name} output missing: {p}")
        validator(p)
    if pair_validator is not None:
        pair_validator(*[p for p, _ in paths])






def _atomic_copy(src: Path, dst: Path) -> None:
    _ensure_parent(dst)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dst.parent)) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copy2(src, tmp_path)
        tmp_path.replace(dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _publish_canonical(sample_id: str, work_root: Path, publish_root: Path, *, force: bool) -> Tuple[Path, Path]:
    src_evt = p_events_ref_merged_work(sample_id, work_root)
    src_emb = p_emb_ref_merged_work(sample_id, work_root)
    dst_evt = p_publish_event(sample_id, publish_root)
    dst_emb = p_publish_embed(sample_id, publish_root)

    _validate_merged_events_json(src_evt)
    _validate_merged_embeds_json(src_emb)
    _validate_merged_pair(src_evt, src_emb)

    def _needs_refresh(dst_evt: Path, dst_emb: Path) -> bool:
        if force:
            return True
        if not (_exists_nonempty(dst_evt) and _exists_nonempty(dst_emb)):
            return True
        try:
            _validate_merged_events_json(dst_evt)
            _validate_merged_embeds_json(dst_emb)
            _validate_merged_pair(dst_evt, dst_emb)
            return False
        except Exception as exc:
            _vprint(f"[publish] existing canonical asset invalid, replacing: {dst_evt}, {dst_emb} -> {exc}")
            return True

    if _needs_refresh(dst_evt, dst_emb):
        _atomic_copy(src_evt, dst_evt)
        _atomic_copy(src_emb, dst_emb)

    _validate_merged_events_json(dst_evt)
    _validate_merged_embeds_json(dst_emb)
    _validate_merged_pair(dst_evt, dst_emb)
    return dst_evt, dst_emb






def _published_assets_valid(sample_id: str, publish_root: Path) -> bool:
    evt = p_publish_event(sample_id, publish_root)
    emb = p_publish_embed(sample_id, publish_root)
    if not (_exists_nonempty(evt) and _exists_nonempty(emb)):
        return False
    try:
        _validate_merged_events_json(evt)
        _validate_merged_embeds_json(emb)
        _validate_merged_pair(evt, emb)
        return True
    except Exception:
        return False


def build_one_sample(
    *,
    project_root: Path,
    sample_id: str,
    topic: str,
    ref_video_root: Path,
    work_root: Path,
    publish_root: Path,
    cfg_default: Path,
    cfg_shot: Path,
    cfg_gebd: Path,
    cfg_vlm: Path,
    cfg_embed: Path,
    force: bool,
) -> Dict[str, str]:
    ref_video = p_ref_video(sample_id, ref_video_root)
    if not ref_video.exists():
        raise FileNotFoundError(f"raw reference video missing: {ref_video}")

    if not force and _published_assets_valid(sample_id, publish_root):
        _vprint(f"[build] skip published ref assets: sample_id={sample_id}")
        return {"sample_id": sample_id, "topic": topic, "status": "skipped_published"}

    scene_path = p_scene(sample_id, work_root)
    events_path = p_events_ref(sample_id, work_root)
    vlm_path = p_vlm_ref(sample_id, work_root)
    emb_path = p_emb_ref(sample_id, work_root)
    merged_evt_path = p_events_ref_merged_work(sample_id, work_root)
    merged_emb_path = p_emb_ref_merged_work(sample_id, work_root)

    _ensure_valid_stage(
        scene_path,
        _validate_scene_json,
        lambda: _run_module(
            project_root,
            TRANS_MODULE,
            ["--video", ref_video, "--out", scene_path, "--config", cfg_shot],
        ),
        force=force,
        stage_name="scene",
    )

    _ensure_valid_stage(
        events_path,
        _validate_event_json,
        lambda: _run_module(
            project_root,
            DDM_MODULE,
            ["--video", ref_video, "--out", events_path, "--config", cfg_gebd, "--scenes", scene_path],
        ),
        force=force,
        stage_name="events",
    )

    _ensure_valid_stage(
        vlm_path,
        _validate_vlm_json,
        lambda: _run_module(
            project_root,
            VLM_MODULE,
            ["--video", ref_video, "--events", events_path, "--config", cfg_vlm, "--out", vlm_path],
        ),
        force=force,
        stage_name="vlm",
    )

    _ensure_valid_stage(
        emb_path,
        _validate_embed_json,
        lambda: _run_module(
            project_root,
            EMBED_MODULE,
            ["--vlm", vlm_path, "--config", cfg_embed, "--out", emb_path],
        ),
        force=force,
        stage_name="embed",
    )

    _ensure_valid_stage_pair(
        [
            (merged_evt_path, _validate_merged_events_json),
            (merged_emb_path, _validate_merged_embeds_json),
        ],
        lambda: _run_module(
            project_root,
            MERGE_MODULE,
            [
                "--events", events_path,
                "--vlm", vlm_path,
                "--embeds", emb_path,
                "--out-root", work_root,
                "--cfg", cfg_default,
            ],
        ),
        force=force,
        stage_name="merge",
        pair_validator=_validate_merged_pair,
    )

    pub_evt, pub_emb = _publish_canonical(sample_id, work_root, publish_root, force=force)
    _vprint(f"[publish] sample_id={sample_id} -> {pub_evt.name}, {pub_emb.name}")

    return {"sample_id": sample_id, "topic": topic, "status": "built"}






def _parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def discover_samples(
    *,
    meta_path: Path,
    ref_video_root: Path,
    topics: Sequence[str],
    sample_ids: Sequence[str],
) -> List[Tuple[str, str]]:
    topic_filter = set(topics)
    sample_filter = set(sample_ids)
    items: List[Tuple[str, str]] = []
    seen = set()

    if meta_path.exists():
        seen_sid: Dict[str, int] = {}
        with meta_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                topic = str(obj.get("topic") or obj.get("theme") or "")
                sid = str(obj.get("sample_id") or obj.get("id") or "")
                if not sid:
                    continue
                if topic_filter and topic not in topic_filter:
                    continue
                if sample_filter and sid not in sample_filter:
                    continue
                if sid in seen_sid:
                    raise ValueError(
                        f"duplicate sample_id={sid} in metadata {meta_path}:{line_no}; "
                        f"first seen at line {seen_sid[sid]}"
                    )
                seen_sid[sid] = line_no
                key = (topic, sid)
                if key in seen:
                    continue
                items.append(key)
                seen.add(key)

    if not items:
        for mp4 in sorted(ref_video_root.glob("*.mp4")):
            sid = mp4.stem
            if sample_filter and sid not in sample_filter:
                continue
            items.append(("", sid))
        return items

    if sample_filter:
        for sid in sorted(sample_filter):
            if any(existing_sid == sid for _, existing_sid in items):
                continue
            if p_ref_video(sid, ref_video_root).exists():
                items.append(("", sid))

    return items






def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build canonical reference-side event caches.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_shared(sp):
        sp.add_argument("--cfg-default", required=True, help="Path to event default.yaml")
        sp.add_argument("--cfg-shot", required=True, help="Path to model_shot.yaml")
        sp.add_argument("--cfg-gebd", required=True, help="Path to model_gebd.yaml")
        sp.add_argument("--cfg-vlm", required=True, help="Path to model_vlm.yaml")
        sp.add_argument("--cfg-embed", required=True, help="Path to model_embed.yaml")
        sp.add_argument("--ref-video-root", default="data/refvideo", help="Raw reference video root")
        sp.add_argument("--work-root", default="outputs/event/cache", help="Intermediate event cache root")
        sp.add_argument("--publish-root", default="data/metadata/event_evidence", help="Canonical published event evidence root")
        sp.add_argument("--meta-path", default="data/metadata/ref4d_meta.jsonl", help="Reference metadata jsonl for batch discovery")
        sp.add_argument("--force", action="store_true", help="Rebuild even if outputs already exist")
        sp.add_argument("--verbose", action="store_true", help="Show per-stage logs and child process output")

    p_run = sub.add_parser("run", help="Build one reference sample")
    p_run.add_argument("--sample-id", required=True)
    p_run.add_argument("--topic", default="")
    add_shared(p_run)

    p_batch = sub.add_parser("batch", help="Build multiple reference samples")
    p_batch.add_argument("--topics", default="", help="Optional comma-separated topic filter")
    p_batch.add_argument("--sample-ids", default="", help="Optional comma-separated sample_id filter")
    add_shared(p_batch)

    return ap.parse_args()






def main() -> None:
    global VERBOSE

    args = parse_args()
    VERBOSE = bool(getattr(args, "verbose", False))

    project_root = PROJECT
    cfg_default = _resolve_repo_relative_path(args.cfg_default, project_root)
    cfg_shot = _resolve_repo_relative_path(args.cfg_shot, project_root)
    cfg_gebd = _resolve_repo_relative_path(args.cfg_gebd, project_root)
    cfg_vlm = _resolve_repo_relative_path(args.cfg_vlm, project_root)
    cfg_embed = _resolve_repo_relative_path(args.cfg_embed, project_root)
    ref_video_root = _resolve_repo_relative_path(args.ref_video_root, project_root)
    work_root = _resolve_repo_relative_path(args.work_root, project_root)
    publish_root = _resolve_repo_relative_path(args.publish_root, project_root)
    meta_path = _resolve_repo_relative_path(args.meta_path, project_root)

    if args.cmd == "run":
        result = build_one_sample(
            project_root=project_root,
            sample_id=args.sample_id,
            topic=str(args.topic or ""),
            ref_video_root=ref_video_root,
            work_root=work_root,
            publish_root=publish_root,
            cfg_default=cfg_default,
            cfg_shot=cfg_shot,
            cfg_gebd=cfg_gebd,
            cfg_vlm=cfg_vlm,
            cfg_embed=cfg_embed,
            force=bool(args.force),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    topics = _parse_csv_arg(args.topics)
    sample_ids = _parse_csv_arg(args.sample_ids)
    samples = discover_samples(
        meta_path=meta_path,
        ref_video_root=ref_video_root,
        topics=topics,
        sample_ids=sample_ids,
    )

    built = 0
    skipped = 0
    failed = 0

    iterable = samples
    pbar = None
    if tqdm is not None:
        pbar = tqdm(samples, total=len(samples), desc="build_ref_event_cache", unit="sample")
        iterable = pbar
    elif VERBOSE:
        print(f"[batch] found {len(samples)} reference samples")

    for topic, sample_id in iterable:
        if VERBOSE:
            print(f"\n=== BUILD REF: topic={topic} sample_id={sample_id} ===")
        try:
            result = build_one_sample(
                project_root=project_root,
                sample_id=sample_id,
                topic=topic,
                ref_video_root=ref_video_root,
                work_root=work_root,
                publish_root=publish_root,
                cfg_default=cfg_default,
                cfg_shot=cfg_shot,
                cfg_gebd=cfg_gebd,
                cfg_vlm=cfg_vlm,
                cfg_embed=cfg_embed,
                force=bool(args.force),
            )
            if result.get("status") == "skipped_published":
                skipped += 1
            else:
                built += 1
        except Exception as exc:
            failed += 1
            msg = f"[ERROR] sample_id={sample_id} -> {exc}"
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg)

    if pbar is not None:
        pbar.close()

    summary = {
        "n_total": len(samples),
        "n_built": built,
        "n_skipped_published": skipped,
        "n_failed": failed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
