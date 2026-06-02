from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_META_PATH = "data/metadata/ref4d_meta.jsonl"
DEFAULT_SEMANTIC_ROOT = "data/metadata/semantic_evidence"
DEFAULT_EVENT_ROOT = "data/metadata/event_evidence/events_merged_ref"
DEFAULT_OUT_ROOT = "data/metadata/semantic_event_evidence"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def _load_meta(meta_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = str(obj.get("sample_id") or "").strip()
            if not sid:
                raise ValueError(f"{meta_path}:{line_no} missing sample_id")
            rows.append(obj)
    return rows


def _candidate_paths(root: Path, sample_id: str, names: Iterable[str]) -> List[Path]:
    return [root / name.format(sample_id=sample_id) for name in names]


def _find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _find_semantic_path(root: Path, sample_id: str) -> Optional[Path]:
    direct = _candidate_paths(
        root,
        sample_id,
        (
            "{sample_id}.json",
            "{sample_id}.semantic_evidence.json",
            "{sample_id}_semantic.json",
            "{sample_id}_semantic_evidence.json",
        ),
    )
    found = _find_first_existing(direct)
    if found is not None:
        return found
    matches = sorted(root.glob(f"{sample_id}*.json"))
    return _find_first_existing(matches)


def _find_event_path(root: Path, sample_id: str) -> Optional[Path]:
    direct = _candidate_paths(
        root,
        sample_id,
        (
            "{sample_id}.newevents.json",
            "{sample_id}.events.json",
            "{sample_id}.json",
            "{sample_id}_events.json",
        ),
    )
    found = _find_first_existing(direct)
    if found is not None:
        return found
    matches = sorted(root.glob(f"{sample_id}*.json"))
    return _find_first_existing(matches)


def _portable_path(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _validate_semantic_payload(payload: Any, path: Path) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"semantic evidence must be a JSON object: {path}")
    fine = payload.get("fine")
    if isinstance(fine, dict) and isinstance(fine.get("entities"), list):
        return
    if isinstance(payload.get("entities"), list):
        return
    raise ValueError(f"semantic evidence has no fine.entities/entities list: {path}")


def _validate_event_payload(payload: Any, path: Path) -> None:
    events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError(f"event evidence must be a list or object with events list: {path}")
    for idx, item in enumerate(events):
        if not isinstance(item, dict):
            raise ValueError(f"event[{idx}] must be an object: {path}")
        eid = item.get("id") or item.get("eid") or item.get("event_id")
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"event[{idx}] missing id/eid/event_id: {path}")


def build_one(
    *,
    sample: Dict[str, Any],
    semantic_root: Path,
    event_root: Path,
    out_root: Path,
    repo_root: Path,
    force: bool,
) -> Tuple[str, Path]:
    sample_id = str(sample["sample_id"])
    out_path = out_root / f"{sample_id}_semantic_event.json"
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return "skipped", out_path

    semantic_path = _find_semantic_path(semantic_root, sample_id)
    event_path = _find_event_path(event_root, sample_id)

    if semantic_path is None:
        raise FileNotFoundError(f"semantic evidence missing for sample_id={sample_id} under {semantic_root}")
    if event_path is None:
        raise FileNotFoundError(f"event evidence missing for sample_id={sample_id} under {event_root}")

    semantic_payload = _read_json(semantic_path)
    event_payload = _read_json(event_path)

    _validate_semantic_payload(semantic_payload, semantic_path)
    _validate_event_payload(event_payload, event_path)

    merged: Dict[str, Any] = {
        "sample_id": sample_id,
        "theme": sample.get("theme") or sample.get("topic"),
        "ref_video": sample.get("ref_video"),
        "semantic_source": _portable_path(semantic_path, repo_root),
        "event_source": _portable_path(event_path, repo_root),
        "semantic_evidence": semantic_payload,
        "event_evidence": event_payload,
    }

    _write_json(out_path, merged)
    return "built", out_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge reference semantic and event evidence into shared assets.")
    ap.add_argument("--semantic", default="", help="Backward-compatible single semantic evidence JSON.")
    ap.add_argument("--event", default="", help="Backward-compatible single event evidence JSON.")
    ap.add_argument("--out", default="", help="Backward-compatible single output JSON.")
    ap.add_argument("--repo-root", default="", help="Repository root; inferred by default.")
    ap.add_argument("--meta-path", default=DEFAULT_META_PATH)
    ap.add_argument("--semantic-root", default=DEFAULT_SEMANTIC_ROOT)
    ap.add_argument("--event-root", default=DEFAULT_EVENT_ROOT)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--sample-id", default="", help="Optional single sample_id.")
    ap.add_argument("--limit", type=int, default=0, help="Optional max samples for debugging.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="Continue when an input sample is missing evidence.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _resolve(args.repo_root, Path.cwd()).resolve() if args.repo_root else _repo_root()

    if args.semantic or args.event or args.out:
        if not (args.semantic and args.event and args.out):
            raise ValueError("--semantic, --event, and --out must be provided together")
        semantic_path = _resolve(args.semantic, repo_root)
        event_path = _resolve(args.event, repo_root)
        semantic_payload = _read_json(semantic_path)
        event_payload = _read_json(event_path)
        _validate_semantic_payload(semantic_payload, semantic_path)
        _validate_event_payload(event_payload, event_path)
        sample_id = semantic_path.name.split(".")[0]
        merged: Dict[str, Any] = {
            "sample_id": sample_id,
            "semantic_source": _portable_path(semantic_path, repo_root),
            "event_source": _portable_path(event_path, repo_root),
            "semantic_evidence": semantic_payload,
            "event_evidence": event_payload,
        }
        out_path = _resolve(args.out, repo_root)
        _write_json(out_path, merged)
        print(f"[built] {sample_id} -> {_portable_path(out_path, repo_root)}")
        return

    meta_path = _resolve(args.meta_path, repo_root)
    semantic_root = _resolve(args.semantic_root, repo_root)
    event_root = _resolve(args.event_root, repo_root)
    out_root = _resolve(args.out_root, repo_root)

    samples = _load_meta(meta_path)
    if args.sample_id:
        samples = [s for s in samples if str(s.get("sample_id")) == str(args.sample_id)]
        if not samples:
            raise ValueError(f"sample_id not found in metadata: {args.sample_id}")
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    counts = {"built": 0, "skipped": 0, "missing": 0, "failed": 0}
    for sample in samples:
        try:
            status, out_path = build_one(
                sample=sample,
                semantic_root=semantic_root,
                event_root=event_root,
                out_root=out_root,
                repo_root=repo_root,
                force=bool(args.force),
            )
            counts[status] = counts.get(status, 0) + 1
            print(f"[{status}] {sample['sample_id']} -> {_portable_path(out_path, repo_root)}")
        except FileNotFoundError as exc:
            counts["missing"] += 1
            if args.allow_missing:
                print(f"[missing] {sample.get('sample_id')}: {exc}")
                continue
            raise
        except Exception as exc:
            counts["failed"] += 1
            if args.allow_missing:
                print(f"[failed] {sample.get('sample_id')}: {type(exc).__name__}: {exc}")
                continue
            raise

    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
