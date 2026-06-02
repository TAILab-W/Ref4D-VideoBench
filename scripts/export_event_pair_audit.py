#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


FIELD_SCORES = ("EGA", "ERel", "ECR")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
            n += 1
    return n


def rel_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return x


def score_from_metric(obj: Any) -> Optional[float]:
    if isinstance(obj, Mapping):
        return as_float(obj.get("score"))
    return None


def load_events(path: Path) -> List[Dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, Mapping):
        data = data.get("events", [])
    if not isinstance(data, list):
        raise ValueError(f"events JSON must be a list or {{events: [...]}}: {path}")
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, Mapping):
            continue
        eid = str(item.get("id") or item.get("eid") or item.get("event_id") or "").strip()
        if not eid:
            eid = f"event_{idx:04d}"
        out.append(
            {
                "id": eid,
                "text": str(item.get("text", "") or "").strip(),
                "s": as_float(item.get("s")),
                "e": as_float(item.get("e")),
                "dur": as_float(item.get("dur")),
                "s_abs": as_float(item.get("s_abs")),
                "e_abs": as_float(item.get("e_abs")),
                "dur_abs": as_float(item.get("dur_abs")),
                "members": list(item.get("members", [])) if isinstance(item.get("members", []), list) else [],
            }
        )
    return out


def event_map(events: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(e.get("id")): e for e in events if e.get("id") is not None}


def parse_score_path(path: Path, scores_root: Path) -> Tuple[Tuple[str, ...], str, str, str]:
    rel = path.resolve().relative_to(scores_root.resolve())
    parts = rel.parts
    if len(parts) < 3 or parts[-1] != "event_scores.json":
        raise ValueError(f"unexpected score path shape: {path}")
    pair_id = parts[-2]
    model = parts[-3]
    prefix = tuple(parts[:-3])
    sample_id = pair_id.rsplit("__", 1)[0] if "__" in pair_id else pair_id
    return prefix, model, pair_id, sample_id


def candidate_cache_roots(cache_root: Path, prefix: Sequence[str]) -> List[Path]:
    roots: List[Path] = []
    if prefix:
        roots.append(cache_root.joinpath(*prefix))
    roots.append(cache_root)
    out: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            out.append(root)
            seen.add(key)
    return out


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def load_pairs(path: Optional[Path]) -> Tuple[List[List[Any]], Dict[str, Any]]:
    if path is None:
        return [], {}
    data = read_json(path)
    if not isinstance(data, Mapping):
        return [], {}
    pairs = data.get("M", [])
    meta = data.get("meta", {})
    return (pairs if isinstance(pairs, list) else []), (meta if isinstance(meta, dict) else {})


def event_public(e: Mapping[str, Any], matched_id: Optional[str] = None) -> Dict[str, Any]:
    out = dict(e)
    if matched_id is not None:
        out["matched_id"] = matched_id
    return out


def build_matched_pairs(
    pairs: Sequence[Sequence[Any]],
    ref_by_id: Mapping[str, Mapping[str, Any]],
    gen_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], set[str], set[str]]:
    rows: List[Dict[str, Any]] = []
    matched_ref: set[str] = set()
    matched_gen: set[str] = set()
    for item in pairs:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        ref_id = str(item[0])
        gen_id = str(item[1])
        detail = item[2] if len(item) >= 3 and isinstance(item[2], Mapping) else {}
        ref_event = ref_by_id.get(ref_id, {})
        gen_event = gen_by_id.get(gen_id, {})
        matched_ref.add(ref_id)
        matched_gen.add(gen_id)
        rows.append(
            {
                "ref_id": ref_id,
                "gen_id": gen_id,
                "ref_text": ref_event.get("text", ""),
                "gen_text": gen_event.get("text", ""),
                "ref_span": {
                    "s": ref_event.get("s"),
                    "e": ref_event.get("e"),
                    "s_abs": ref_event.get("s_abs"),
                    "e_abs": ref_event.get("e_abs"),
                },
                "gen_span": {
                    "s": gen_event.get("s"),
                    "e": gen_event.get("e"),
                    "s_abs": gen_event.get("s_abs"),
                    "e_abs": gen_event.get("e_abs"),
                },
                "sim_sem": as_float(detail.get("sim_sem")),
                "r_tIoU": as_float(detail.get("r_tIoU")),
                "q": as_float(detail.get("q")),
            }
        )
    return rows, matched_ref, matched_gen


def _np_str_list(arr: Any) -> List[str]:
    return [str(x) for x in np.asarray(arr).tolist()]


def load_candidate_pairs(
    gate_path: Optional[Path],
    ref_by_id: Mapping[str, Mapping[str, Any]],
    gen_by_id: Mapping[str, Mapping[str, Any]],
    w1: float,
    w2: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    if gate_path is None or top_k <= 0:
        return []
    try:
        dat = np.load(gate_path, allow_pickle=True)
        ref_ids = _np_str_list(dat["ref_ids"])
        gen_ids = _np_str_list(dat["gen_ids"])
        sim = np.asarray(dat["sim_sem"], dtype=float)
        rt = np.asarray(dat["r_tiou"], dtype=float)
        gate = np.asarray(dat["gate"], dtype=bool)
    except Exception as exc:
        return [{"error": f"failed to load gate candidates: {exc}", "path": gate_path.as_posix()}]

    rows: List[Dict[str, Any]] = []
    for i, ref_id in enumerate(ref_ids):
        scored: List[Tuple[float, int]] = []
        for j, _gen_id in enumerate(gen_ids):
            q = float(w1 * sim[i, j] + w2 * rt[i, j])
            scored.append((q, j))
        scored.sort(key=lambda x: x[0], reverse=True)
        for rank, (q, j) in enumerate(scored[:top_k], start=1):
            gen_id = gen_ids[j]
            rows.append(
                {
                    "ref_id": ref_id,
                    "gen_id": gen_id,
                    "rank_for_ref": rank,
                    "ref_text": ref_by_id.get(ref_id, {}).get("text", ""),
                    "gen_text": gen_by_id.get(gen_id, {}).get("text", ""),
                    "sim_sem": as_float(sim[i, j]),
                    "r_tIoU": as_float(rt[i, j]),
                    "q": as_float(q),
                    "gate": bool(gate[i, j]),
                }
            )
    return rows


def build_audit_rows(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    project_root = Path(args.project_root).resolve()
    event_root = Path(args.event_root)
    scores_root = Path(args.scores_root) if args.scores_root else event_root / "scores"
    cache_root = Path(args.cache_root) if args.cache_root else event_root / "cache"
    ref_events_root = Path(args.ref_events_root)

    model_filter = {x.strip() for x in args.models.split(",") if x.strip()} if args.models else set()
    sample_filter = {x.strip() for x in args.sample_ids.split(",") if x.strip()} if args.sample_ids else set()

    score_paths = sorted(scores_root.rglob("event_scores.json"))
    for score_path in score_paths:
        try:
            prefix, model, pair_id, sample_id = parse_score_path(score_path, scores_root)
        except Exception as exc:
            print(f"[audit] skip score path {score_path}: {exc}", file=sys.stderr)
            continue
        if model_filter and model not in model_filter:
            continue
        if sample_filter and sample_id not in sample_filter:
            continue

        cache_roots = candidate_cache_roots(cache_root, prefix)
        ref_events_path = ref_events_root / f"{sample_id}.newevents.json"
        gen_events_path = first_existing(root / "events_merged" / "gen" / f"{pair_id}.newevents.json" for root in cache_roots)
        pairs_path = first_existing(root / "match" / pair_id / "pairs.json" for root in cache_roots)
        gate_path = first_existing(root / "match" / pair_id / "gate_masks.npz" for root in cache_roots)

        missing = []
        if not ref_events_path.exists():
            missing.append(f"ref_events:{ref_events_path}")
        if gen_events_path is None:
            missing.append(f"gen_events:{pair_id}")
        if pairs_path is None:
            missing.append(f"pairs:{pair_id}")
        if missing:
            yield {
                "modelname": model,
                "sample_id": sample_id,
                "pair_id": pair_id,
                "status": "missing_inputs",
                "missing": missing,
                "score_path": rel_path(score_path, project_root),
            }
            continue

        scores = read_json(score_path)
        ref_events = load_events(ref_events_path)
        gen_events = load_events(gen_events_path)
        ref_by_id = event_map(ref_events)
        gen_by_id = event_map(gen_events)
        pairs, pairs_meta = load_pairs(pairs_path)
        matched_pairs, matched_ref, matched_gen = build_matched_pairs(pairs, ref_by_id, gen_by_id)
        ref_match_map = {p["ref_id"]: p["gen_id"] for p in matched_pairs}
        gen_match_map = {p["gen_id"]: p["ref_id"] for p in matched_pairs}
        w1 = float(pairs_meta.get("w1", args.default_w1))
        w2 = float(pairs_meta.get("w2", args.default_w2))

        yield {
            "modelname": model,
            "sample_id": sample_id,
            "pair_id": pair_id,
            "status": "ok",
            "counts": {
                "ref_events": len(ref_events),
                "gen_events": len(gen_events),
                "matched_pairs": len(matched_pairs),
                "unmatched_ref": len([e for e in ref_events if e["id"] not in matched_ref]),
                "unmatched_gen": len([e for e in gen_events if e["id"] not in matched_gen]),
            },
            "scores": {
                "EGA": score_from_metric(scores.get("EGA")),
                "ERel": score_from_metric(scores.get("ERel")),
                "ERel_valid": bool(scores.get("ERel", {}).get("valid", False)) if isinstance(scores.get("ERel"), Mapping) else None,
                "ECR": score_from_metric(scores.get("ECR")),
                "event_score": as_float(scores.get("event_score")),
                "event_score_0_100": as_float(scores.get("event_score_0_100")),
            },
            "paths": {
                "score": rel_path(score_path, project_root),
                "ref_events": rel_path(ref_events_path, project_root),
                "gen_events": rel_path(gen_events_path, project_root),
                "pairs": rel_path(pairs_path, project_root),
                "gate": rel_path(gate_path, project_root) if gate_path else None,
            },
            "ref_events": [event_public(e, ref_match_map.get(str(e["id"]))) for e in ref_events],
            "gen_events": [event_public(e, gen_match_map.get(str(e["id"]))) for e in gen_events],
            "matched_pairs": matched_pairs,
            "unmatched_ref_events": [event_public(e) for e in ref_events if e["id"] not in matched_ref],
            "unmatched_gen_events": [event_public(e) for e in gen_events if e["id"] not in matched_gen],
            "candidate_pairs_topk_by_ref": load_candidate_pairs(
                gate_path,
                ref_by_id,
                gen_by_id,
                w1=w1,
                w2=w2,
                top_k=int(args.top_k_candidates),
            ),
        }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export per-video-pair event audit records with merged event texts, matches, unmatched events, and top candidates."
    )
    ap.add_argument("--project-root", default=".", help="Repository root used for relative paths.")
    ap.add_argument("--event-root", default="outputs/event", help="Event output root containing scores/ and cache/.")
    ap.add_argument("--scores-root", default="", help="Override scores root. Defaults to <event-root>/scores.")
    ap.add_argument("--cache-root", default="", help="Override cache root. Defaults to <event-root>/cache.")
    ap.add_argument(
        "--ref-events-root",
        default="data/metadata/event_evidence/events_merged_ref",
        help="Reference merged event evidence root.",
    )
    ap.add_argument("--out", default="outputs/event/analysis/event_pair_audit.jsonl", help="Output JSONL path.")
    ap.add_argument("--models", default="", help="Optional comma-separated model filter.")
    ap.add_argument("--sample-ids", default="", help="Optional comma-separated sample_id filter.")
    ap.add_argument("--top-k-candidates", type=int, default=5, help="Per-reference top candidate pairs from gate matrix; 0 disables.")
    ap.add_argument("--default-w1", type=float, default=0.8)
    ap.add_argument("--default-w2", type=float, default=0.2)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    n = write_jsonl(Path(args.out), build_audit_rows(args))
    print(f"[event-audit] wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
