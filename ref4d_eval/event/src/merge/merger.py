
"""
Merge adjacent items within a single video based on:
  - semantic gate (cosine similarity mapped to [0,1])
  - temporal adjacency (absolute-gap preferred; normalized-gap fallback)
and produce the merged event set for the video.

Outputs:
  outputs/event/cache/events_merged/{ref|gen}/<key>.newevents.json
      -> {"events": [...], "meta": {...}}
  outputs/event/cache/embeds/{ref|gen}/<key>.emb.merged.json
  outputs/event/cache/merge_map/{ref|gen}/<key>.json
"""

from __future__ import annotations
import os, json, math, argparse, hashlib
from typing import Dict, List, Tuple, Any, Optional






def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _infer_lane_and_key(events_path: str, out_dir_root: str) -> Tuple[str, str]:
    ap = os.path.abspath(events_path)
    rel = os.path.relpath(ap, os.path.abspath(out_dir_root))
    rel = rel.replace("\\", "/")
    if "/ref/" in rel:
        lane = "ref"
    elif "/gen/" in rel:
        lane = "gen"
    else:
        lane = "ref"
    stem = os.path.splitext(os.path.basename(ap))[0].replace(".events", "")
    return lane, stem







def _l2norm(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / s for x in vec]


def _cosine(u: List[float], v: List[float]) -> float:
    if not u or not v:
        return 0.0
    du = math.sqrt(sum(x * x for x in u))
    dv = math.sqrt(sum(x * x for x in v))
    if du == 0.0 or dv == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(u, v))
    return dot / (du * dv)


def _vec_add_inplace(acc: List[float], src: List[float], w: float):
    if not src:
        return
    if not acc:
        acc.extend([w * x for x in src])
    else:
        for i, x in enumerate(src):
            if i < len(acc):
                acc[i] += w * x
            else:
                acc.append(w * x)







def _require_nonempty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _to_finite_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _to_embedding_list(value: Any, name: str) -> List[float]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty list")
    out = [_to_finite_float(x, f"{name}[]") for x in value]
    return out


def _validate_time_fields(obj: Dict[str, Any], prefix: str) -> Dict[str, float]:
    if "s" not in obj or "e" not in obj:
        raise ValueError(f"{prefix} missing required normalized fields: s/e")
    s = _to_finite_float(obj["s"], f"{prefix}.s")
    e = _to_finite_float(obj["e"], f"{prefix}.e")
    if e < s:
        raise ValueError(f"{prefix} requires e >= s")
    out: Dict[str, float] = {"s": s, "e": e}

    has_s_abs = "s_abs" in obj
    has_e_abs = "e_abs" in obj
    if has_s_abs != has_e_abs:
        raise ValueError(f"{prefix} requires s_abs/e_abs to appear together")
    if has_s_abs and has_e_abs:
        s_abs = _to_finite_float(obj["s_abs"], f"{prefix}.s_abs")
        e_abs = _to_finite_float(obj["e_abs"], f"{prefix}.e_abs")
        if e_abs < s_abs:
            raise ValueError(f"{prefix} requires e_abs >= s_abs")
        out["s_abs"] = s_abs
        out["e_abs"] = e_abs
    return out


def _ensure_unique_ids(ids: List[str], what: str) -> None:
    seen = set()
    dup = set()
    for iid in ids:
        if iid in seen:
            dup.add(iid)
        seen.add(iid)
    if dup:
        raise ValueError(f"Duplicate ids found in {what}: {sorted(dup)}")







def _to_event_list(events_json) -> List[Dict[str, Any]]:
    if isinstance(events_json, dict) and "events" in events_json:
        evs = events_json["events"]
    else:
        evs = events_json
    if not isinstance(evs, list):
        raise ValueError("events input must be a list or an object with key 'events'")

    out: List[Dict[str, Any]] = []
    ids: List[str] = []
    for idx, raw in enumerate(evs):
        if not isinstance(raw, dict):
            raise ValueError(f"events[{idx}] must be an object")
        eid = raw.get("id") or raw.get("eid") or raw.get("event_id")
        eid = _require_nonempty_str(eid, f"events[{idx}].id")
        vals = _validate_time_fields(raw, f"events[{idx}] ({eid})")
        rec: Dict[str, Any] = {"id": eid, "s": vals["s"], "e": vals["e"]}
        if "s_abs" in vals and "e_abs" in vals:
            rec["s_abs"] = vals["s_abs"]
            rec["e_abs"] = vals["e_abs"]
        out.append(rec)
        ids.append(eid)

    _ensure_unique_ids(ids, "events")
    out.sort(key=lambda d: (d["s"], d["e"], d["id"]))
    return out


def _to_text_map_from_vlm(vlm_json) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if isinstance(vlm_json, dict) and all(isinstance(k, str) for k in vlm_json.keys()):
        _ensure_unique_ids(list(vlm_json.keys()), "vlm text map")
        for k, v in vlm_json.items():
            m[str(k)] = str(v).strip()
        return m

    if not isinstance(vlm_json, list):
        raise ValueError("vlm input must be a list or a dict keyed by event id")

    seen: List[str] = []
    for idx, item in enumerate(vlm_json):
        if not isinstance(item, dict):
            raise ValueError(f"vlm[{idx}] must be an object")
        eid = item.get("id") or item.get("eid") or item.get("event_id")
        eid = _require_nonempty_str(eid, f"vlm[{idx}].id")
        seen.append(eid)
        m[eid] = str(item.get("text", "")).strip()
    _ensure_unique_ids(seen, "vlm items")
    return m


def _is_micro_embeds(emb_json) -> bool:
    if not isinstance(emb_json, list) or not emb_json:
        return False
    for it in emb_json:
        if not isinstance(it, dict):
            continue
        if "parent_id" in it:
            return True
        eid = it.get("id")
        if isinstance(eid, str) and "#" in eid:
            return True
    return False


def _build_micro_items_from_embeds(
    emb_json,
    valid_event_ids: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], Dict[str, str]]:
    if not isinstance(emb_json, list):
        raise ValueError("micro-event embeds input must be a list")

    valid_event_id_set = set(valid_event_ids)
    items: List[Dict[str, Any]] = []
    emb_map: Dict[str, List[float]] = {}
    text_map: Dict[str, str] = {}
    ids: List[str] = []

    for idx, it in enumerate(emb_json):
        if not isinstance(it, dict):
            raise ValueError(f"embeds[{idx}] must be an object")
        eid = _require_nonempty_str(it.get("id"), f"embeds[{idx}].id")
        parent_id_raw = it.get("parent_id", None)
        if parent_id_raw is not None:
            parent_id = _require_nonempty_str(parent_id_raw, f"embeds[{idx}] ({eid}).parent_id")
        else:
            if "#" not in eid:
                raise ValueError(f"embeds[{idx}] ({eid}) missing parent_id and cannot infer it from id")
            parent_id = _require_nonempty_str(eid.split("#", 1)[0], f"embeds[{idx}] ({eid}).parent_id")
        if parent_id not in valid_event_id_set:
            raise ValueError(f"embeds[{idx}] ({eid}) parent_id '{parent_id}' not found in events")

        vals = _validate_time_fields(it, f"embeds[{idx}] ({eid})")
        emb = _to_embedding_list(it.get("emb") or it.get("embedding") or it.get("vec"), f"embeds[{idx}] ({eid}).emb")
        text = _require_nonempty_str(it.get("text", ""), f"embeds[{idx}] ({eid}).text")

        rec = {"id": eid, "s": vals["s"], "e": vals["e"]}
        if "s_abs" in vals and "e_abs" in vals:
            rec["s_abs"] = vals["s_abs"]
            rec["e_abs"] = vals["e_abs"]
        items.append(rec)
        emb_map[eid] = _l2norm(emb)
        text_map[eid] = text
        ids.append(eid)

    _ensure_unique_ids(ids, "micro-event embeds")
    items.sort(key=lambda d: (d["s"], d["e"], d["id"]))
    return items, emb_map, text_map


def _to_embed_map_event_level(emb_json, event_ids_in_order: List[str]) -> Dict[str, List[float]]:
    m: Dict[str, List[float]] = {}
    if isinstance(emb_json, list) and emb_json and isinstance(emb_json[0], dict):
        ids: List[str] = []
        for idx, item in enumerate(emb_json):
            if not isinstance(item, dict):
                raise ValueError(f"embeds[{idx}] must be an object")
            eid = item.get("id") or item.get("eid") or item.get("event_id")
            eid = _require_nonempty_str(eid, f"embeds[{idx}].id")
            emb = _to_embedding_list(item.get("emb") or item.get("embedding") or item.get("vec"), f"embeds[{idx}] ({eid}).emb")
            m[eid] = _l2norm(emb)
            ids.append(eid)
        _ensure_unique_ids(ids, "event-level embeds")
        return m

    if isinstance(emb_json, list):
        if len(emb_json) != len(event_ids_in_order):
            raise ValueError(
                f"aligned event-level embeds length mismatch: got {len(emb_json)} vs {len(event_ids_in_order)} events"
            )
        for idx, (eid, vec) in enumerate(zip(event_ids_in_order, emb_json)):
            emb = _to_embedding_list(vec, f"embeds[{idx}] ({eid}).emb")
            m[eid] = _l2norm(emb)
        return m

    raise ValueError("event-level embeds input must be a list")







def _event_duration_norm(ev: Dict[str, Any]) -> float:
    return max(0.0, float(ev.get("e", 0.0)) - float(ev.get("s", 0.0)))


def _event_duration_abs(ev: Dict[str, Any]) -> float:
    if "s_abs" in ev and "e_abs" in ev:
        return max(0.0, float(ev["e_abs"]) - float(ev["s_abs"]))
    return 0.0


def _cluster_gap_ok(cluster_end_ev: Dict[str, Any], next_ev: Dict[str, Any],
                    tau_gap_abs_sec: float, gap_norm_fallback: Optional[float]) -> Tuple[bool, float]:
    if cluster_end_ev.get("e_abs") is not None and next_ev.get("s_abs") is not None:
        gap = max(0.0, float(next_ev["s_abs"]) - float(cluster_end_ev["e_abs"]))
        return (gap <= tau_gap_abs_sec, gap)
    if gap_norm_fallback is not None:
        gap_n = max(0.0, float(next_ev["s"]) - float(cluster_end_ev["e"]))
        return (gap_n <= gap_norm_fallback, gap_n)
    gap = max(0.0, float(next_ev["s"]) - float(cluster_end_ev["e"]))
    return (True, gap)


def _make_merge_id(idx: int) -> str:
    return f"m{idx:04d}"


def _build_source_hash(events_path: str, vlm_path: str, embeds_path: str, cfg: dict) -> str:
    h = hashlib.sha256()
    for p in [events_path, vlm_path, embeds_path]:
        try:
            with open(p, "rb") as f:
                h.update(f.read(1024))
        except Exception:
            pass
    h.update(json.dumps(cfg or {}, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:12]


def _build_meta(cfg: dict, stats: dict) -> Dict[str, Any]:
    merge_cfg = (cfg or {}).get("merge", {}) or {}
    return {
        "lane": stats.get("lane"),
        "key": stats.get("key"),
        "mode": stats.get("mode"),
        "merge_config": {
            "tau_sem": float(merge_cfg.get("tau_sem", 0.85)),
            "gap_abs_sec": float(merge_cfg.get("gap_abs_sec", 0.30)),
            "gap_norm_fallback": merge_cfg.get("gap_norm_fallback", 0.01),
            "min_duration_norm": float(merge_cfg.get("min_duration_norm", 0.0)),
        },
        "stats": {
            "num_input_items": int(stats.get("num_input_items", 0)),
            "num_kept_items": int(stats.get("num_kept_items", 0)),
            "num_clusters": int(stats.get("num_clusters", 0)),
        },
        "provenance": {
            "source_hash": stats.get("source_hash"),
        },
    }


def _start_track(it: Dict[str, Any], have_abs: bool, it_text, it_vec) -> Dict[str, Any]:
    cs, ce = float(it["s"]), float(it["e"])
    cs_abs = float(it.get("s_abs", 0.0)) if have_abs else None
    ce_abs = float(it.get("e_abs", 0.0)) if have_abs else None
    v0 = it_vec(it["id"])
    w0 = (float(it["e_abs"]) - float(it["s_abs"])) if have_abs else (ce - cs)
    sum_vec: List[float] = []
    _vec_add_inplace(sum_vec, v0, w0)
    return {
        "members": [it],
        "cs": cs,
        "ce": ce,
        "cs_abs": cs_abs,
        "ce_abs": ce_abs,
        "rep_text": it_text(it["id"]),
        "sum_vec": sum_vec,
        "sum_w": w0,
    }


def _append_track(track: Dict[str, Any], cur: Dict[str, Any], v_next: List[float], have_abs: bool) -> None:
    track["members"].append(cur)
    track["ce"] = max(track["ce"], float(cur["e"]))
    if have_abs:
        track["ce_abs"] = max(track["ce_abs"], float(cur.get("e_abs", track["ce_abs"])))
    w = (float(cur["e_abs"]) - float(cur["s_abs"])) if have_abs else (float(cur["e"]) - float(cur["s"]))
    _vec_add_inplace(track["sum_vec"], v_next, w)
    track["sum_w"] += w


def _emit_track(
    track: Dict[str, Any],
    have_abs: bool,
    out_events: List[Dict[str, Any]],
    out_vecs: List[Dict[str, Any]],
    merge_map: Dict[str, List[str]],
) -> None:
    _emit_cluster(
        track["members"],
        track["cs"],
        track["ce"],
        track["cs_abs"],
        track["ce_abs"],
        track["rep_text"],
        track["sum_vec"],
        track["sum_w"],
        have_abs,
        out_events,
        out_vecs,
        merge_map,
    )







def merge_events(events_path: str, vlm_path: str, embeds_path: str, out_dir_root: str, cfg: dict) -> dict:
    stats = {
        "num_input_items": 0,
        "num_kept_items": 0,
        "num_clusters": 0,
        "num_merged_ops": 0,
        "skipped_short": 0,
        "skipped_no_embed": 0,
        "skipped_low_sem": 0,
        "skipped_large_gap": 0,
        "used_gap_abs": 0,
        "used_gap_norm": 0,
        "lane": None,
        "key": None,
        "source_hash": None,
        "mode": "micro",  
    }

    
    events_raw = _load_json(events_path)
    vlm_raw = _load_json(vlm_path)
    embeds_raw = _load_json(embeds_path)

    
    merge_cfg = (cfg or {}).get("merge", {})
    tau_sem = float(merge_cfg.get("tau_sem", 0.85))
    tau_gap_abs_sec = float(merge_cfg.get("gap_abs_sec", 0.30))
    gap_norm_fallback_default = merge_cfg.get("gap_norm_fallback", 0.01)
    gap_norm_fallback_default = None if gap_norm_fallback_default is None else float(gap_norm_fallback_default)
    min_duration_norm = float(merge_cfg.get("min_duration_norm", 0.0))

    lane, key = _infer_lane_and_key(events_path, out_dir_root)
    stats["lane"], stats["key"] = lane, key
    stats["source_hash"] = _build_source_hash(events_path, vlm_path, embeds_path, cfg)

    
    items: List[Dict[str, Any]]
    emb_map: Dict[str, List[float]]
    text_map: Dict[str, str]

    events = _to_event_list(events_raw)

    if _is_micro_embeds(embeds_raw):
        
        event_ids = [e["id"] for e in events]
        items, emb_map, text_map = _build_micro_items_from_embeds(embeds_raw, event_ids)
        stats["mode"] = "micro"
    else:
        
        text_map = _to_text_map_from_vlm(vlm_raw)
        event_ids = [e["id"] for e in events]
        emb_map = _to_embed_map_event_level(embeds_raw, event_ids)
        items = []
        for idx, e in enumerate(events):
            eid = e["id"]
            text = text_map.get(eid, "").strip()
            if not text:
                raise ValueError(f"Missing non-empty text for event id: {eid}")
            if eid not in emb_map or not emb_map[eid]:
                raise ValueError(f"Missing valid embedding for event id: {eid}")
            rec = {"id": eid, "s": e["s"], "e": e["e"]}
            if "s_abs" in e and "e_abs" in e:
                rec["s_abs"] = e["s_abs"]
                rec["e_abs"] = e["e_abs"]
            items.append(rec)
        _ensure_unique_ids([it["id"] for it in items], "event merge items")
        stats["mode"] = "event"

    stats["num_input_items"] = len(items)
    if len(items) == 0:
        _write_empty_outputs(out_dir_root, lane, key, cfg, stats)
        return stats

    
    kept: List[Dict[str, Any]] = []
    for it in items:
        if min_duration_norm > 0.0 and (it.get("e", 0.0) - it.get("s", 0.0)) < min_duration_norm:
            stats["skipped_short"] += 1
            continue
        kept.append(it)
    items = kept
    stats["num_kept_items"] = len(items)

    have_abs = all(("s_abs" in it and "e_abs" in it) for it in items)
    gap_norm_fallback = None if have_abs else gap_norm_fallback_default

    def it_text(iid: str) -> str:
        return text_map.get(iid, "").strip()

    def it_vec(iid: str) -> List[float]:
        v = emb_map.get(iid)
        if not isinstance(v, list) or not v:
            raise ValueError(f"Missing valid embedding for merge item id: {iid}")
        return _l2norm(v)

    
    merged_events: List[Dict[str, Any]] = []
    merged_vecs: List[Dict[str, Any]] = []
    merge_map: Dict[str, List[str]] = {}

    active_tracks: List[Dict[str, Any]] = [_start_track(items[0], have_abs, it_text, it_vec)]

    for k in range(1, len(items)):
        cur = items[k]
        v_next = it_vec(cur["id"])

        
        still_active: List[Dict[str, Any]] = []
        for tr in active_tracks:
            ok_gap, _ = _cluster_gap_ok(
                {"e_abs": tr["ce_abs"], "e": tr["ce"]},
                {"s_abs": float(cur.get("s_abs", 0.0)), "s": float(cur["s"])},
                tau_gap_abs_sec,
                gap_norm_fallback,
            )
            if have_abs:
                stats["used_gap_abs"] += 1
            else:
                stats["used_gap_norm"] += 1

            if not ok_gap:
                stats["skipped_large_gap"] += 1
                _emit_track(tr, have_abs, merged_events, merged_vecs, merge_map)
            else:
                still_active.append(tr)

        active_tracks = still_active

        
        
        
        best_idx: Optional[int] = None
        best_sim = -1.0
        best_gap = float("inf")

        for i, tr in enumerate(active_tracks):
            center = _l2norm(tr["sum_vec"]) if tr["sum_w"] > 0 else it_vec(tr["members"][-1]["id"])
            cos = _cosine(center, v_next)
            sim_hat = (cos + 1.0) * 0.5

            ok_gap, gap_used = _cluster_gap_ok(
                {"e_abs": tr["ce_abs"], "e": tr["ce"]},
                {"s_abs": float(cur.get("s_abs", 0.0)), "s": float(cur["s"])},
                tau_gap_abs_sec,
                gap_norm_fallback,
            )
            if have_abs:
                stats["used_gap_abs"] += 1
            else:
                stats["used_gap_norm"] += 1

            if (sim_hat >= tau_sem) and ok_gap:
                if (sim_hat > best_sim) or (sim_hat == best_sim and gap_used < best_gap):
                    best_idx = i
                    best_sim = sim_hat
                    best_gap = gap_used

        if best_idx is not None:
            _append_track(active_tracks[best_idx], cur, v_next, have_abs)
            stats["num_merged_ops"] += 1
        else:
            if len(active_tracks) > 0:
                stats["skipped_low_sem"] += 1
            active_tracks.append(_start_track(cur, have_abs, it_text, it_vec))

    
    for tr in active_tracks:
        _emit_track(tr, have_abs, merged_events, merged_vecs, merge_map)

    
    vec_by_id = {rec["id"]: rec for rec in merged_vecs}
    map_by_id = dict(merge_map)
    merged_events.sort(key=lambda ev: (float(ev["s"]), float(ev["e"]), ev["id"]))

    reindexed_vecs: List[Dict[str, Any]] = []
    reindexed_map: Dict[str, List[str]] = {}
    for idx, ev in enumerate(merged_events, start=1):
        old_id = ev["id"]
        mid = _make_merge_id(idx)
        ev["id"] = mid

        vec_rec = dict(vec_by_id[old_id])
        vec_rec["id"] = mid
        reindexed_vecs.append(vec_rec)

        reindexed_map[mid] = map_by_id[old_id]

    merged_vecs = reindexed_vecs
    merge_map = reindexed_map

    stats["num_clusters"] = len(merged_events)
    _write_outputs(out_dir_root, lane, key, merged_events, merged_vecs, merge_map, cfg, stats)
    return stats







def _emit_cluster(
    members: List[Dict[str, Any]],
    s: float,
    e: float,
    s_abs: float | None,
    e_abs: float | None,
    rep_text: str,
    sum_vec: List[float],
    sum_w: float,
    use_abs: bool,
    out_events: List[Dict[str, Any]],
    out_vecs: List[Dict[str, Any]],
    merge_map: Dict[str, List[str]],
):
    mid = f"tmp{len(out_events)+1:04d}"
    evt = {
        "id": mid,
        "s": s,
        "e": e,
        "dur": max(0.0, e - s),
        "members": [m["id"] for m in members],
        "text": rep_text,
    }
    if use_abs:
        evt.update({"s_abs": s_abs, "e_abs": e_abs, "dur_abs": max(0.0, (e_abs or 0.0) - (s_abs or 0.0))})
    out_events.append(evt)
    if sum_w > 0 and sum_vec:
        merged = _l2norm(sum_vec)
    else:
        merged = []
    out_vecs.append({"id": mid, "emb": merged})
    merge_map[mid] = [m["id"] for m in members]


def _validate_merged_outputs(events, vecs, merge_map) -> None:
    event_ids = []
    for idx, item in enumerate(events):
        if not isinstance(item, dict):
            raise ValueError(f"merged events[{idx}] must be an object")
        event_ids.append(_require_nonempty_str(item.get("id"), f"merged events[{idx}].id"))
    _ensure_unique_ids(event_ids, "merged events")

    vec_ids = []
    for idx, item in enumerate(vecs):
        if not isinstance(item, dict):
            raise ValueError(f"merged embeds[{idx}] must be an object")
        vid = _require_nonempty_str(item.get("id"), f"merged embeds[{idx}].id")
        emb = _to_embedding_list(item.get("emb") or item.get("embedding") or item.get("vec"), f"merged embeds[{idx}] ({vid}).emb")
        vec_ids.append(vid)
    _ensure_unique_ids(vec_ids, "merged embeds")

    if not isinstance(merge_map, dict):
        raise ValueError("merge_map must be a dict")
    map_ids = [_require_nonempty_str(mid, "merge_map.id") for mid in merge_map.keys()]
    _ensure_unique_ids(map_ids, "merge_map")

    event_id_set = set(event_ids)
    vec_id_set = set(vec_ids)
    map_id_set = set(map_ids)
    if event_id_set != vec_id_set or event_id_set != map_id_set:
        raise ValueError(
            "merged outputs id mismatch: "
            f"events={sorted(event_id_set)}, embeds={sorted(vec_id_set)}, merge_map={sorted(map_id_set)}"
        )


def _write_outputs(out_root: str, lane: str, key: str, events, vecs, merge_map, cfg: dict, stats: dict):
    _validate_merged_outputs(events, vecs, merge_map)
    p_events = os.path.join(out_root, "events_merged", lane, f"{key}.newevents.json")
    p_emb = os.path.join(out_root, "embeds", lane, f"{key}.emb.merged.json")
    p_map = os.path.join(out_root, "merge_map", lane, f"{key}.json")
    payload = {"events": events, "meta": _build_meta(cfg, stats)}
    _dump_json(payload, p_events)
    _dump_json(vecs, p_emb)
    _dump_json(merge_map, p_map)


def _write_empty_outputs(out_root: str, lane: str, key: str, cfg: dict, stats: dict):
    _write_outputs(out_root, lane, key, [], [], {}, cfg, stats)







def _load_cfg_from_yaml_or_none(path: str | None) -> dict:
    if not path:
        return {}
    try:
        import yaml  
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _merge_cli():
    ap = argparse.ArgumentParser("Event Merge (semantic + adjacency)")
    ap.add_argument("--events", required=True, help="outputs/event/cache/events/{ref|gen}/<id>.events.json")
    ap.add_argument("--vlm", required=True, help="outputs/event/cache/vlm/{ref|gen}/<id>.vlm.json")
    ap.add_argument("--embeds", required=True, help="outputs/event/cache/embeds/{ref|gen}/<id>.emb.json")
    ap.add_argument("--out-root", default="outputs/event/cache", help="event cache root, e.g. outputs/event/cache")
    ap.add_argument("--cfg", default="", help="configs/default.yaml (optional)")
    
    ap.add_argument("--tau-sem", type=float, default=None)
    ap.add_argument("--gap-abs-sec", type=float, default=None)
    ap.add_argument("--gap-norm-fallback", type=float, default=None)
    ap.add_argument("--min-duration-norm", type=float, default=None)
    args = ap.parse_args()

    cfg = _load_cfg_from_yaml_or_none(args.cfg)
    if "merge" not in cfg:
        cfg["merge"] = {}
    if args.tau_sem is not None:
        cfg["merge"]["tau_sem"] = float(args.tau_sem)
    if args.gap_abs_sec is not None:
        cfg["merge"]["gap_abs_sec"] = float(args.gap_abs_sec)
    if args.gap_norm_fallback is not None:
        cfg["merge"]["gap_norm_fallback"] = float(args.gap_norm_fallback)
    if args.min_duration_norm is not None:
        cfg["merge"]["min_duration_norm"] = float(args.min_duration_norm)

    stats = merge_events(args.events, args.vlm, args.embeds, args.out_root, cfg)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _merge_cli()
