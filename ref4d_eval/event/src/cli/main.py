
from __future__ import annotations
"""
End-to-end event evaluation entrypoint.

Intermediate cache:
  outputs/event/cache/

Final scores:
  outputs/event/scores/
"""
import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import shutil
import json
import csv
import numpy as np


LAUNCHER: List[str] = [sys.executable]







ROOT = Path(__file__).resolve().parents[2]      
PROJECT = ROOT.parents[1]                        
DATA = PROJECT / "outputs" / "event" / "cache"    
SCORES = PROJECT / "outputs" / "event" / "scores" 

DATA_META = PROJECT / "data" / "metadata" / "event_evidence"
REF_VIDEO_ROOT = PROJECT / "data" / "refvideo"
GEN_VIDEO_ROOT = PROJECT / "data" / "genvideo"
META_PATH = PROJECT / "data" / "metadata" / "ref4d_meta.jsonl"

PKG = "ref4d_eval.event"


def _resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT / p


def configure_paths(args) -> None:
    global DATA, SCORES, DATA_META, REF_VIDEO_ROOT, GEN_VIDEO_ROOT, META_PATH
    DATA = _resolve_project_path(getattr(args, "cache_root", DATA))
    SCORES = _resolve_project_path(getattr(args, "scores_root", SCORES))
    DATA_META = _resolve_project_path(getattr(args, "ref_event_root", DATA_META))
    REF_VIDEO_ROOT = _resolve_project_path(getattr(args, "ref_video_root", REF_VIDEO_ROOT))
    GEN_VIDEO_ROOT = _resolve_project_path(getattr(args, "gen_video_root", GEN_VIDEO_ROOT))
    META_PATH = _resolve_project_path(getattr(args, "meta_path", META_PATH))





def p_ref_video(topic: str, sample_id: str) -> Path:
    
    return REF_VIDEO_ROOT / f"{sample_id}.mp4"


def p_gen_video(model: str, topic: str, sample_id: str) -> Path:
    
    p_new = GEN_VIDEO_ROOT / model / f"{sample_id}.mp4"
    if p_new.exists():
        return p_new

    
    p_new_topic = GEN_VIDEO_ROOT / model / topic / f"{sample_id}.mp4"
    if p_new_topic.exists():
        return p_new_topic

    
    p_old = DATA / "genvideo" / model / topic / f"{sample_id}.mp4"
    if p_old.exists():
        return p_old

    
    return DATA / "genvideo" / model / f"{sample_id}.mp4"


def p_events_ref(sample_id: str) -> Path:
    return DATA / "events" / "ref" / f"{sample_id}.events.json"


def p_events_gen(pair_id: str) -> Path:
    return DATA / "events" / "gen" / f"{pair_id}.events.json"


def p_events_ref_merged(sample_id: str) -> Path:
    if DATA_META.name == "events_merged_ref":
        p_meta = DATA_META / f"{sample_id}.newevents.json"
    else:
        p_meta = DATA_META / "events_merged_ref" / f"{sample_id}.newevents.json"
    if p_meta.exists():
        return p_meta
    return DATA / "events_merged" / "ref" / f"{sample_id}.newevents.json"


def p_events_gen_merged(pair_id: str) -> Path:
    return DATA / "events_merged" / "gen" / f"{pair_id}.newevents.json"


def p_vlm_ref(sample_id: str) -> Path:
    return DATA / "vlm" / "ref" / f"{sample_id}.vlm.json"


def p_vlm_gen(pair_id: str) -> Path:
    return DATA / "vlm" / "gen" / f"{pair_id}.vlm.json"


def p_emb_ref(sample_id: str) -> Path:
    return DATA / "embeds" / "ref" / f"{sample_id}.emb.json"


def p_emb_gen(pair_id: str) -> Path:
    return DATA / "embeds" / "gen" / f"{pair_id}.emb.json"


def p_emb_ref_merged(sample_id: str) -> Path:
    if DATA_META.name == "embeds_merged_ref":
        p_meta = DATA_META / f"{sample_id}.emb.merged.json"
    elif DATA_META.name == "events_merged_ref":
        p_meta = DATA_META.parent / "embeds_merged_ref" / f"{sample_id}.emb.merged.json"
    else:
        p_meta = DATA_META / "embeds_merged_ref" / f"{sample_id}.emb.merged.json"
    if p_meta.exists():
        return p_meta
    return DATA / "embeds" / "ref" / f"{sample_id}.emb.merged.json"


def p_emb_gen_merged(pair_id: str) -> Path:
    return DATA / "embeds" / "gen" / f"{pair_id}.emb.merged.json"


def p_scene(video_id: str) -> Path:
    return DATA / "scenes" / f"{video_id}.scenes.json"


def p_match_dir(pair_id: str) -> Path:
    return DATA / "match" / pair_id


def p_gate(pair_id: str) -> Path:
    return p_match_dir(pair_id) / "gate_masks.npz"


def p_cost(pair_id: str) -> Path:
    return p_match_dir(pair_id) / "cost_matrix.npz"


def p_pairs(pair_id: str) -> Path:
    return p_match_dir(pair_id) / "pairs.json"


def p_scores(sample_id: str, model: str) -> Path:
    pair_id = pair_id_of(sample_id, model)
    return SCORES / model / pair_id / "event_scores.json"


def p_summary_csv() -> Path:
    return SCORES / "event_scores_summary.csv"


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def exists_and_nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def _read_json_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_merged_events_file(path: Path) -> None:
    data = _read_json_file(path)
    events = data.get("events") if isinstance(data, dict) and "events" in data else data
    if not isinstance(events, list):
        raise ValueError(f"invalid merged events file: {path}")
    seen = set()
    for idx, item in enumerate(events):
        if not isinstance(item, dict):
            raise ValueError(f"merged events[{idx}] must be a dict")
        eid = item.get("id") or item.get("eid") or item.get("event_id")
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"merged events[{idx}] missing non-empty id")
        if eid in seen:
            raise ValueError(f"duplicate merged event id: {eid}")
        seen.add(eid)
        if "s" not in item or "e" not in item:
            raise ValueError(f"merged events[{idx}] missing s/e")
        s = float(item["s"])
        e = float(item["e"])
        if e < s:
            raise ValueError(f"merged events[{idx}] invalid interval")


def _validate_merged_embeds_file(path: Path) -> None:
    data = _read_json_file(path)
    if not isinstance(data, list):
        raise ValueError(f"invalid merged embeds file: {path}")
    seen = set()
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"merged embeds[{idx}] must be a dict")
        eid = item.get("id") or item.get("eid") or item.get("event_id")
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"merged embeds[{idx}] missing non-empty id")
        if eid in seen:
            raise ValueError(f"duplicate merged embed id: {eid}")
        seen.add(eid)
        emb = item.get("emb") or item.get("embedding") or item.get("vec")
        if not isinstance(emb, list) or len(emb) == 0:
            raise ValueError(f"merged embeds[{idx}] has empty embedding")


def _validate_merged_pair(events_path: Path, embeds_path: Path) -> None:
    _validate_merged_events_file(events_path)
    _validate_merged_embeds_file(embeds_path)
    events_data = _read_json_file(events_path)
    events = events_data.get("events") if isinstance(events_data, dict) and "events" in events_data else events_data
    embeds = _read_json_file(embeds_path)
    event_ids = {str((it.get("id") or it.get("eid") or it.get("event_id") or "")).strip() for it in events}
    embed_ids = {str((it.get("id") or it.get("eid") or it.get("event_id") or "")).strip() for it in embeds}
    if event_ids != embed_ids:
        miss_emb = sorted(event_ids - embed_ids)
        extra_emb = sorted(embed_ids - event_ids)
        raise ValueError(
            f"merged events/embeds id mismatch: missing_emb={miss_emb[:10]} extra_emb={extra_emb[:10]}"
        )


def _validate_gate_npz(path: Path) -> None:
    with np.load(path, allow_pickle=True) as d:
        required = ("ref_ids", "gen_ids", "sim_sem", "r_tiou", "gate")
        for key in required:
            if key not in d:
                raise ValueError(f"gate npz missing key: {key}")
        sim = d["sim_sem"]
        rt = d["r_tiou"]
        gate = d["gate"]
        if sim.ndim != 2 or rt.ndim != 2 or gate.ndim != 2:
            raise ValueError("gate tensors must be 2D")
        if sim.shape != rt.shape or sim.shape != gate.shape:
            raise ValueError("gate tensor shapes mismatch")


def _validate_cost_npz(path: Path) -> None:
    with np.load(path, allow_pickle=True) as d:
        required = ("C", "Nr", "Ng", "w1", "w2", "null_ref", "null_gen")
        for key in required:
            if key not in d:
                raise ValueError(f"cost npz missing key: {key}")
        C = d["C"]
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError("cost matrix must be square")
        int(d["Nr"])
        int(d["Ng"])
        float(d["w1"])
        float(d["w2"])
        float(d["null_ref"])
        float(d["null_gen"])


def _validate_pairs_json(path: Path) -> None:
    data = _read_json_file(path)
    if not isinstance(data, dict):
        raise ValueError("pairs json must be a dict")
    if "M" not in data or not isinstance(data["M"], list):
        raise ValueError("pairs json missing list M")
    if "meta" not in data or not isinstance(data["meta"], dict):
        raise ValueError("pairs json missing dict meta")


def _is_valid_file(path: Path, validator) -> bool:
    if not exists_and_nonempty(path):
        return False
    try:
        validator(path)
        return True
    except Exception:
        return False


def _is_valid_pair(events_path: Path, embeds_path: Path) -> bool:
    if not (exists_and_nonempty(events_path) and exists_and_nonempty(embeds_path)):
        return False
    try:
        _validate_merged_pair(events_path, embeds_path)
        return True
    except Exception:
        return False


def has_shipped_ref_assets(sample_id: str) -> bool:
    return _is_valid_pair(p_events_ref_merged(sample_id), p_emb_ref_merged(sample_id))


def pair_id_of(sample_id: str, model: str) -> str:
    return f"{sample_id}__{model}"



def _count_ref_merged_events(sample_id: str) -> str:
    p = p_events_ref_merged(sample_id)
    try:
        if p.exists() and p.stat().st_size > 0:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("events"), list):
                return str(len(data["events"]))
            if isinstance(data, list):
                return str(len(data))
    except Exception as e:
        print(f"[summary] read merged events fail: {p} -> {e}")
    return ""





def run_module(launcher: List[str], module: str, args: List[str], env: Optional[Dict[str, str]] = None):
    cmd = list(launcher) + ["-m", module] + args
    print("[CMD]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")





def _norm_key(s: str) -> str:
    return "".join(ch for ch in s if ch.isalpha()).lower()


def _dig_score(v, allow_omitted: bool = False):
    
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        if allow_omitted and v.get("valid") is False:
            return None
        x = v.get("score")
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            return float(x)
    return None


def _extract_scores_from_json(scores_json: Path) -> Optional[Tuple[float, Optional[float], float, Optional[float], Optional[float]]]:
    try:
        with open(scores_json, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        print(f"[summary] read fail: {scores_json} -> {e}")
        return None

    if not isinstance(d, dict):
        print(f"[summary] invalid top-level schema: {scores_json}")
        return None

    for key in ("EGA", "ERel", "ECR"):
        if key not in d or not isinstance(d[key], dict):
            print(f"[summary] missing top-level key '{key}': {scores_json}")
            return None

    ega = _dig_score(d["EGA"])
    erel = _dig_score(d["ERel"], allow_omitted=True)
    ecr = _dig_score(d["ECR"])

    if not isinstance(ega, float) or not isinstance(ecr, float) or not (erel is None or isinstance(erel, float)):
        print(f"[summary] invalid atomic score schema: {scores_json}")
        return None

    event_score = None
    if "event_score" in d:
        x = d["event_score"]
        if x is None:
            event_score = None
        elif not isinstance(x, bool) and isinstance(x, (int, float)):
            event_score = float(x)
        else:
            print(f"[summary] invalid event_score schema: {scores_json}")
            return None

    event_score_0_100 = None
    if "event_score_0_100" in d:
        x = d["event_score_0_100"]
        if x is None:
            event_score_0_100 = None
        elif not isinstance(x, bool) and isinstance(x, (int, float)):
            event_score_0_100 = float(x)
        else:
            print(f"[summary] invalid event_score_0_100 schema: {scores_json}")
            return None

    return ega, erel, ecr, event_score, event_score_0_100


def _is_valid_scores(path: Path) -> bool:
    return exists_and_nonempty(path) and (_extract_scores_from_json(path) is not None)


def _write_summary_csv(pairs: List[Tuple[str, str]], out_csv: Optional[Path] = None) -> Tuple[int, int]:
    if out_csv is None:
        out_csv = p_summary_csv()
    ensure_dir(out_csv)

    
    merged: Dict[Tuple[str, str], Dict[str, str]] = {}
    if out_csv.exists() and out_csv.stat().st_size > 0:
        try:
            with open(out_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get("modelname", ""), row.get("sample_id", ""))
                    merged[key] = {
                        "modelname": row.get("modelname", ""),
                        "sample_id": row.get("sample_id", ""),
                        "EGA": row.get("EGA", ""),
                        "ERel": row.get("ERel", ""),
                        "ECR": row.get("ECR", ""),
                        "event_score": row.get("event_score", ""),
                        "event_score_0_100": row.get("event_score_0_100", ""),
                    }
        except Exception as e:
            print(f"[summary] read existing summary fail: {out_csv} -> {e}")

    
    new_count = 0
    for sample_id, model in pairs:
        sp = p_scores(sample_id, model)
        if not exists_and_nonempty(sp):
            print(f"[summary] skip (scores missing): {sp}")
            continue
        res = _extract_scores_from_json(sp)
        if res is None:
            print(f"[summary] skip (scores parse fail): {sp}")
            continue

        EGA, ERel, ECR, event_score, event_score_0_100 = res

        key = (model, sample_id)
        merged[key] = {
            "modelname": model,
            "sample_id": sample_id,
            "EGA": f"{EGA:.6f}",
            "ERel": (f"{ERel:.6f}" if isinstance(ERel, float) else ""),
            "ECR": f"{ECR:.6f}",
            "event_score": (f"{event_score:.6f}" if isinstance(event_score, float) else ""),
            "event_score_0_100": (f"{event_score_0_100:.6f}" if isinstance(event_score_0_100, float) else ""),
        }
        new_count += 1

    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["modelname", "sample_id", "EGA", "ERel", "ECR", "event_score", "event_score_0_100"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (_, _), row in sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            writer.writerow(row)

    print(f"[summary] merged {len(merged)} rows (+{new_count} new) -> {out_csv}")
    return len(merged), new_count





def step_detect(topic: str, sample_id: str, model: str,
                cfg_shot: str, cfg_gebd: str, force: bool):
    ref_vid = p_ref_video(topic, sample_id)
    gen_vid = p_gen_video(model, topic, sample_id)

    
    if not gen_vid.exists():
        raise FileNotFoundError(f"gen video missing: {gen_vid}")

    ref_scene = p_scene(sample_id)
    gen_scene = p_scene(pair_id_of(sample_id, model))
    ref_evt = p_events_ref(sample_id)
    gen_evt = p_events_gen(pair_id_of(sample_id, model))

    L_trans = LAUNCHER
    L_ddm = LAUNCHER

    ref_ready = has_shipped_ref_assets(sample_id)

    
    if not force and ref_ready:
        print(f"[detect] skip ref side: released merged ref assets already available for {sample_id}")
    elif ref_vid.exists():
        if force or not exists_and_nonempty(ref_scene):
            ensure_dir(ref_scene)
            run_module(
                L_trans,
                f"{PKG}.src.eventdetect.transnetv2_runner",
                ["--video", str(ref_vid), "--out", str(ref_scene), "--config", str(cfg_shot)],
            )
        else:
            print(f"[detect] skip scenes (ref): {ref_scene.name}")

        if force or not exists_and_nonempty(ref_evt):
            ensure_dir(ref_evt)
            run_module(
                L_ddm,
                f"{PKG}.src.eventdetect.ddm_runner",
                ["--video", str(ref_vid), "--out", str(ref_evt),
                 "--config", str(cfg_gebd), "--scenes", str(ref_scene)],
            )
        else:
            print(f"[detect] skip events (ref): {ref_evt.name}")
    else:
        print(f"[detect] no raw ref video, skip ref side: {ref_vid}")

    
    if force or not exists_and_nonempty(gen_scene):
        ensure_dir(gen_scene)
        run_module(
            L_trans,
            f"{PKG}.src.eventdetect.transnetv2_runner",
            ["--video", str(gen_vid), "--out", str(gen_scene), "--config", str(cfg_shot)],
        )
    else:
        print(f"[detect] skip scenes (gen): {gen_scene.name}")

    if force or not exists_and_nonempty(gen_evt):
        ensure_dir(gen_evt)
        run_module(
            L_ddm,
            f"{PKG}.src.eventdetect.ddm_runner",
            ["--video", str(gen_vid), "--out", str(gen_evt),
             "--config", str(cfg_gebd), "--scenes", str(gen_scene)],
        )
    else:
        print(f"[detect] skip events (gen): {gen_evt.name}")





def step_vlm(topic: str, sample_id: str, model: str, cfg_vlm: str, force: bool):
    ref_vid = p_ref_video(topic, sample_id)
    gen_vid = p_gen_video(model, topic, sample_id)
    ref_evt = p_events_ref(sample_id)
    gen_evt = p_events_gen(pair_id_of(sample_id, model))
    ref_vlm = p_vlm_ref(sample_id)
    gen_vlm = p_vlm_gen(pair_id_of(sample_id, model))

    L_vlm = LAUNCHER
    ref_ready = has_shipped_ref_assets(sample_id)

    
    if not force and ref_ready:
        print(f"[vlm] skip ref side: released merged ref assets already available for {sample_id}")
    elif ref_vid.exists() and ref_evt.exists():
        if force or not exists_and_nonempty(ref_vlm):
            ensure_dir(ref_vlm)
            run_module(
                L_vlm,
                f"{PKG}.src.vlm.vllama3_infer",
                ["--video", str(ref_vid), "--events", str(ref_evt),
                 "--config", str(cfg_vlm), "--out", str(ref_vlm)],
            )
        else:
            print(f"[vlm] skip (ref): {ref_vlm.name}")
    else:
        print(f"[vlm] no raw ref video/events, skip ref side: {ref_vid}")

    
    if force or not exists_and_nonempty(gen_vlm):
        ensure_dir(gen_vlm)
        run_module(
            L_vlm,
            f"{PKG}.src.vlm.vllama3_infer",
            ["--video", str(gen_vid), "--events", str(gen_evt),
             "--config", str(cfg_vlm), "--out", str(gen_vlm)],
        )
    else:
        print(f"[vlm] skip (gen): {gen_vlm.name}")





def step_embed(sample_id: str, model: str, cfg_embed: str, force: bool):
    ref_vlm = p_vlm_ref(sample_id)
    gen_vlm = p_vlm_gen(pair_id_of(sample_id, model))
    ref_emb = p_emb_ref(sample_id)
    gen_emb = p_emb_gen(pair_id_of(sample_id, model))

    L_emb = LAUNCHER

    
    
    
    merged_ref_emb = p_emb_ref_merged(sample_id)
    if not (not force and _is_valid_file(merged_ref_emb, _validate_merged_embeds_file)):
        if ref_vlm.exists():
            if force or not exists_and_nonempty(ref_emb):
                ensure_dir(ref_emb)
                run_module(
                    L_emb,
                    f"{PKG}.src.embed.e5_encoder",
                    ["--vlm", str(ref_vlm), "--config", str(cfg_embed), "--out", str(ref_emb)],
                )
            else:
                print(f"[embed] skip (ref): {ref_emb.name}")
        else:
            print("[embed] skip (ref): no ref VLM file, assuming merged ref embedding is provided offline if needed")
    else:
        print(f"[embed] skip (ref): merged ref embedding already provided ({merged_ref_emb})")

    
    if force or not exists_and_nonempty(gen_emb):
        ensure_dir(gen_emb)
        run_module(
            L_emb,
            f"{PKG}.src.embed.e5_encoder",
            ["--vlm", str(gen_vlm), "--config", str(cfg_embed), "--out", str(gen_emb)],
        )
    else:
        print(f"[embed] skip (gen): {gen_emb.name}")





def step_merge(sample_id: str, model: str, cfg_default: str, force: bool):
    pair = pair_id_of(sample_id, model)
    ref_evt = p_events_ref(sample_id)
    gen_evt = p_events_gen(pair)
    ref_vlm = p_vlm_ref(sample_id)
    gen_vlm = p_vlm_gen(pair)
    ref_emb = p_emb_ref(sample_id)
    gen_emb = p_emb_gen(pair)
    ref_evt_m = p_events_ref_merged(sample_id)
    gen_evt_m = p_events_gen_merged(pair)
    ref_emb_m = p_emb_ref_merged(sample_id)
    gen_emb_m = p_emb_gen_merged(pair)

    L_eval = LAUNCHER

    
    ref_ready = _is_valid_pair(ref_evt_m, ref_emb_m)
    if not force and ref_ready:
        print(f"[merge] skip (ref): merged ref evidence already provided ({ref_evt_m.name}, {ref_emb_m.name})")
    else:
        if ref_evt.exists() and ref_vlm.exists() and ref_emb.exists():
            ensure_dir(ref_emb_m)
            run_module(
                L_eval,
                f"{PKG}.src.merge.merger",
                ["--events", str(ref_evt), "--vlm", str(ref_vlm), "--embeds", str(ref_emb),
                 "--out-root", str(DATA), "--cfg", str(cfg_default)],
            )
        else:
            print("[merge] skip (ref): missing ref events / vlm / embeds, assume merged ref evidence is precomputed")

    
    gen_ready = _is_valid_pair(gen_evt_m, gen_emb_m)
    if not force and gen_ready:
        print(f"[merge] skip (gen): merged gen evidence already exists ({gen_evt_m.name}, {gen_emb_m.name})")
    else:
        ensure_dir(gen_emb_m)
        run_module(
            L_eval,
            f"{PKG}.src.merge.merger",
            ["--events", str(gen_evt), "--vlm", str(gen_vlm), "--embeds", str(gen_emb),
             "--out-root", str(DATA), "--cfg", str(cfg_default)],
        )





def _prefer_merged(path_merged: Path, path_orig: Path, validator) -> Path:
    return path_merged if _is_valid_file(path_merged, validator) else path_orig


def step_match(sample_id: str, model: str, cfg_default: str, force: bool):
    pair = pair_id_of(sample_id, model)
    gate_p = p_gate(pair)
    cost_p = p_cost(pair)
    pairs_p = p_pairs(pair)

    ref_emb = _prefer_merged(p_emb_ref_merged(sample_id), p_emb_ref(sample_id), _validate_merged_embeds_file)
    gen_emb = _prefer_merged(p_emb_gen_merged(pair), p_emb_gen(pair), _validate_merged_embeds_file)
    ref_evt_m = p_events_ref_merged(sample_id)
    gen_evt_m = p_events_gen_merged(pair)

    L_eval = LAUNCHER

    
    if force or not _is_valid_file(gate_p, _validate_gate_npz):
        ensure_dir(gate_p)
        run_module(
            L_eval,
            f"{PKG}.src.match.gating",
            ["--ref-events", str(ref_evt_m),
             "--ref-embeds", str(ref_emb),
             "--gen-events", str(gen_evt_m),
             "--gen-embeds", str(gen_emb),
             "--config", str(cfg_default),
             "--out", str(gate_p)],
        )
    else:
        print(f"[gate] skip: {gate_p.name}")

    
    need_cost = force or not _is_valid_file(cost_p, _validate_cost_npz)
    if need_cost:
        ensure_dir(cost_p)
        run_module(
            L_eval,
            f"{PKG}.src.match.costs",
            ["--gate", str(gate_p), "--config", str(cfg_default), "--out", str(cost_p)],
        )
        if not exists_and_nonempty(cost_p):
            print("[match] cost_matrix.npz was not generated; retrying gate and cost ...")
            try:
                if gate_p.exists():
                    gate_p.unlink()
            except Exception:
                pass
            ensure_dir(gate_p)
            run_module(
                L_eval,
                f"{PKG}.src.match.gating",
                ["--ref-events", str(ref_evt_m),
                 "--ref-embeds", str(ref_emb),
                 "--gen-events", str(gen_evt_m),
                 "--gen-embeds", str(gen_emb),
                 "--config", str(cfg_default),
                 "--out", str(gate_p)],
            )
            ensure_dir(cost_p)
            run_module(
                L_eval,
                f"{PKG}.src.match.costs",
                ["--gate", str(gate_p), "--config", str(cfg_default), "--out", str(cost_p)],
            )
            if not exists_and_nonempty(cost_p):
                raise RuntimeError("cost_matrix.npz is still missing; check the costs.py output path or gate_masks.npz contents.")
    else:
        print(f"[cost] skip: {cost_p.name}")

    
    if force or not _is_valid_file(pairs_p, _validate_pairs_json):
        ensure_dir(pairs_p)
        run_module(
            L_eval,
            f"{PKG}.src.match.hungarian",
            ["--cost", str(cost_p), "--gate", str(gate_p), "--out", str(pairs_p)],
        )
    else:
        print(f"[hung] skip: {pairs_p.name}")





def step_metrics(sample_id: str, model: str, cfg_default: str, force: bool):
    pair = pair_id_of(sample_id, model)
    pairs_p = p_pairs(pair)
    scores_p = p_scores(sample_id, model)

    ref_evt_m = p_events_ref_merged(sample_id)
    gen_evt_m = p_events_gen_merged(pair)

    L_eval = LAUNCHER

    if force or not _is_valid_scores(scores_p):
        ensure_dir(scores_p)
        run_module(
            L_eval,
            f"{PKG}.src.metrics.aggregate",
            ["--ref_events", str(ref_evt_m), "--gen_events", str(gen_evt_m),
             "--pairs", str(pairs_p), "--config", str(cfg_default), "--out", str(scores_p)],
        )
    else:
        print(f"[metrics] skip: {scores_p.name}")





def run_single(topic: str, sample_id: str, model: str,
               steps: List[str],
               cfg_default: str, cfg_vlm: str, cfg_embed: str,
               cfg_shot: Optional[str], cfg_gebd: Optional[str],
               force: bool):
    steps = [s.strip().lower() for s in steps]
    if "detect" in steps:
        if not cfg_shot or not cfg_gebd:
            raise ValueError("detect requires both --cfg-shot and --cfg-gebd")
        step_detect(topic, sample_id, model, cfg_shot, cfg_gebd, force)
    if "vlm" in steps:
        step_vlm(topic, sample_id, model, cfg_vlm, force)
    if "embed" in steps:
        step_embed(sample_id, model, cfg_embed, force)
    if "merge" in steps:
        step_merge(sample_id, model, cfg_default, force)
    if any(s in steps for s in ("gate", "cost", "hung", "match")):
        step_match(sample_id, model, cfg_default, force)
    if "metrics" in steps or "score" in steps:
        step_metrics(sample_id, model, cfg_default, force)

    if os.environ.get("EVENT_SKIP_SUMMARY", "0") != "1":
        _write_summary_csv([(sample_id, model)])





def discover_samples(topics: List[str]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []

    meta_path = META_PATH
    if meta_path.exists():
        seen: Dict[str, int] = {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    topic = obj.get("topic") or obj.get("theme") or ""
                    sid = obj.get("sample_id") or obj.get("id")
                    if not sid:
                        continue
                    sid = str(sid)
                    if topics and topic not in topics:
                        continue
                    if sid in seen:
                        raise ValueError(
                            f"duplicate sample_id={sid} in metadata {meta_path}:{line_no}; "
                            f"first seen at line {seen[sid]}"
                        )
                    seen[sid] = line_no
                    items.append((topic, sid))
        except Exception as e:
            raise RuntimeError(f"[discover] read meta fail: {meta_path} -> {e}") from e
        return items

    
    ref_dir = REF_VIDEO_ROOT
    if ref_dir.exists():
        for mp4 in sorted(ref_dir.glob("*.mp4")):
            items.append(("", mp4.stem))
    return items


def discover_models(models: List[str]) -> List[str]:
    if models:
        return models
    root = GEN_VIDEO_ROOT
    found: List[str] = []
    if root.exists():
        for path in sorted(root.iterdir()):
            if path.is_dir() and not path.name.startswith("."):
                found.append(path.name)
    return found


def _gen_root_label() -> str:
    return f"{GEN_VIDEO_ROOT}/<model>/"


def batch_run(topics: List[str], models: List[str],
              steps: List[str],
              cfg_default: str, cfg_vlm: str, cfg_embed: str,
              cfg_shot: Optional[str], cfg_gebd: Optional[str],
              force: bool,
              limit: int = 0):
    samples = discover_samples(topics)
    models = discover_models(models)
    print(f"[batch] found {len(samples)} ref samples")
    print(f"[batch] found {len(models)} models")
    if not samples:
        raise ValueError("No samples discovered from data/metadata/ref4d_meta.jsonl or data/refvideo/*.mp4")
    if not models:
        raise ValueError(f"No models specified and none discovered under {_gen_root_label()}")
    pairs_for_summary: List[Tuple[str, str]] = []
    ran = 0
    missing_gen_count = 0

    for topic, sample_id in samples:
        for model in models:
            if not p_gen_video(model, topic, sample_id).exists():
                missing_gen_count += 1
                continue
            if limit > 0 and ran >= limit:
                print(f"[batch] reached LIMIT={limit}; stop scheduling new pairs")
                merged_count, _ = _write_summary_csv(pairs_for_summary)
                if missing_gen_count:
                    print(f"[batch] skipped {missing_gen_count} missing generated videos")
                if merged_count <= 0:
                    raise RuntimeError("No event score rows were written; check generated videos, cache roots, and MODELS")
                return
            print(f"\n=== RUN: topic={topic} sample={sample_id} model={model} ===")
            pairs_for_summary.append((sample_id, model))
            ran += 1
            try:
                run_single(
                    topic, sample_id, model, steps,
                    cfg_default, cfg_vlm, cfg_embed,
                    cfg_shot, cfg_gebd,
                    force,
                )
            except Exception as e:
                print(f"[ERROR] topic={topic} sample={sample_id} model={model} -> {e}")
                continue

    if missing_gen_count:
        print(f"[batch] skipped {missing_gen_count} missing generated videos")
    merged_count, _ = _write_summary_csv(pairs_for_summary)
    if ran <= 0:
        raise RuntimeError("No event pairs ran; check GEN_VIDEO_ROOT, MODELS, META_PATH, and sample_id filenames")
    if merged_count <= 0:
        raise RuntimeError("No event score rows were written; check generated videos, cache roots, and failed samples above")





def parse_args():
    ap = argparse.ArgumentParser(description="event_eval end-to-end pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    
    p1 = sub.add_parser("run", help="run a single pair")
    p1.add_argument("--topic", required=True)
    p1.add_argument("--sample-id", required=True)
    p1.add_argument("--model", required=True)
    p1.add_argument(
        "--steps",
        default="detect,vlm,embed,merge,match,metrics",
        help="Comma-separated steps: detect,vlm,embed,merge,gate,cost,hung,match,metrics",
    )
    p1.add_argument("--cfg-default", required=False, help="Shared event config; required by merge/match/metrics")
    p1.add_argument("--cfg-vlm", required=False, help="VLM config; required by vlm")
    p1.add_argument("--cfg-embed", required=False, help="Embedding config; required by embed")
    p1.add_argument("--cfg-shot", required=False, help="Shot detection config; required by detect")
    p1.add_argument("--cfg-gebd", required=False, help="GEBD config; required by detect")
    p1.add_argument("--ref-video-root", default="data/refvideo")
    p1.add_argument("--gen-video-root", default="data/genvideo")
    p1.add_argument("--ref-event-root", default="data/metadata/event_evidence")
    p1.add_argument("--cache-root", default="outputs/event/cache")
    p1.add_argument("--scores-root", default="outputs/event/scores")
    p1.add_argument("--meta-path", default="data/metadata/ref4d_meta.jsonl")
    p1.add_argument("--force", action="store_true")

    
    p2 = sub.add_parser("batch", help="batch over topics & models")
    p2.add_argument("--topics", default="", help="Optional legacy topic filter, e.g. people_daily,news_v1")
    p2.add_argument("--models", default="", help="Optional model list; defaults to discovering model directories under --gen-video-root")
    p2.add_argument("--steps", default="detect,vlm,embed,merge,match,metrics")
    p2.add_argument("--cfg-default", required=False, help="Shared event config; required by merge/match/metrics")
    p2.add_argument("--cfg-vlm", required=False, help="VLM config; required by vlm")
    p2.add_argument("--cfg-embed", required=False, help="Embedding config; required by embed")
    p2.add_argument("--cfg-shot", required=False, help="Shot detection config; required by detect")
    p2.add_argument("--cfg-gebd", required=False, help="GEBD config; required by detect")
    p2.add_argument("--ref-video-root", default="data/refvideo")
    p2.add_argument("--gen-video-root", default="data/genvideo")
    p2.add_argument("--ref-event-root", default="data/metadata/event_evidence")
    p2.add_argument("--cache-root", default="outputs/event/cache")
    p2.add_argument("--scores-root", default="outputs/event/scores")
    p2.add_argument("--meta-path", default="data/metadata/ref4d_meta.jsonl")
    p2.add_argument("--limit", type=int, default=0, help="Maximum number of discovered model/sample pairs to run; 0 means all")
    p2.add_argument("--force", action="store_true")
    return ap.parse_args()


def _validate_step_requirements(steps: List[str], args) -> None:
    sset = {s.strip().lower() for s in steps if s.strip()}

    def _need(name: str, value: Optional[str]):
        if not value:
            raise ValueError(f"steps={sorted(sset)} require --{name}")

    if "detect" in sset:
        _need("cfg-shot", getattr(args, "cfg_shot", None))
        _need("cfg-gebd", getattr(args, "cfg_gebd", None))

    if "vlm" in sset:
        _need("cfg-vlm", getattr(args, "cfg_vlm", None))

    if "embed" in sset:
        _need("cfg-embed", getattr(args, "cfg_embed", None))

    if any(s in sset for s in ("merge", "gate", "cost", "hung", "match", "metrics", "score")):
        _need("cfg-default", getattr(args, "cfg_default", None))


def main():
    args = parse_args()
    configure_paths(args)
    if args.cmd == "run":
        steps = [s for s in args.steps.split(",") if s.strip()]
        _validate_step_requirements(steps, args)
        run_single(
            args.topic, args.sample_id, args.model,
            steps, args.cfg_default, args.cfg_vlm, args.cfg_embed,
            args.cfg_shot, args.cfg_gebd,
            args.force,
        )
    else:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        steps = [s for s in args.steps.split(",") if s.strip()]
        _validate_step_requirements(steps, args)
        batch_run(
            topics, models, steps,
            args.cfg_default, args.cfg_vlm, args.cfg_embed,
            args.cfg_shot, args.cfg_gebd,
            args.force,
            args.limit,
        )


if __name__ == "__main__":
    main()
