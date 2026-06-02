
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def _import_build_api():
    try:
        from ref4d_eval.semantic.softalign.api import build_api as _build_api
        return _build_api
    except Exception:
        pass

    this_file = Path(__file__).resolve()
    repo_root = None
    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "ref4d_eval").is_dir():
            repo_root = parent
            break
    if repo_root is None:
        raise ImportError(f"Unable to infer repo root from {this_file}")

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from ref4d_eval.semantic.softalign.api import build_api as _build_api
    return _build_api

def _child_dirs_with_json(root: Path) -> list[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir() and any(p.rglob("*.json"))]

def _looks_like_multi_model_parent(root: Path) -> bool:
    child_dirs = _child_dirs_with_json(root)
    if len(child_dirs) < 2:
        return False

    seen: dict[str, Path] = {}
    for sub in child_dirs:
        for p in sub.rglob("*.json"):
            sid = p.stem
            prev = seen.get(sid)
            if prev is not None and prev != sub:
                return True
            seen[sid] = sub
    return False

def iter_models(gen_root: Path, multi_model: bool):
    if not multi_model:
        if not any(gen_root.rglob("*.json")):
            raise FileNotFoundError(
                f"[single-model mode] No *.json files found under {gen_root}. For multi-model mode, add --multi-model and point --gen-dir to the parent directory containing model subdirectories."
            )
        if _looks_like_multi_model_parent(gen_root):
            raise FileNotFoundError(
                f"[single-model mode] {gen_root} looks like a multi-model parent directory: duplicate sample_id values were detected across direct subdirectories."
                f" Add --multi-model and point --gen-dir to that parent directory."
            )
        yield gen_root.name, gen_root
        return

    any_found = False
    for sub in _child_dirs_with_json(gen_root):
        any_found = True
        yield sub.name, sub
    if not any_found:
        raise FileNotFoundError(f"[multi-model mode] No *.json files were found in subdirectories of {gen_root}.")

def build_json_index(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in sorted(root.rglob("*.json")):
        sid = p.stem
        if sid in index:
            raise RuntimeError(f"Duplicate sample_id '{sid}' under {root}: {index[sid]} vs {p}")
        index[sid] = p
    return index

def _safe_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def _extract_semantic_scores_from_report(report):
    info = getattr(report, "info", {}) or {}
    semantic_final = info.get("semantic_final", {}) if isinstance(info, dict) else {}

    semantic_score = _safe_float(semantic_final.get("raw"))
    if semantic_score is None:
        semantic_score = _safe_float(getattr(report, "s_base", None))

    semantic_score_0_100 = _safe_float(semantic_final.get("cdf_0_100"))
    if semantic_score_0_100 is None and semantic_score is not None:
        semantic_score_0_100 = 100.0 * 0.5 * (1.0 + math.erf(float(semantic_score) / math.sqrt(2.0)))

    return semantic_score, semantic_score_0_100

def is_valid_score(report) -> bool:
    vals = [
        _safe_float(getattr(report.axis, "catcov", None)),
        _safe_float(getattr(report.axis, "aic", None)),
        _safe_float(getattr(report.axis, "hal", None)),
    ]
    semantic_score, semantic_score_0_100 = _extract_semantic_scores_from_report(report)
    vals.extend([semantic_score, semantic_score_0_100])
    return all(v is not None for v in vals)

def _stat_mtime_ns(path: Path) -> Optional[int]:
    try:
        return int(path.stat().st_mtime_ns)
    except Exception:
        return None

def _compute_code_signature() -> Dict[str, Any]:
    softalign_dir = Path(__file__).resolve().parent
    py_files = sorted(softalign_dir.glob("*.py"))
    hasher = hashlib.sha256()
    rel_files = []
    for p in py_files:
        rel_files.append(p.name)
        hasher.update(p.name.encode("utf-8"))
        hasher.update(b"\0")
        with p.open("rb") as f:
            hasher.update(f.read())
        hasher.update(b"\0")
    return {
        "scope": str(softalign_dir),
        "files": rel_files,
        "sha256": hasher.hexdigest(),
    }

def _cache_meta(
    *,
    sample_id: str,
    ref_path: Path,
    gen_path: Path,
    yaml_path: Path,
    code_sig: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "ref_path": str(ref_path.resolve()),
        "gen_path": str(gen_path.resolve()),
        "yaml_path": str(yaml_path.resolve()),
        "ref_mtime_ns": _stat_mtime_ns(ref_path),
        "gen_mtime_ns": _stat_mtime_ns(gen_path),
        "yaml_mtime_ns": _stat_mtime_ns(yaml_path),
        "softalign_code_sha256": code_sig.get("sha256"),
        "softalign_code_files": code_sig.get("files", []),
    }

def _same_cache_meta(existing: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    keys = (
        "sample_id",
        "ref_path",
        "gen_path",
        "yaml_path",
        "ref_mtime_ns",
        "gen_mtime_ns",
        "yaml_mtime_ns",
        "softalign_code_sha256",
    )
    for k in keys:
        if existing.get(k) != expected.get(k):
            return False
    return True

def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        obj = asdict(obj)

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [_to_jsonable(v) for v in sorted(obj, key=lambda x: str(x))]
    return obj

def _report_payload(report, *, ref_path: Path, gen_path: Path, yaml_path: Path, code_sig: Dict[str, Any]) -> Dict[str, Any]:
    semantic_score, semantic_score_0_100 = _extract_semantic_scores_from_report(report)
    valid_score = is_valid_score(report)
    return {
        "sample_id": report.sample_id,
        "valid_score": bool(valid_score),
        "cache_meta": _cache_meta(
            sample_id=report.sample_id,
            ref_path=ref_path,
            gen_path=gen_path,
            yaml_path=yaml_path,
            code_sig=code_sig,
        ),
        "axis": {
            "catcov": _safe_float(getattr(report.axis, "catcov", None)),
            "aic": _safe_float(getattr(report.axis, "aic", None)),
            "hal": _safe_float(getattr(report.axis, "hal", None)),
        },
        "semantic_score": semantic_score,
        "semantic_score_0_100": semantic_score_0_100,
        "sizes": _to_jsonable(getattr(report, "sizes", {})),
        "details": _to_jsonable(getattr(report, "details", {})),
        "info": _to_jsonable(getattr(report, "info", {})),
    }

def load_existing_report(
    report_path: Path,
    *,
    sample_id: str,
    ref_path: Path,
    gen_path: Path,
    yaml_path: Path,
    code_sig: Dict[str, Any],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not report_path.exists():
        return "none", None

    try:
        with report_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "none", None

    expected_meta = _cache_meta(
        sample_id=sample_id,
        ref_path=ref_path,
        gen_path=gen_path,
        yaml_path=yaml_path,
        code_sig=code_sig,
    )

    if data.get("sample_id") != sample_id:
        return "stale", data

    existing_meta = data.get("cache_meta")
    if not isinstance(existing_meta, dict):
        return "stale", data

    if not _same_cache_meta(existing_meta, expected_meta):
        return "stale", data

    valid_flag = data.get("valid_score", None)
    if valid_flag is False:
        return "invalid_final", data

    axis = data.get("axis", {}) if isinstance(data.get("axis"), dict) else {}
    catcov = _safe_float(axis.get("catcov"))
    aic = _safe_float(axis.get("aic"))
    hal = _safe_float(axis.get("hal"))
    semantic_score = _safe_float(data.get("semantic_score"))
    semantic_score_0_100 = _safe_float(data.get("semantic_score_0_100"))

    if all(v is not None for v in (catcov, aic, hal, semantic_score, semantic_score_0_100)):
        return "valid", data

    return "stale", data

def write_csv_rows(
    _unused_pcsv,
    gcsv_path: Path,
    model_name: str,
    sample_id: str,
    catcov: float,
    aic: float,
    hal: float,
    semantic_score: float,
    semantic_score_0_100: float,
):
    row_vals = [
        f"{catcov:.6f}",
        f"{aic:.6f}",
        f"{hal:.6f}",
        f"{semantic_score:.6f}",
        f"{semantic_score_0_100:.6f}",
    ]
    with open(gcsv_path, "a", encoding="utf-8") as gcsv:
        gcsv.write(",".join([model_name, sample_id] + row_vals) + "\n")

def _unlink_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


class _NullCsv:
    def write(self, _text: str) -> None:
        return None

    def __enter__(self) -> "_NullCsv":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def main():
    parser = argparse.ArgumentParser(description="SoftAlign batch scoring with cache reuse and NaN filtering")
    parser.add_argument(
        "--yaml",
        type=str,
        required=True,
        help="Path to the softalign.yaml config file",
    )
    parser.add_argument(
        "--ref-dir",
        type=str,
        required=True,
        help="Reference JSON directory",
    )
    parser.add_argument(
        "--gen-dir",
        type=str,
        required=True,
        help="Generated JSON directory for single-model mode, or its parent directory for multi-model mode",
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Treat --gen-dir as a parent directory containing multiple model subdirectories",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="scores_out",
        help="Output root directory; model-specific subdirectories will be created under it",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N samples for debugging; 0 means no limit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation, ignoring existing reports and overwriting outputs",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    ref_dir = Path(args.ref_dir)
    gen_dir = Path(args.gen_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not yaml_path.exists():
        print(f"[ERR] Config file does not exist: {yaml_path}", file=sys.stderr)
        sys.exit(1)
    if not ref_dir.exists():
        print(f"[ERR] Reference directory does not exist: {ref_dir}", file=sys.stderr)
        sys.exit(1)
    if not gen_dir.exists():
        print(f"[ERR] Generated directory does not exist: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    ref_list = sorted(ref_dir.glob("*.json"))
    if not ref_list:
        print(f"[ERR] Reference directory contains no *.json files: {ref_dir}", file=sys.stderr)
        sys.exit(1)

    model_entries = list(iter_models(gen_dir, args.multi_model))
    if not model_entries:
        print(f"[ERR] Generated directory has no usable model directories: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    build_api = _import_build_api()
    api = build_api(str(yaml_path))
    code_sig = _compute_code_signature()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    global_csv = out_root / f"scores_all_models_{ts}.csv"
    with open(global_csv, "w", encoding="utf-8") as gcsv:
        gcsv.write(",".join(["modelname", "sample_id", "catcov", "aic", "hal", "semantic_score", "semantic_score_0_100"]) + "\n")

    for model_name, model_path in model_entries:
        print(f"\n=== Model: {model_name} ===")
        out_dir = out_root / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = out_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        gen_index = build_json_index(model_path)
        model_ref_list = [p for p in ref_list if p.stem in gen_index]
        if args.limit > 0:
            model_ref_list = model_ref_list[: args.limit]

        with _NullCsv() as pcsv:
            iterable: Iterable[Path] = model_ref_list
            if tqdm is not None:
                iterable = tqdm(model_ref_list, desc=f"{model_name}", ncols=100)

            stats = {
                "scored": 0,
                "valid": 0,
                "reused": 0,
                "reused_invalid": 0,
                "skipped": 0,
                "missing": 0,
                "error": 0,
                "written": 0,
            }

            for ref_path in iterable:
                name = ref_path.name
                sample_id = ref_path.stem
                gen_path = gen_index.get(sample_id)
                report_path = reports_dir / name
                error_path = reports_dir / f"{name}.error.json"
                missing_path = reports_dir / f"{name}.missing.json"

                if gen_path is None or (not gen_path.exists()):
                    stats["missing"] += 1
                    if args.force or not missing_path.exists():
                        with open(missing_path, "w", encoding="utf-8") as fw:
                            json.dump(
                                {
                                    "error": "generated json missing",
                                    "sample_id": sample_id,
                                    "ref": str(ref_path),
                                    "gen": None if gen_path is None else str(gen_path),
                                },
                                fw,
                                ensure_ascii=False,
                                indent=2,
                            )
                    continue

                if not args.force:
                    status, payload = load_existing_report(
                        report_path,
                        sample_id=sample_id,
                        ref_path=ref_path,
                        gen_path=gen_path,
                        yaml_path=yaml_path,
                        code_sig=code_sig,
                    )
                    if status == "valid":
                        stats["reused"] += 1
                        try:
                            axis = payload["axis"]
                            catcov = float(axis["catcov"])
                            aic = float(axis["aic"])
                            hal = float(axis["hal"])
                            semantic_score = float(payload["semantic_score"])
                            semantic_score_0_100 = float(payload["semantic_score_0_100"])
                            write_csv_rows(
                                pcsv,
                                global_csv,
                                model_name,
                                sample_id,
                                catcov,
                                aic,
                                hal,
                                semantic_score,
                                semantic_score_0_100,
                            )
                            stats["written"] += 1
                            stats["valid"] += 1
                            _unlink_if_exists(error_path)
                            _unlink_if_exists(missing_path)
                        except Exception as e:
                            print(f"[WARN] failed to reuse report {report_path}: {e}", file=sys.stderr)
                        else:
                            continue
                    elif status == "invalid_final":
                        stats["reused_invalid"] += 1
                        stats["skipped"] += 1
                        _unlink_if_exists(error_path)
                        _unlink_if_exists(missing_path)
                        continue

                try:
                    report = api.score_pair_from_files(str(ref_path), str(gen_path), sample_id=sample_id)
                    stats["scored"] += 1

                    payload = _report_payload(
                        report,
                        ref_path=ref_path,
                        gen_path=gen_path,
                        yaml_path=yaml_path,
                        code_sig=code_sig,
                    )
                    semantic_score = _safe_float(payload.get("semantic_score"))
                    semantic_score_0_100 = _safe_float(payload.get("semantic_score_0_100"))

                    with open(report_path, "w", encoding="utf-8") as fw:
                        json.dump(payload, fw, ensure_ascii=False, indent=2, allow_nan=False)

                    if payload.get("valid_score", False):
                        write_csv_rows(
                            pcsv,
                            global_csv,
                            model_name,
                            sample_id,
                            float(payload["axis"]["catcov"]),
                            float(payload["axis"]["aic"]),
                            float(payload["axis"]["hal"]),
                            float(semantic_score),
                            float(semantic_score_0_100),
                        )
                        stats["written"] += 1
                        stats["valid"] += 1
                    else:
                        stats["skipped"] += 1

                    _unlink_if_exists(error_path)
                    _unlink_if_exists(missing_path)

                except Exception as e:
                    stats["error"] += 1
                    with open(error_path, "w", encoding="utf-8") as fw:
                        json.dump(
                            {
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                                "sample_id": sample_id,
                                "ref": str(ref_path),
                                "gen": str(gen_path),
                            },
                            fw,
                            ensure_ascii=False,
                            indent=2,
                        )

            print(
                f"[{model_name}] scored={stats['scored']} valid={stats['valid']} reused={stats['reused']} "
                f"reused_invalid={stats['reused_invalid']} skipped={stats['skipped']} missing={stats['missing']} "
                f"error={stats['error']} written={stats['written']}"
            )
            print(f"[{model_name}] reports: {reports_dir}")
            print(f"[all-models] CSV: {global_csv}")

    summary_csv = out_root / "semantic_scores_summary.csv"
    shutil.copyfile(global_csv, summary_csv)
    print(f"[summary] CSV: {summary_csv}")

if __name__ == "__main__":
    main()
