#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0,1,2,3}"
GEN_VIDEO_ROOT="${GEN_VIDEO_ROOT:-data/genvideo}"
META_PATH="${META_PATH:-data/metadata/ref4d_meta.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/event_multigpu}"
SHARED_OUTPUTS="${SHARED_OUTPUTS:-}"
STEPS="${STEPS:-detect,vlm,embed,merge,match,metrics}"
MODELS="${MODELS:-}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
CFG_DEFAULT="${CFG_DEFAULT:-ref4d_eval/event/configs/default.yaml}"
CFG_VLM="${CFG_VLM:-ref4d_eval/event/configs/model_vlm.yaml}"
CFG_EMBED="${CFG_EMBED:-ref4d_eval/event/configs/model_embed.yaml}"
CFG_SHOT="${CFG_SHOT:-ref4d_eval/event/configs/model_shot.yaml}"
CFG_GEBD="${CFG_GEBD:-ref4d_eval/event/configs/model_gebd.yaml}"
EXPORT_AUDIT="${EXPORT_AUDIT:-1}"
AUDIT_OUT="${AUDIT_OUT:-$OUTPUT_ROOT/analysis/event_pair_audit.jsonl}"
AUDIT_TOP_K="${AUDIT_TOP_K:-5}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
N_WORKERS="${#GPU_LIST[@]}"
if (( N_WORKERS <= 0 )); then
  echo "[event-mg] GPUS is empty" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"/{logs,tmp,scores}

if [[ -z "$SHARED_OUTPUTS" ]]; then
  case "$OUTPUT_ROOT" in
    outputs/event|*/outputs/event) SHARED_OUTPUTS=1 ;;
    *) SHARED_OUTPUTS=0 ;;
  esac
fi

samples_tsv="$OUTPUT_ROOT/tmp/samples.tsv"
models_txt="$OUTPUT_ROOT/tmp/models.txt"

"$PYTHON_BIN" - "$META_PATH" "$LIMIT_SAMPLES" > "$samples_tsv" <<'PY'
import json
import sys
from pathlib import Path

meta = Path(sys.argv[1])
limit = int(sys.argv[2])
if not meta.exists():
    raise SystemExit(f"missing META_PATH: {meta}")
seen = set()
count = 0
with meta.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        sid = str(obj.get("sample_id") or obj.get("id") or "").strip()
        if not sid or sid in seen:
            continue
        topic = str(obj.get("topic") or obj.get("theme") or "").strip()
        # Keep sample_id first because Bash read strips leading IFS whitespace;
        # an empty topic in the first column would otherwise shift fields.
        print(f"{sid}\t{topic}")
        seen.add(sid)
        count += 1
        if limit > 0 and count >= limit:
            break
PY

if [[ -n "$MODELS" ]]; then
  tr ',' '\n' <<< "$MODELS" | sed '/^[[:space:]]*$/d' > "$models_txt"
else
  find "$GEN_VIDEO_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -printf '%f\n' | sort > "$models_txt"
fi

sample_count="$(wc -l < "$samples_tsv" | tr -d ' ')"
model_count="$(wc -l < "$models_txt" | tr -d ' ')"
echo "[event-mg] samples=$sample_count models=$model_count workers=$N_WORKERS gpus=$GPUS"
echo "[event-mg] output root: $OUTPUT_ROOT"
echo "[event-mg] shared canonical outputs: $SHARED_OUTPUTS"

if [[ "$sample_count" == "0" || "$model_count" == "0" ]]; then
  echo "[event-mg] no samples or models to run" >&2
  exit 2
fi

run_worker() {
  local worker_idx="$1"
  local gpu="$2"
  local log_file="$OUTPUT_ROOT/logs/worker_${worker_idx}_gpu${gpu}.log"
  local cache_root="$OUTPUT_ROOT/cache/worker_${worker_idx}"
  local scores_root="$OUTPUT_ROOT/scores/worker_${worker_idx}"
  if [[ "$SHARED_OUTPUTS" == "1" ]]; then
    cache_root="$OUTPUT_ROOT/cache"
    scores_root="$OUTPUT_ROOT/scores"
  fi
  mkdir -p "$cache_root" "$scores_root"

  {
    echo "[worker-$worker_idx] gpu=$gpu cache=$cache_root scores=$scores_root"
    local line_no=0
    local assigned=0
    local attempted=0
    local missing=0
    local failed_pairs=0
    while IFS=$'\t' read -r sample_id topic; do
      if (( line_no % N_WORKERS != worker_idx )); then
        line_no=$((line_no + 1))
        continue
      fi
      assigned=$((assigned + 1))
      while IFS= read -r model; do
        [[ -z "$model" ]] && continue
        local p1="$GEN_VIDEO_ROOT/$model/$sample_id.mp4"
        local p2="$GEN_VIDEO_ROOT/$model/$topic/$sample_id.mp4"
        if [[ ! -f "$p1" && ! -f "$p2" ]]; then
          missing=$((missing + 1))
          echo "[worker-$worker_idx] skip missing: model=$model sample=$sample_id topic=$topic"
          continue
        fi
        attempted=$((attempted + 1))
        echo
        echo "[worker-$worker_idx] RUN sample=$sample_id model=$model topic=$topic"
        if ! env \
          CUDA_VISIBLE_DEVICES="$gpu" \
          PYTHON_BIN="$PYTHON_BIN" \
          MODE=run \
          SAMPLE_ID="$sample_id" \
          TOPIC="$topic" \
          MODEL="$model" \
          STEPS="$STEPS" \
          CFG_DEFAULT="$CFG_DEFAULT" \
          CFG_VLM="$CFG_VLM" \
          CFG_EMBED="$CFG_EMBED" \
          CFG_SHOT="$CFG_SHOT" \
          CFG_GEBD="$CFG_GEBD" \
          META_PATH="$META_PATH" \
          GEN_VIDEO_ROOT="$GEN_VIDEO_ROOT" \
          CACHE_ROOT="$cache_root" \
          SCORES_ROOT="$scores_root" \
          EVENT_SKIP_SUMMARY=1 \
          bash scripts/run_event_eval.sh; then
          failed_pairs=$((failed_pairs + 1))
          echo "[worker-$worker_idx] ERROR sample=$sample_id model=$model"
        fi
      done < "$models_txt"
      line_no=$((line_no + 1))
    done < "$samples_tsv"
    echo "[worker-$worker_idx] assigned_samples=$assigned attempted_pairs=$attempted missing_pairs=$missing failed_pairs=$failed_pairs done"
  } > "$log_file" 2>&1
}

pids=()
for idx in "${!GPU_LIST[@]}"; do
  run_worker "$idx" "${GPU_LIST[$idx]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = root / "scores" / "event_scores_summary.csv"
fieldnames = ["modelname", "sample_id", "EGA", "ERel", "ECR", "event_score", "event_score_0_100"]
rows = {}

def dig_score(x):
    if not isinstance(x, dict):
        return ""
    try:
        return f"{float(x.get('score')):.6f}"
    except Exception:
        return ""

def scalar(x):
    if x is None:
        return ""
    try:
        return f"{float(x):.6f}"
    except Exception:
        return ""

for p in sorted((root / "scores").rglob("event_scores.json")):
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[event-mg] skip unreadable score json: {p} -> {exc}", file=sys.stderr)
        continue
    pair_id = p.parent.name
    model = p.parent.parent.name
    sample_id = pair_id.rsplit("__", 1)[0] if "__" in pair_id else pair_id
    if not model or not sample_id:
        continue
    rows[(model, sample_id)] = {
        "modelname": model,
        "sample_id": sample_id,
        "EGA": dig_score(data.get("EGA")),
        "ERel": dig_score(data.get("ERel")),
        "ECR": dig_score(data.get("ECR")),
        "event_score": scalar(data.get("event_score")),
        "event_score_0_100": scalar(data.get("event_score_0_100")),
    }
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix(out.suffix + ".tmp")
with tmp.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for key in sorted(rows):
        w.writerow(rows[key])
tmp.replace(out)
print(f"[event-mg] merged {len(rows)} rows -> {out}")
if not rows:
    log_dir = root / "logs"
    print("[event-mg] ERROR: merged 0 rows; inspect worker logs under", log_dir, file=sys.stderr)
    for log in sorted(log_dir.glob("worker_*.log"))[:4]:
        print(f"[event-mg] log hint: tail -n 80 {log}", file=sys.stderr)
    raise SystemExit(3)
PY

if [[ "$EXPORT_AUDIT" == "1" ]]; then
  "$PYTHON_BIN" scripts/export_event_pair_audit.py \
    --project-root "$repo_root" \
    --event-root "$OUTPUT_ROOT" \
    --out "$AUDIT_OUT" \
    --top-k-candidates "$AUDIT_TOP_K"
fi

exit "$failed"
