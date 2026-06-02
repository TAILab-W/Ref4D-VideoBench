# Custom Reference Videos

This guide describes Mode 2: building Ref4D-style reference-side caches and prompts from user-provided reference videos. Run all commands from the repository root.

After completing this workflow, generated videos for the new prompts can be evaluated with the same semantic, event, motion, and world-knowledge evaluators used in Mode 1.

## Workflow Overview

Input:

```text
data/refvideo_custom/<sample_id>.mp4
data/metadata/custom/ref4d_meta_custom.jsonl
```

Reference-side outputs:

```text
data/metadata/custom/semantic_evidence/
data/metadata/custom/event_evidence/events_merged_ref/
data/metadata/custom/event_evidence/embeds_merged_ref/
data/metadata/custom/semantic_event_evidence/
data/metadata/custom/ref4d_prompts_custom.jsonl
data/metadata/custom/motion_ref/
data/metadata/custom/world_qa/
```

Generated videos for evaluation:

```text
<video_root>/<model_name>/<sample_id>.mp4
```

The wrapper scripts use `data/genvideo/` by default, but you can point them to another generated-video root with `GEN_VIDEO_ROOT=/path/to/videos`.

## 1. Prepare Inputs

Create a metadata JSONL file. Each row needs a stable `sample_id`, a reference video path, and prompt-generation metadata.

```bash
mkdir -p data/refvideo_custom data/metadata/custom

printf '{"sample_id":"custom_0001","ref_video":"data/refvideo_custom/custom_0001.mp4","shot_type":"single","theme":"custom"}\n' \
  > data/metadata/custom/ref4d_meta_custom.jsonl
```

Place the reference video at:

```text
data/refvideo_custom/custom_0001.mp4
```

For multiple custom videos, add one JSON object per line and keep each filename aligned with its `sample_id`.

## 2. Build Reference Evidence

### Semantic Evidence

```bash
conda activate ref4d_semantic_world
python -m ref4d_build.semantic_ref.build_semantic_ref \
  --meta-path data/metadata/custom/ref4d_meta_custom.jsonl \
  --out-dir data/metadata/custom/semantic_evidence \
  --local-path checkpoints/minicpm-v-4_5 \
  --local-files-only \
  --device cuda \
  --dtype bf16 \
  --force
```

Expected output:

```text
data/metadata/custom/semantic_evidence/custom_0001.json
```

### Event Evidence

```bash
conda activate ref4d_event
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m ref4d_build.event_ref.build_event_ref_cache run \
  --sample-id custom_0001 \
  --cfg-default ref4d_eval/event/configs/default.yaml \
  --cfg-shot ref4d_eval/event/configs/model_shot.yaml \
  --cfg-gebd ref4d_eval/event/configs/model_gebd.yaml \
  --cfg-vlm ref4d_eval/event/configs/model_vlm.yaml \
  --cfg-embed ref4d_eval/event/configs/model_embed.yaml \
  --ref-video-root data/refvideo_custom \
  --work-root outputs/event_custom/cache \
  --publish-root data/metadata/custom/event_evidence \
  --meta-path data/metadata/custom/ref4d_meta_custom.jsonl \
  --force
```

For multiple custom videos, use the `batch` subcommand with the same shared paths:

```bash
python -m ref4d_build.event_ref.build_event_ref_cache batch \
  --cfg-default ref4d_eval/event/configs/default.yaml \
  --cfg-shot ref4d_eval/event/configs/model_shot.yaml \
  --cfg-gebd ref4d_eval/event/configs/model_gebd.yaml \
  --cfg-vlm ref4d_eval/event/configs/model_vlm.yaml \
  --cfg-embed ref4d_eval/event/configs/model_embed.yaml \
  --ref-video-root data/refvideo_custom \
  --work-root outputs/event_custom/cache \
  --publish-root data/metadata/custom/event_evidence \
  --meta-path data/metadata/custom/ref4d_meta_custom.jsonl \
  --force
```

Expected outputs:

```text
data/metadata/custom/event_evidence/events_merged_ref/custom_0001.newevents.json
data/metadata/custom/event_evidence/embeds_merged_ref/custom_0001.emb.merged.json
```

### Merged Semantic-Event Evidence

```bash
conda activate ref4d_semantic_world
python -m ref4d_build.common.merge_semantic_event_evidence \
  --meta-path data/metadata/custom/ref4d_meta_custom.jsonl \
  --semantic-root data/metadata/custom/semantic_evidence \
  --event-root data/metadata/custom/event_evidence/events_merged_ref \
  --out-root data/metadata/custom/semantic_event_evidence \
  --force
```

Expected output:

```text
data/metadata/custom/semantic_event_evidence/custom_0001_semantic_event.json
```

Merged semantic-event evidence excludes event embeddings. Event embeddings remain in `event_evidence/embeds_merged_ref/` for event matching.

## 3. Generate Prompts

For one video:

```bash
conda activate ref4d_semantic_world
python ref4d_build/prompt/video_prompt_generator_scenedetect.py \
  --video data/refvideo_custom/custom_0001.mp4 \
  --json data/metadata/custom/semantic_event_evidence/custom_0001_semantic_event.json \
  --out outputs/prompt_custom \
  --output-jsonl data/metadata/custom/ref4d_prompts_custom.jsonl \
  --video-type single \
  --sample-id custom_0001 \
  --theme custom \
  --model-path checkpoints/minicpm-v-4_5 \
  --device cuda \
  --dtype bf16
```

For a flat directory of custom videos:

```bash
conda activate ref4d_semantic_world
VIDEO_BASE_DIR=data/refvideo_custom \
EVIDENCE_BASE_DIR=data/metadata/custom/semantic_event_evidence \
OUTPUT_JSONL=data/metadata/custom/ref4d_prompts_custom.jsonl \
METADATA_JSONL=data/metadata/custom/ref4d_meta_custom.jsonl \
MODEL_PATH=checkpoints/minicpm-v-4_5 \
bash scripts/build_prompts_from_refvideo.sh --device cuda --dtype bf16
```

For mixed single-shot and multi-shot custom sets, provide `shot_type` in `METADATA_JSONL` or `SOURCE_INDEX` so the batch runner can resolve it by `sample_id`. Use `--default-video-type single|multi` only as a fallback for custom samples that do not have metadata.

Expected output:

```text
data/metadata/custom/ref4d_prompts_custom.jsonl
```

## 4. Build Motion Reference Cache

Create a custom motion config, for example `outputs/motion/custom_motion_ref4d.yaml`:

```yaml
dataset:
  meta_path: data/metadata/custom/ref4d_meta_custom.jsonl
ref:
  cache_root: data/metadata/custom/motion_ref
subject:
  semantic_root: data/metadata/custom/semantic_evidence
```

Run motion reference-cache construction:

```bash
conda activate ref4d_motion
python -m ref4d_build.motion_ref.build_motion_ref_cache \
  --cfg outputs/motion/custom_motion_ref4d.yaml \
  --base . \
  --workers 1
```

Expected output:

```text
data/metadata/custom/motion_ref/custom_0001.npz
```

## 5. Build World Knowledge QA Bank

The world-knowledge dimension requires a reference-side QA bank built from the merged semantic-event evidence.

```bash
conda activate ref4d_semantic_world
python ref4d_build/world_ref/build_world_qa.py \
  --evidence-dir data/metadata/custom/semantic_event_evidence \
  --video-dir data/refvideo_custom \
  --out-dir data/metadata/custom/world_qa \
  --cache-dir outputs/world_custom/cache/build \
  --model-path checkpoints/minicpm-v-4_5 \
  --device cuda \
  --dtype bf16
```

This runs a four-step pipeline (rule generation, assertion generation, VQA generation, and bank scoring) and produces:

```text
data/metadata/custom/world_qa/<sample_id>_scored.json
```

Intermediate build caches are written to `outputs/world_custom/cache/build/`.

## 6. Evaluate Generated Videos

Place generated videos for the custom prompts under:

```text
data/genvideo/<model_name>/custom_0001.mp4
```

Semantic:

```bash
conda activate ref4d_semantic_world
REF_VIDEO_DIR=data/refvideo_custom \
REF_OUT_DIR=data/metadata/custom/semantic_evidence \
GEN_VIDEO_ROOT=data/genvideo \
GEN_OUT_ROOT=outputs/semantic_custom/cache/evidence_gen \
SCORES_OUT_DIR=outputs/semantic_custom/scores \
MODELS=<model_name> \
GPUS=0 \
STEPS=both \
bash scripts/run_semantic_eval.sh
```

Event:

```bash
conda activate ref4d_event
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
META_PATH=data/metadata/custom/ref4d_meta_custom.jsonl \
REF_VIDEO_ROOT=data/refvideo_custom \
GEN_VIDEO_ROOT=data/genvideo \
REF_EVENT_ROOT=data/metadata/custom/event_evidence \
CACHE_ROOT=outputs/event_custom_eval/cache \
SCORES_ROOT=outputs/event_custom_eval/scores \
MODELS=<model_name> \
STEPS=detect,vlm,embed,merge,match,metrics \
bash scripts/run_event_eval.sh
```

Motion:

```bash
conda activate ref4d_motion
CFG=outputs/motion/custom_motion_ref4d.yaml \
OUT=outputs/motion_custom/scores/motion_scores_summary.csv \
WORKERS=1 \
MODELS=<model_name> \
bash scripts/run_motion_eval.sh
```

World Knowledge:

```bash
conda activate ref4d_semantic_world
python ref4d_eval/world/runner.py \
  --bank-dir data/metadata/custom/world_qa \
  --video-dir <video_root>/<model_name> \
  --out-dir outputs/world_custom \
  --local-path checkpoints/minicpm-v-4_5 \
  --device cuda \
  --dtype bf16
```

Expected summaries:

```text
outputs/semantic_custom/scores/semantic_scores_summary.csv
outputs/event_custom_eval/scores/event_scores_summary.csv
outputs/motion_custom/scores/motion_scores_summary.csv
outputs/world_custom/scores/world_scores_summary.csv
```

Custom event evaluation requires explicit custom roots. Use `META_PATH`, `REF_VIDEO_ROOT`, `GEN_VIDEO_ROOT`, `REF_EVENT_ROOT`, `CACHE_ROOT`, and `SCORES_ROOT` instead of copying custom caches into the default Ref4D metadata directory.
