# Prompt Generation Pipeline (`ref4d_build/prompt`)

This directory contains the Ref4D video-to-prompt generation workflow.

- `video_prompt_generator_scenedetect.py`: Single-video prompt generator (supports both single-shot and multi-shot videos).
- `batch_video_prompt_generator.py`: Batch entrypoint that scans videos by theme and calls the single-video script.

## Unified Directory Convention

The scripts now use the following default paths:

- Video input root: `Ref4D-VideoBench/data/refvideo`
- Evidence input root: `Ref4D-VideoBench/data/metadata/semantic_event_evidence`
- Prompt output file: `Ref4D-VideoBench/data/metadata/ref4d_prompts.jsonl`
- MiniCPM model root: `Ref4D-VideoBench/checkpoints/minicpm-v-4_5`

Batch progress is stored at:

- `Ref4D-VideoBench/data/metadata/prompt_progress.json`

If `data/metadata/semantic_event_evidence/` has no generated `*_semantic_event.json` files, build them first:

```bash
python -m ref4d_build.common.merge_semantic_event_evidence --force
```

## 1) Single Video Script

File: `video_prompt_generator_scenedetect.py`

What it does:

- Takes one video plus its matching evidence JSON.
- Uses `--video-type single|multi` to choose single-shot or multi-shot behavior.
- Uses `--sample-id` and `--theme` when provided; otherwise infers `sample_id` from the flat video stem and resolves `theme` from `data/metadata/ref4d_prompts.jsonl` when available.
- In multi-shot mode, runs PySceneDetect for shot boundary detection.
- Writes a temporary `<video_stem>.txt` into `--out` (default: `data/metadata/_prompt_tmp`).
- Appends one merged JSONL record into `--output-jsonl` (default: `data/metadata/ref4d_prompts.jsonl`).

Example (single shot):

```bash
python ref4d_build/prompt/video_prompt_generator_scenedetect.py \
  --video data/refvideo/ref4d_0001.mp4 \
  --json data/metadata/semantic_event_evidence/ref4d_0001_semantic_event.json \
  --video-type single \
  --model-path checkpoints/minicpm-v-4_5
```

Example (multi shot):

```bash
python ref4d_build/prompt/video_prompt_generator_scenedetect.py \
  --video data/refvideo/ref4d_0002.mp4 \
  --json data/metadata/semantic_event_evidence/ref4d_0002_semantic_event.json \
  --video-type multi \
  --shot-threshold 35 \
  --min-shot-length 3.0 \
  --model-path checkpoints/minicpm-v-4_5
```

## 2) Batch Script

File: `batch_video_prompt_generator.py`

What it does:

- Recursively scans `--video-base-dir` for `*.mp4` files.
- Resolves `shot_type` and `theme` by `sample_id` from `data/metadata/ref4d_prompts.jsonl` and `data/metadata/ref4d_videobench_reference_sources.csv`; `--default-video-type` is only used when metadata is unavailable.
- Resolves evidence files from `data/metadata/semantic_event_evidence` using the current flat Ref4D naming convention:
  - `ref4d_0001.mp4 -> ref4d_0001_semantic_event.json`
  - looked up first under `--json-base-dir/<theme>/`, then directly under `--json-base-dir/`
- Calls the single-video script per file, then merges outputs into one JSONL file.
- Supports resume mode via `--resume`.

Example (default directories):

```bash
python ref4d_build/prompt/batch_video_prompt_generator.py \
  --model-path checkpoints/minicpm-v-4_5 \
  --resume
```

Example (selected themes):

```bash
python ref4d_build/prompt/batch_video_prompt_generator.py \
  --theme landscape transportation \
  --model-path checkpoints/minicpm-v-4_5
```

Example (dry run):

```bash
python ref4d_build/prompt/batch_video_prompt_generator.py \
  --dry-run \
  --model-path checkpoints/minicpm-v-4_5
```

## Dependencies

At minimum:

- `torch`, `transformers`
- `opencv-python` or `decord`
- `scenedetect` (required for multi-shot mode)

Install PySceneDetect if needed:

```bash
pip install scenedetect
```
