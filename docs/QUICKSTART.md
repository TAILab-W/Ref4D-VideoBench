# Quickstart

This guide gets you from a fresh checkout to a Ref4D-VideoBench evaluation run. Run all commands from the repository root.

This release supports four evaluation dimensions:

| Dimension | Environment | Main output |
| --- | --- | --- |
| semantic | `ref4d_semantic_world` | `outputs/semantic/scores/semantic_scores_summary.csv` |
| event | `ref4d_event` | `outputs/event/scores/event_scores_summary.csv` |
| motion | `ref4d_motion` | `outputs/motion/scores/motion_scores_summary.csv` |
| world | `ref4d_semantic_world` | `outputs/world/scores/world_scores_summary.csv` |

## 1. Pick A Workflow

**Mode 1: evaluate generated videos with released Ref4D caches.** Use this when you already have Ref4D metadata and reference-side caches. This is the standard benchmark evaluation path.

**Mode 2: build caches from custom reference videos.** Use this when you want to create prompts and reference caches from your own videos. See [CUSTOM_REFERENCE.md](CUSTOM_REFERENCE.md) for the full workflow.

## 2. Install Environments

Use three conda environments. The semantic, event, and motion stacks rely on different model runtimes, so a single merged environment is not recommended.

```bash
conda env create -f envs/ref4d_semantic_world.yml
conda env create -f envs/ref4d_event.yml
conda env create -f envs/ref4d_motion.yml
```

Install the repository and check entrypoints:

```bash
conda activate ref4d_semantic_world
bash scripts/install_env.sh
python -m ref4d_eval.semantic.semantics_evi_score_dist --help
python -m ref4d_eval.world.runner --help
```

```bash
conda activate ref4d_event
bash scripts/install_env.sh
python -m pip install ninja flash-attn==2.7.4.post1 --no-build-isolation
python -m ref4d_eval.event.src.cli.main --help
```

```bash
conda activate ref4d_motion
python -m pip install -e . --no-deps
python -m ref4d_eval.motion.run_batch_motion --help
```

## 3. Prepare Checkpoints

Run the download scripts once, then run the offline checks.

```bash
conda activate ref4d_semantic_world
bash scripts/download_semantic_world_models.sh
LOCAL_FILES_ONLY=1 bash scripts/download_semantic_world_models.sh
```

```bash
conda activate ref4d_event
bash scripts/download_event_models.sh
LOCAL_FILES_ONLY=1 bash scripts/download_event_models.sh
```

```bash
conda activate ref4d_motion
AUTO_CLONE=1 bash scripts/download_motion_models.sh
LOCAL_FILES_ONLY=1 bash scripts/download_motion_models.sh
```

Required checkpoint locations:

```text
checkpoints/minicpm-v-4_5/
checkpoints/e5-large-v2/
checkpoints/videollama3-7b/
checkpoints/ddmnet/checkpoint.pth.tar
checkpoints/transnetv2/transnetv2-pytorch-weights.pth
checkpoints/groundingdino/groundingdino_swint_ogc.pth
checkpoints/sam2/sam2.1_hiera_large.pt
checkpoints/tapnet_checkpoints/bootstapir_checkpoint_v2.pt
checkpoints/bert-base-uncased/
```

If an automatic download is unavailable, place the file at the expected path and rerun the matching `LOCAL_FILES_ONLY=1` command.

## 4. Prepare Mode 1 Data

Ref4D metadata and reference caches:

```text
data/metadata/ref4d_meta.jsonl
data/metadata/ref4d_prompts.jsonl
data/metadata/semantic_evidence/
data/metadata/event_evidence/events_merged_ref/
data/metadata/event_evidence/embeds_merged_ref/
data/metadata/motion_ref/
data/metadata/world_qa/
```

`data/metadata/semantic_event_evidence/` is a derived merge cache for prompt and custom-reference construction. It is not required for standard Mode 1 evaluation and can be regenerated from the released semantic and event evidence if needed.

Generated videos:

```text
data/genvideo/<model_name>/<sample_id>.mp4
```

Example:

```text
data/genvideo/my_model/ref4d_0001.mp4
data/genvideo/my_model/ref4d_0002.mp4
```

The generated video filename stem must match the `sample_id` in the metadata and reference caches. `data/genvideo/` is only the default root; wrapper scripts also accept `GEN_VIDEO_ROOT=/path/to/videos` with the same `<video_root>/<model_name>/<sample_id>.mp4` layout.

## 5. Run Evaluation

### Unified Run

Use the unified entrypoint for the common four-dimension run:

```bash
USE_CONDA_ENVS=1 \
OUTPUT_SUFFIX=run1 \
MODELS=my_model \
SEMANTIC_GPUS=0 \
MOTION_WORKERS=6 \
bash scripts/run_4d_eval.sh
```

### Single-Dimension Runs

Use these when you only need one dimension or want separate control over each step:

```bash
export GEN_VIDEO_ROOT=/path/to/videos
export MODELS=my_model
```

```bash
conda activate ref4d_semantic_world
GPUS=0 bash scripts/run_semantic_eval.sh
```

```bash
conda activate ref4d_event
bash scripts/run_event_eval.sh
```

```bash
conda activate ref4d_motion
WORKERS=2 bash scripts/run_motion_eval.sh
```

```bash
conda activate ref4d_semantic_world
bash scripts/run_world_eval.sh
```

### Multiple Models

Use comma-separated model directory names:

```bash
USE_CONDA_ENVS=1 \
OUTPUT_SUFFIX=run_multi \
MODELS=model_a,model_b \
SEMANTIC_GPUS=0 \
MOTION_WORKERS=6 \
bash scripts/run_4d_eval.sh
```

Generated videos should be arranged as:

```text
data/genvideo/model_a/<sample_id>.mp4
data/genvideo/model_b/<sample_id>.mp4
```

## 6. Check Outputs

Default summary outputs:

```text
outputs/semantic/scores/semantic_scores_summary.csv
outputs/event/scores/event_scores_summary.csv
outputs/motion/scores/motion_scores_summary.csv
outputs/world/scores/world_scores_summary.csv
```

If `OUTPUT_SUFFIX=name` is used, summaries are written to:

```text
outputs/<name>_semantic/scores/semantic_scores_summary.csv
outputs/<name>_event/scores/event_scores_summary.csv
outputs/<name>_motion/scores/motion_scores_summary.csv
outputs/<name>_world/scores/world_scores_summary.csv
```

## 7. Custom References

To build prompts and reference caches from your own videos, follow [CUSTOM_REFERENCE.md](CUSTOM_REFERENCE.md). After the custom caches are built and generated videos are placed under `<video_root>/<model_name>/<custom_sample_id>.mp4`, the same semantic, event, motion, and world-knowledge evaluators can be reused with custom roots.
