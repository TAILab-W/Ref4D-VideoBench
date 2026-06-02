# DATA_SCHEMA

## 1. Purpose

This document defines the data contract for Ref4D-VideoBench. It standardizes:

- the relationship between reference-video collection code and the finalized reference-video records;
- the one-command build path from reference videos to final prompts, including inputs and outputs;
- the organization of input data;
- the locations of reference videos and reference-side assets;
- the separation between prompts and the sample-level metadata index;
- the organization of semantic evidence, event evidence, and merged semantic-event evidence;
- the naming of runtime intermediate artifacts for each evaluation dimension;
- the locations of final evaluation outputs;
- which fields are required and which files are caches or auxiliary artifacts.

This contract covers:

- `data/`
- `outputs/`
- the data-facing parts of `ref4d_eval/`, `ref4d_build/`, and `scripts/`

---

## 2. Global Conventions

### 2.1 Sample Primary Key

The repository uses `sample_id` as the global sample primary key. The same sample must keep the same identifier across all directories and files.

Relevant locations include:

- `data/metadata/ref4d_meta.jsonl`
- `data/metadata/ref4d_prompts.jsonl`
- `data/refvideo/<sample_id>.mp4`
- `<video_root>/<model_name>/<sample_id>.mp4`
- `data/metadata/semantic_event_evidence/<sample_id>_semantic_event.json`, when the derived merged cache is generated
- `data/metadata/motion_ref/<sample_id>.npz`
- sample identifiers in per-dimension caches and result files

### 2.2 Model Name

`model_name` denotes the generated-video model name. It maps directly to directory names:

- `<video_root>/<model_name>/`
- `outputs/semantic/cache/evidence_gen/<model_name>/`
- `outputs/event/scores/<model_name>/`

### 2.3 Reference Video Root

The repository uses `data/refvideo/` as the default reference-video root.

Each reference video should be named as:

```text
data/refvideo/<sample_id>.mp4
```

### 2.4 Generated Video Root

The repository uses `video_root` to denote the root directory of generated videos to be evaluated.

- Default value: `data/genvideo`
- Users may explicitly specify another root during evaluation.
- Regardless of the selected root, generated videos must be organized as:

```text
<video_root>/<model_name>/<sample_id>.mp4
```

### 2.4.1 Source-Index Contract

`data/metadata/ref4d_videobench_reference_sources.csv` is a provenance index for research reproducibility and auditing. Fields such as `video_id`, `url`, `title`, `author`, and temporal clip boundaries identify the public source record used during dataset construction.

This source index is not part of the runtime evaluation contract. Official evaluation code should rely on `sample_id` and the released reference-side caches under `data/metadata/`; it must not require source URLs to remain reachable.

The source index is not a download manifest and does not grant rights to the underlying videos. Ref4D-VideoBench does not redistribute source videos, frames, audio, subtitles, or thumbnails. Source availability and platform metadata may change after collection, and users are responsible for obtaining and using any source videos in compliance with platform terms and rights-holder permissions.

### 2.5 File Naming

Unless a dimension has its own established internal naming scheme, use the following conventions:

- Sample file: `<sample_id>.<ext>`
- Sample-level cache: `<sample_id>.<task>.json`
- Summary result: `<dimension>_scores_summary.csv`, where `dimension` is one of `semantic`, `event`, `motion`, and `world`

### 2.6 Path Case

Directory names are case-sensitive. Public data directories should use lowercase names, for example:

- `refvideo/`
- `semantic_evidence/`
- `event_evidence/`
- `semantic_event_evidence/`
- `world_qa/`
- `motion_ref/`

---

## 3. Main Input Data

## 3.1 Sample Metadata Index

### Path

```text
data/metadata/ref4d_meta.jsonl
```

### Purpose

This is the main metadata file. It defines the sample set and provides the mapping between `sample_id` and the reference-video path.

### Minimum Required Fields

Each row must contain at least:

- `sample_id`
- `ref_video`

### Field Definitions

- `sample_id`: unique sample primary key.
- `ref_video`: reference-video path. Repository-relative paths such as `data/refvideo/<sample_id>.mp4` are recommended.

### Constraints

- One row corresponds to one sample.
- `sample_id` must be globally unique.
- `ref_video` must point to the reference video for that sample.
- `ref4d_meta.jsonl` must not store `prompt`.
- `ref4d_meta.jsonl` must not store four-dimensional evaluation results.
- `ref4d_meta.jsonl` must not store runtime cache paths.

### Source

- Official samples are provided by the dataset release.
- User-added samples may be appended by users.

### Usage

- Reference-side build code uses this file to determine the sample set and reference-video locations.
- Evaluation code uses this file to validate sample keys.
- All reference-side assets are aligned by this primary key.

---

## 3.2 Prompt File

### Path

```text
data/metadata/ref4d_prompts.jsonl
```

### Purpose

This file stores the final prompt for each sample. It is separated from `ref4d_meta.jsonl` and serves as an independent, traceable text asset.

### Minimum Required Fields

Each row must contain at least:

- `sample_id`
- `prompt`

### Recommended Fields

Additional fields may be included as needed:

- `topic`

### Constraints

- One row corresponds to one sample.
- `sample_id` must exist in `ref4d_meta.jsonl`.
- `prompt` is the final text prompt used for the sample.

### Source

Built by `ref4d_build/prompt/`, or generated in batch through `scripts/build_prompts_from_refvideo.sh`; the generated prompts may also be manually reviewed and exported.

### Usage

- Used for dataset release.
- Used as prompt input for text-to-video model inference.

---

## 3.3 Reference Videos

### Path

```text
data/refvideo/<sample_id>.mp4
```

### Purpose

Stores the original reference-video input.

### Naming Rule

- The filename must be `<sample_id>.mp4`.

### Source

- Official dataset release, when reference videos are provided separately.
- User-provided videos for custom samples.

### Usage

- Used by `ref4d_build/` to build semantic evidence, event evidence, merged evidence, world-knowledge assets, and motion reference caches.

---

## 3.4 Generated Videos

### Path

```text
<video_root>/<model_name>/<sample_id>.mp4
```

### Default Directory

```text
data/genvideo/<model_name>/<sample_id>.mp4
```

### Purpose

Input generated videos to be evaluated.

### Naming Rule

- The directory name must be the model name: `<model_name>`.
- The filename must be the sample name: `<sample_id>.mp4`.

### Source

- Outputs from external text-to-video generation models.
- User-organized generated videos placed under the required layout.

### Usage

- Main video input for semantic, event, motion, and world-knowledge evaluation.

---

## 4. Reference-Side Data

All reference-side data is stored under `data/metadata/`.

## 4.1 Semantic Reference Evidence

### Path

```text
data/metadata/semantic_evidence/
```

### Purpose

Stores basic semantic evidence for reference videos.

### File Format

The current contract requires this directory to contain JSON-style evidence files. The exact filename template is not strictly enforced, but one-to-one storage by `sample_id` is recommended.

### Source

Built by `ref4d_build/semantic_ref/`.

### Usage

- Used by `ref4d_eval/semantic/`.
- Read by `ref4d_build/common/merge_semantic_event_evidence.py`.

---

## 4.2 Event Reference Evidence

### Paths

```text
data/metadata/event_evidence/events_merged_ref/
data/metadata/event_evidence/embeds_merged_ref/
```

### Purpose

Stores reference-side event evidence and reference-side event embeddings.

### File Naming

- Event evidence: `<sample_id>.newevents.json`
- Embedding: `<sample_id>.emb.merged.json`

### Source

Built by `ref4d_build/event_ref/`.

### Usage

- Used by `ref4d_eval/event/`.
- Read by `ref4d_build/common/merge_semantic_event_evidence.py`.

---

## 4.3 Merged Semantic-Event Reference Evidence

### Path

```text
data/metadata/semantic_event_evidence/
```

### Purpose

Stores unified reference evidence produced by merging reference-side semantic evidence and reference-side event evidence. It is a shared upstream asset for prompt construction and world-knowledge QA-bank construction.

For the official benchmark release, these merged files are treated as a derived cache. They may be omitted from the repository because they can be regenerated from the released `semantic_evidence/` and `event_evidence/` files. Standard Mode 1 evaluation does not require precomputed `semantic_event_evidence/` files.

### File Naming

```text
<sample_id>_semantic_event.json
```

### Minimum Content Requirement

Each file should be traceable to at least the following information:

- `sample_id`
- `semantic_source`
- `event_source`

### Source

Built by `ref4d_build/common/merge_semantic_event_evidence.py`.

### Usage

- Used by `ref4d_build/prompt/` to generate prompts.
- Used by `ref4d_build/world_ref/` to build the world-knowledge QA bank.

### Notes

- `semantic_event_evidence/` is a shared reference-side intermediate asset.
- It does not replace the original `semantic_evidence/` or `event_evidence/`.
- `prompt` and `world_ref` builders read this merged asset directly; they do not fall back to original semantic or event evidence files.
- If generated merged files are absent, regenerate them with `ref4d_build/common/merge_semantic_event_evidence.py` before running prompt or custom world-knowledge QA-bank construction.

---

## 4.4 World-Knowledge QA Bank

### Path

```text
data/metadata/world_qa/
```

### Purpose

Stores the reference-side world-knowledge QA bank or related QA assets.

### Source

Built by `ref4d_build/world_ref/`.

### Upstream Dependency

Default build-time dependency:

```text
data/metadata/semantic_event_evidence/<sample_id>_semantic_event.json
```

The released `world_qa/` files are already built and can be used directly for evaluation. The merged semantic-event evidence is only needed when rebuilding the world-knowledge QA bank.

### Usage

Used by `ref4d_eval/world/`.

---

## 4.5 Motion Reference Cache

### Path

```text
data/metadata/motion_ref/<sample_id>.npz
```

### Purpose

Stores reference-side caches for motion evaluation.

### File Naming

```text
<sample_id>.npz
```

### Source

Built by `ref4d_build/motion_ref/`.

### Usage

Used by `ref4d_eval/motion/`.

### Constraint

The filename must exactly match the `sample_id` in `ref4d_meta.jsonl`.

---

## 5. Code Directories and Their Data Contracts

## 5.1 Evaluation Code

### Path

```text
ref4d_eval/
```

### Purpose

Contains evaluation code only. It should not store raw data.

### Relationship to Data

- Reads the primary index, prompts, and reference-side data from `data/metadata/`.
- Reads generated videos from `<video_root>/`.
- Writes intermediate artifacts to `outputs/<dimension>/cache/`.
- Writes results to `outputs/<dimension>/scores/`.

---

## 5.2 Reference-Side Build Code

### Path

```text
ref4d_build/
```

### Purpose

Contains reference-side build code, including:

- `common/`
- `ref_collect/`
- `prompt/`
- `semantic_ref/`
- `event_ref/`
- `world_ref/`
- `motion_ref/`

### Relationship to Data

- Reads the sample metadata index from `data/metadata/ref4d_meta.jsonl`.
- Reads reference videos from `data/refvideo/`.
- Builds and writes reference-side assets under `data/metadata/`.
- Writes prompts to `data/metadata/ref4d_prompts.jsonl`.

---

## 5.3 Shared Merge Utility

### Path

```text
ref4d_build/common/merge_semantic_event_evidence.py
```

### Purpose

General merge utility. It reads semantic evidence and event evidence, then builds unified semantic-event reference evidence.

### Inputs

- `data/metadata/ref4d_meta.jsonl`
- `data/metadata/semantic_evidence/`
- `data/metadata/event_evidence/events_merged_ref/`
- Optional: `data/metadata/event_evidence/embeds_merged_ref/`

### Output

- `data/metadata/semantic_event_evidence/<sample_id>_semantic_event.json`, when the derived merged cache is generated

### Downstream Consumers

- `ref4d_build/prompt/`
- `ref4d_build/world_ref/`

---

## 5.4 Prompt Build Code

### Path

```text
ref4d_build/prompt/
```

### Purpose

Builds final prompts from merged reference-side evidence.

### Upstream Dependency

Default build-time dependency:

```text
data/metadata/semantic_event_evidence/<sample_id>_semantic_event.json
```

### Output

- `data/metadata/ref4d_prompts.jsonl`

---

## 5.5 World-Knowledge QA-Bank Build Code

### Path

```text
ref4d_build/world_ref/
```

### Purpose

Builds the world-knowledge QA bank from merged reference-side evidence.

### Upstream Dependency

Default build-time dependency:

```text
data/metadata/semantic_event_evidence/<sample_id>_semantic_event.json
```

### Output

- `data/metadata/world_qa/`

---

## 6. Runtime Intermediate Artifacts

## 6.1 Semantic-Dimension Cache

### Path

```text
outputs/semantic/cache/evidence_gen/<model_name>/<sample_id>.semantic_evidence.json
```

### Purpose

Stores semantic evidence extracted from generated videos.

### Source

Produced by `ref4d_eval/semantic/evidence_extract/`.

### Usage

Reused by the semantic scoring workflow.

---

## 6.2 Event-Dimension Cache

### Path

```text
outputs/event/cache/
```

### Subdirectories

- `events/gen/`
- `events_merged/gen/`
- `vlm/gen/`
- `embeds/gen/`
- `scenes/`
- `match/`

### Purpose

Stores intermediate outputs from each stage of the event pipeline.

### Source

Produced by modules under `ref4d_eval/event/src/`.

### Usage

Reused by later event-processing stages to avoid repeated computation.

---

## 6.3 Motion-Dimension Cache

### Path

```text
outputs/motion/cache/
```

### Purpose

Reserved runtime cache directory for motion evaluation.

---

## 6.4 World-Knowledge-Dimension Cache

### Path

```text
outputs/world/cache/
```

### Purpose

Runtime cache directory for world-knowledge evaluation, used for inference intermediates and debugging artifacts.

---

## 7. Final Result Outputs

Final results are written under `outputs/<dimension>/scores/`.

## 7.1 Semantic-Dimension Results

### Path

```text
outputs/semantic/scores/semantic_scores_summary.csv
```

### Purpose

Sample-level summary table for semantic evaluation.

### Fields

- `modelname`
- `sample_id`
- `catcov`
- `aic`
- `hal`
- `semantic_score`
- `semantic_score_0_100`

---

## 7.2 Event-Dimension Results

### Paths

```text
outputs/event/scores/<model_name>/<sample_id>__<model_name>/event_scores.json
outputs/event/scores/event_scores_summary.csv
```

### Purpose

- `event_scores.json`: sample-level event result.
- `event_scores_summary.csv`: event-dimension summary table.

### Fields

- `modelname`
- `sample_id`
- `EGA`
- `ERel`
- `ECR`
- `event_score`
- `event_score_0_100`

---

## 7.3 Motion-Dimension Results

### Path

```text
outputs/motion/scores/motion_scores_summary.csv
```

### Purpose

Sample-level summary table for motion evaluation.

### Fields

- `modelname`
- `sample_id`
- `D_dir`
- `D_mag`
- `D_smo`
- `S_dir`
- `S_mag`
- `S_smo`
- `RF`
- `LS`
- `is_valid_motion`
- `motion_score`
- `motion_score_0_100`
- `error`

---

## 7.4 World-Knowledge-Dimension Results

### Path

```text
outputs/world/scores/world_scores_summary.csv
```

### Purpose

Sample-level summary table for world-knowledge evaluation, generated by the world evaluator called from `scripts/run_world_eval.sh`.

### Minimum Fields

- `modelname`
- `sample_id`
- `world_score`

---

## 8. Third-Party Code and Weights

## 8.1 Third-Party Code

### Path

```text
third_party/
```

### Purpose

Stores third-party code. It is not a data directory.

## 8.2 Pretrained Weights

### Path

```text
checkpoints/
```

### Purpose

Stores model weights required by different modules. These files are not tracked by Git.

---

## 9. Environments and Scripts

## 9.1 Environment Files

### Path

```text
envs/
```

### Purpose

Defines runtime environments for the different modules.

## 9.2 Execution Scripts

### Path

```text
scripts/
```

### Purpose

Handles environment installation, model download, example preparation, reference-side construction, and evaluation execution.

### Key Scripts

- `run_4d_eval.sh`: unified entrypoint for four-dimensional evaluation.
- `run_world_eval.sh`: entrypoint for world-knowledge evaluation.
- `build_prompts_from_refvideo.sh`: one-command entrypoint for building `ref4d_prompts.jsonl` from reference-side video assets.

---

## 10. Minimum Workflow for Adding a New Sample

To add a new evaluable sample, complete at least the following steps:

1. Place the reference video at `data/refvideo/<sample_id>.mp4`.
2. Add one row to `data/metadata/ref4d_meta.jsonl` with `sample_id` and `ref_video`.
3. Run the reference-side build workflow to generate:
   - `semantic_evidence/`
   - `event_evidence/`
   - `semantic_event_evidence/`, if prompt generation or world-knowledge QA-bank construction is needed
   - `world_qa/`, if world-knowledge evaluation is needed
   - `motion_ref/`, if motion evaluation is needed
4. Add or generate the corresponding `prompt` in `data/metadata/ref4d_prompts.jsonl`.
5. Place the corresponding generated video at:

```text
<video_root>/<model_name>/<sample_id>.mp4
```

If `video_root` is not explicitly specified, the default is:

```text
data/genvideo/<model_name>/<sample_id>.mp4
```

---

## 11. Validation Rules

The following rules must be satisfied:

- `sample_id` must be unique across the repository.
- Every `sample_id` used for reference-cache construction or custom-reference construction must have an existing `ref_video`; the official evaluation release may omit local reference videos.
- Every `sample_id` in `ref4d_prompts.jsonl` must exist in `ref4d_meta.jsonl`.
- Every `sample_id` in `semantic_event_evidence/`, when present, must be traceable to the corresponding original semantic and event evidence.
- Every sample participating in evaluation must have a generated video.
- Dimensions that require reference-side caches must have the corresponding reference files.
- Output directories must not write back into `data/`.
- `data/` should contain only inputs and reference-side assets; `outputs/` should contain only intermediate artifacts and results.

---

## 12. Directory Responsibility Summary

- `data/metadata/ref4d_meta.jsonl`: sample metadata index; records only `sample_id` and reference-video paths.
- `data/metadata/ref4d_prompts.jsonl`: sample-level prompt assets.
- `data/metadata/semantic_evidence/`: reference-side semantic evidence.
- `data/metadata/event_evidence/`: reference-side event evidence.
- `data/metadata/semantic_event_evidence/`: derived merged semantic-event reference evidence; regenerate on demand when precomputed files are absent.
- `data/metadata/world_qa/`: reference-side world-knowledge QA bank.
- `data/metadata/motion_ref/`: reference-side motion cache.
- `data/refvideo/`: reference videos.
- `data/genvideo/`: default generated-video directory for evaluation.
- `ref4d_eval/`: evaluation code.
- `ref4d_build/`: reference-side build code.
- `ref4d_build/common/merge_semantic_event_evidence.py`: shared merge utility.
- `outputs/`: runtime caches and results.
- `third_party/`: third-party code.
- `checkpoints/`: model weights.
- `envs/`: environment definitions.
- `scripts/`: execution scripts.
- `docs/`: documentation.
