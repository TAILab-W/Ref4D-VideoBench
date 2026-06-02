# Ref Collect Pipeline

This directory contains the end-to-end reference data collection pipeline for Ref4D:

1. Collect YouTube Creative Commons metadata (`youtube_cc_manifest.py`)
2. Download videos by category (`download_videos.py`)
3. Detect shot boundaries and extract clips (`video_shot_detector.py` / `video_shot_detector_gpu.py`)

## Compliance Notice

These scripts are research utilities for users who have the right to access and process the source videos. They do not grant rights to any third-party content and do not bypass platform terms, rights-holder permissions, or local law. The public reference-source index in `data/metadata/ref4d_videobench_reference_sources.csv` is provenance metadata, not a download manifest.

## Default Path Layout

All scripts are now aligned to the current repository structure.

- Metadata outputs:
  - `Ref4D-VideoBench/ref4d_build/ref_collect/youtube_manifest.json`
  - `Ref4D-VideoBench/ref4d_build/ref_collect/youtube_manifest.csv`
- Downloaded reference videos:
  - `Ref4D-VideoBench/ref4d_build/ref_collect/downloaded_videos/<category>/*.mp4`
- Shot clips (CPU):
  - `Ref4D-VideoBench/ref4d_build/ref_collect/downloaded_clip/`
- Shot clips (GPU):
  - `Ref4D-VideoBench/ref4d_build/ref_collect/downloaded_clip/`
- Runtime state / resume files:
  - `Ref4D-VideoBench/ref4d_build/ref_collect/state/`
- Logs:
  - `Ref4D-VideoBench/ref4d_build/ref_collect/logs/`

## 1) Collect Metadata

Run from repository root (`Ref4D-VideoBench`):

```bash
python ref4d_build/ref_collect/youtube_cc_manifest.py \
  --api-key "YOUR_YOUTUBE_DATA_API_KEY" \
  --max-per-category 50
```

Useful options:

- `--category <name>`: collect only one category
- `--progress-file <path>`: custom resume file
- `--output-json <path>` / `--output-csv <path>`: custom output paths
- `--list-categories`: list supported categories

The collector requests `videoLicense=creativeCommon` during YouTube search and then verifies each retained item through `videos.list(..., part="snippet,contentDetails,status")` by checking `status.license == creativeCommon`. Items whose license cannot be verified are not exported to the JSON/CSV manifest.

## 2) Prepare Authorized Videos

If you have the necessary rights and platform permission, use the metadata from step 1 to prepare local source videos:

```bash
python ref4d_build/ref_collect/download_videos.py \
  --input ref4d_build/ref_collect/youtube_manifest.json
```

Useful options:

- `--category <name>` (repeatable): prepare selected categories only
- `--limit <N>`: max videos per category
- `--output-dir <path>`: custom target (default `ref4d_build/ref_collect/downloaded_videos`)
- `--state-dir <path>`: custom resume-state directory
- `--proxy <url>`: set proxy for yt-dlp
- `--quality best|1080p|720p|480p`
- `--cleanup`: clean stale fragment files and exit

The downloader also checks the manifest license field and skips entries that are missing `license=creativeCommon`.

## 3) Shot Detection and Clip Extraction

### CPU

```bash
python ref4d_build/ref_collect/video_shot_detector.py
```

### GPU (NVENC)

```bash
python ref4d_build/ref_collect/video_shot_detector_gpu.py
```

Useful options:

- `--theme <name>`: one or more themes
- `--input <path>`: input video root (default `ref4d_build/ref_collect/downloaded_videos`)
- `--output <path>`: clip output path
- `--ffmpeg <path>`: FFmpeg executable
- `--progress-file <path>`: resume-state file
- GPU script: `--gpu-id <id>` and `--no-gpu`

## Notes

- The scripts support resume mode through files in `ref4d_build/ref_collect/state/`.
- If an interruption happens, rerun the same command to continue.
- Keep `ffmpeg` and `yt-dlp` available in your environment `PATH`.
