# Checkpoints

This page lists the expected local checkpoint paths for each module.

For installation commands and offline checks, see [docs/QUICKSTART.md](../docs/QUICKSTART.md).

Expected local paths:

| Path | Used By | Setup Note |
| --- | --- | --- |
| `checkpoints/minicpm-v-4_5/` | Semantic evidence extraction and world-knowledge evaluation | Download with `scripts/download_semantic_world_models.sh` or place the local MiniCPM-V checkpoint here. |
| `checkpoints/e5-large-v2/` | Semantic SoftAlign and event embedding | Download with `scripts/download_semantic_world_models.sh` or `scripts/download_event_models.sh`. |
| `checkpoints/videollama3-7b/` | Event VLM event description | Download or place the VideoLLaMA3 checkpoint here. |
| `checkpoints/ddmnet/checkpoint.pth.tar` | Event dense boundary / GEBD support | Download or place the DDMNet checkpoint here. |
| `checkpoints/transnetv2/transnetv2-pytorch-weights.pth` | Event shot-boundary detection | Download or place the TransNetV2 PyTorch weights here. |
| `checkpoints/groundingdino/groundingdino_swint_ogc.pth` | Motion foreground/background masking | Download with `scripts/download_motion_models.sh` or place the GroundingDINO weights here. |
| `checkpoints/sam2/sam2.1_hiera_large.pt` | Motion mask propagation | Download with `scripts/download_motion_models.sh` or place the SAM 2 weights here. |
| `checkpoints/tapnet_checkpoints/bootstapir_checkpoint_v2.pt` | Motion point tracking | Download with `scripts/download_motion_models.sh` or place the TAPIR checkpoint here. |
| `checkpoints/bert-base-uncased/` | GroundingDINO text encoder for motion masks | Download with `scripts/download_motion_models.sh` or place the local BERT checkpoint here. |
