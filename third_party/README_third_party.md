# Third-Party Dependencies

This directory contains vendored third-party code used by Ref4D-VideoBench. Each upstream project keeps its own license files and citation requirements. The top-level Apache-2.0 license only applies to original Ref4D-VideoBench code and documentation.

| Component | Upstream | License | Used By | Setup Note |
| --- | --- | --- | --- | --- |
| GroundingDINO | https://github.com/IDEA-Research/GroundingDINO | Apache-2.0 | Motion foreground/background masking | Code is vendored under `third_party/GroundingDINO`; weights go to `checkpoints/groundingdino/`. |
| SAM 2 | https://github.com/facebookresearch/sam2 | Apache-2.0, with additional notices for bundled demo assets/fonts | Motion mask propagation | Code is vendored under `third_party/sam2`; weights go to `checkpoints/sam2/`. |
| TransNetV2 | https://github.com/soCzech/TransNetV2 | MIT | Event shot-boundary detection | Code is vendored under `third_party/transnetv2`; weights go to `checkpoints/transnetv2/`. |
| DDMNet | https://github.com/MCG-NJU/DDM | MIT | Event dense boundary / GEBD scoring support | Code is vendored under `third_party/ddmnet`; weights go to `checkpoints/ddmnet/`. |
| TAPIR / TAPNet | https://github.com/google-deepmind/tapnet | Apache-2.0, with separate dataset notices in upstream subdirectories | Motion point tracking | Code is vendored under `third_party/tapir`; weights go to `checkpoints/tapnet_checkpoints/`. |
| VideoLLaMA3 | https://github.com/DAMO-NLP-SG/VideoLLaMA3 | Apache-2.0 | Event VLM event description | Code is vendored under `third_party/videollama3`; model weights go to `checkpoints/videollama3-7b/`. |

Model-only dependencies such as MiniCPM-V, E5, and BERT are not vendored as source code here. Their local checkpoint locations are documented in `checkpoints/README_checkpoints.md` and `docs/QUICKSTART.md`.
