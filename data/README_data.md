# Data Layout

This page describes the metadata, prompts, reference-side caches, and video directories used by Ref4D evaluation.

Official metadata and reference-side caches are tracked under `data/metadata/`.

The official release tracks the reference-side caches needed for standard evaluation. `data/metadata/semantic_event_evidence/` is a derived merge cache for prompt and custom-reference construction; it can be regenerated from `semantic_evidence/` and `event_evidence/` when needed.

The reference-source index is `data/metadata/ref4d_videobench_reference_sources.csv`. It maps each `sample_id` to its prompt, shot type, public YouTube source ID/URL, temporal clip boundaries, title, author, and publication year. This file is provenance metadata for research reproducibility and auditing. It is not a download manifest, does not grant rights to the underlying source videos, and does not guarantee that source URLs or platform metadata remain available over time.
