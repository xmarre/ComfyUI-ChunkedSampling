# ComfyUI Chunked Batch Nodes

Custom nodes for chunked batched image-to-image workflows and sequential temporal video cleanup.

Included nodes:

- **VAE Encode Batch Chunked**
- **SamplerCustomAdvanced Chunked**
- **VAE Decode Batch Chunked**
- **Flux Video Cleanup Temporal Advanced**

## Purpose

These nodes let you process large image/frame batches in smaller execution chunks while preserving the normal ComfyUI `SamplerCustomAdvanced` path as closely as possible.

Typical non-temporal workflow:

`IMAGE batch -> VAE Encode Batch Chunked -> SamplerCustomAdvanced Chunked -> VAE Decode Batch Chunked`

Typical temporal workflow:

`IMAGE batch -> Flux Video Cleanup Temporal Advanced`

## Architecture

The package now splits the reusable runtime into shared helpers:

- `core_sampling.py`
  - `encode_image_batch_chunked(...)`
  - `decode_latent_batch_chunked(...)`
  - `sample_latent_batch_chunked(...)`
  - `sample_single_latent_temporal(...)`
- `nodes_batch_chunked.py`
  - thin wrappers for the generic chunked batch nodes
- `nodes_temporal.py`
  - the temporal recurrent video cleanup node

This keeps `SamplerCustomAdvanced Chunked` as a generic memory-safety node instead of overloading it with frame-to-frame recurrence.

## Flux Video Cleanup Temporal Advanced

Supported temporal modes:

- `off`
  - pure batch path; no recurrence
- `prev_output_blend`
  - blends the current input frame with the previous cleaned output before VAE encode
- `external_warped_prev`
  - blends the current input frame with an externally warped previous image batch
- `internal_flow_warp`
  - computes optical flow internally and warps the previous cleaned output before blending

### Important behavior

- In temporal modes, sampling runs one frame at a time so frame `t` can depend on frame `t-1`.
- `sample_chunk_size` is only used when `temporal_mode=off`.
- `reset_every_n` and `scene_cut_threshold` break recurrence explicitly.
- `lock_seed=True` supports fixed or sequential per-frame seeds based on the incoming `NOISE.seed`.
- `internal_flow_warp` uses OpenCV Farneback flow if `cv2` is available in the ComfyUI Python environment.

## Notes

- `SamplerCustomAdvanced Chunked` still reuses the normal `guider.sample(...)` path per chunk instead of reimplementing denoising logic.
- When the input latent has no `batch_index` and chunking is required, the sampler node synthesizes a sequential internal `batch_index` so chunked random noise generation stays frame-stable across chunks.
- Temporal sequential sampling preserves frame-indexed noise variation when `lock_seed=False`, and intentionally pins batch index to `0` when `lock_seed=True`.
- `noise_mask` batch slicing supports the common ComfyUI broadcast cases (`1` mask for the whole batch, or shorter masks that repeat across the full batch).
- The sampler node supports OOM fallback by halving the chunk size down to `min_chunk_size`.
- VAE encode trims extra image channels beyond RGB before calling `vae.encode(...)`, which preserves compatibility with RGBA-style image batches without changing normal RGB inputs.

## Limitations

- The temporal node uses a pixel-space temporal prior before encode. It does not rewrite guider conditioning on a per-frame basis.
- `lock_seed=True` assumes the `NOISE` input exposes a mutable `seed` attribute, which matches ComfyUI's standard custom sampler noise objects.
- `internal_flow_warp` is a dependency-light built-in flow path, not a RAFT integration.
- Nested latent edge cases are still handled conservatively and are not the primary tested path.

## Installation

Copy this folder into your ComfyUI `custom_nodes` directory and restart ComfyUI.
