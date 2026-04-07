# ComfyUI Chunked Batch Nodes

Custom nodes for chunked batched image-to-image workflows.

Included nodes:

- **VAE Encode Batch Chunked**
- **SamplerCustomAdvanced Chunked**
- **VAE Decode Batch Chunked**

## Purpose

These nodes let you process large image/frame batches in smaller execution chunks while preserving batch order and `SamplerCustomAdvanced` behavior as closely as possible.

Typical workflow:

`IMAGE batch -> VAE Encode Batch Chunked -> SamplerCustomAdvanced Chunked -> VAE Decode Batch Chunked`

## Notes

- `SamplerCustomAdvanced Chunked` reuses the normal `guider.sample(...)` path per chunk instead of reimplementing denoising logic.
- When the input latent has no `batch_index` and chunking is required, the sampler node synthesizes a sequential internal `batch_index` so chunked random noise generation stays frame-stable across chunks.
- `noise_mask` batch slicing supports the common ComfyUI broadcast cases (`1` mask for the whole batch, or shorter masks that repeat across the full batch).
- The sampler node supports OOM fallback by halving the chunk size down to `min_chunk_size`.

## Limitations

- This does **not** add temporal consistency. It only makes large frame batches tractable.
- Shared conditioning is the intended v1 use case. Per-frame conditioning batches inside the guider path are not explicitly rewritten by these nodes.
- The main target is regular latent batches used for Flux / img2img-style cleanup. Nested latent edge cases are handled conservatively but are not the primary tested path.

## Installation

Copy this folder into your ComfyUI `custom_nodes` directory and restart ComfyUI.
