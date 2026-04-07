import contextlib
import gc
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.utils
import latent_preview


LATENT_KEYS_WITH_BROADCAST_BATCH_ONE = {"noise_mask"}


@dataclass(frozen=True)
class SampleChunkConfig:
    chunk_size: int
    auto_reduce_on_oom: bool = True
    min_chunk_size: int = 1
    clear_cache_on_retry: bool = True
    clear_cache_between_chunks: bool = False


def soft_clear_cache() -> None:
    gc.collect()
    comfy.model_management.soft_empty_cache()


def _is_nested(value: Any) -> bool:
    return getattr(value, "is_nested", False)


def _trim_vae_input_channels(pixels: torch.Tensor) -> torch.Tensor:
    if pixels.ndim >= 4 and pixels.shape[-1] > 3:
        return pixels[..., :3]
    return pixels


def _latent_batch_size(samples: Any) -> int:
    if _is_nested(samples):
        return len(samples.unbind())
    return int(samples.shape[0])


def _slice_samples(samples: Any, start: int, end: int) -> Any:
    if _is_nested(samples):
        return comfy.nested_tensor.NestedTensor(samples.unbind()[start:end])
    return samples[start:end]


def _concat_samples(chunks: List[Any]) -> Any:
    if not chunks:
        raise ValueError("expected at least one chunk")
    first = chunks[0]
    if _is_nested(first):
        tensors = []
        for chunk in chunks:
            tensors.extend(chunk.unbind())
        return comfy.nested_tensor.NestedTensor(tensors)
    return torch.cat(chunks, dim=0)


def _repeat_and_slice_batch_tensor(
    value: torch.Tensor,
    start: int,
    end: int,
    full_batch_size: int,
) -> torch.Tensor:
    if value.shape[0] == full_batch_size:
        return value[start:end]
    if value.shape[0] == 1:
        return value
    if value.shape[0] < full_batch_size:
        window_len = end - start
        offset = start % value.shape[0]
        reps = (math.ceil((offset + window_len) / value.shape[0]),) + ((1,) * (value.ndim - 1))
        expanded = value.repeat(reps)
        return expanded[offset : offset + window_len]
    return value[start:end]


def _slice_latent_value(key: str, value: Any, start: int, end: int, full_batch_size: int) -> Any:
    if key == "samples":
        return _slice_samples(value, start, end)

    if isinstance(value, torch.Tensor) and value.ndim > 0:
        if value.shape[0] == full_batch_size:
            return value[start:end]
        if key in LATENT_KEYS_WITH_BROADCAST_BATCH_ONE and value.shape[0] <= full_batch_size:
            return _repeat_and_slice_batch_tensor(value, start, end, full_batch_size)
        return value

    if isinstance(value, list) and len(value) == full_batch_size:
        return value[start:end]

    if isinstance(value, tuple) and len(value) == full_batch_size:
        return value[start:end]

    return value


def _slice_latent_dict(latent: Dict[str, Any], start: int, end: int, full_batch_size: int) -> Dict[str, Any]:
    return {
        key: _slice_latent_value(key, value, start, end, full_batch_size)
        for key, value in latent.items()
    }


def _append_batch_index_if_needed(
    latent: Dict[str, Any],
    chunking_needed: bool,
    forced_batch_index: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, Any], bool]:
    if forced_batch_index is not None:
        if len(forced_batch_index) != _latent_batch_size(latent["samples"]):
            raise ValueError("forced batch_index length must match latent batch size")
        updated = latent.copy()
        updated["batch_index"] = list(forced_batch_index)
        existing = latent.get("batch_index")
        return updated, existing != updated["batch_index"]

    if not chunking_needed or "batch_index" in latent:
        return latent, False

    updated = latent.copy()
    updated["batch_index"] = list(range(_latent_batch_size(latent["samples"])))
    return updated, True


@contextlib.contextmanager
def _temporary_noise_seed(noise: Any, seed_override: Optional[int]):
    if seed_override is None:
        yield noise
        return

    if not hasattr(noise, "seed"):
        raise TypeError("noise seed override requires a NOISE input with a mutable 'seed' attribute")

    original_seed = noise.seed
    try:
        noise.seed = seed_override
    except Exception as exc:
        raise TypeError("noise seed override requires a mutable 'seed' attribute on the NOISE input") from exc
    try:
        yield noise
    finally:
        noise.seed = original_seed


def _process_denoised_chunk(guider: Any, x0_value: Any, sample_chunk: Any) -> Any:
    x0_out = guider.model_patcher.model.process_latent_out(x0_value.cpu())
    if _is_nested(sample_chunk):
        latent_shapes = [x.shape for x in sample_chunk.unbind()]
        x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
    return x0_out


def _run_sampler_chunk(
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    latent_chunk: Dict[str, Any],
    seed_override: Optional[int] = None,
) -> Tuple[Any, Any, bool]:
    latent_image = latent_chunk["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        guider.model_patcher,
        latent_image,
        latent_chunk.get("downscale_ratio_spacial", None),
    )

    working_latent = latent_chunk.copy()
    working_latent["samples"] = latent_image

    noise_mask = working_latent.get("noise_mask", None)
    x0_output: Dict[str, Any] = {}
    callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    with _temporary_noise_seed(noise, seed_override) as seeded_noise:
        samples = guider.sample(
            seeded_noise.generate_noise(working_latent),
            latent_image,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seeded_noise.seed,
        )
    samples = samples.to(comfy.model_management.intermediate_device())

    if "x0" in x0_output:
        denoised = _process_denoised_chunk(guider, x0_output["x0"], samples)
        return samples, denoised, True

    return samples, None, False


def encode_image_batch_chunked(
    pixels: torch.Tensor,
    vae: Any,
    chunk_size: int,
    clear_cache_between_chunks: bool = False,
) -> Dict[str, Any]:
    batch_size = int(pixels.shape[0])
    if batch_size == 0:
        raise ValueError("pixels batch must not be empty")

    encoded_chunks = []
    step = int(chunk_size)
    for start in range(0, batch_size, step):
        end = min(start + step, batch_size)
        encoded_chunks.append(vae.encode(_trim_vae_input_channels(pixels[start:end])))
        if clear_cache_between_chunks and end < batch_size:
            soft_clear_cache()

    return {"samples": _concat_samples(encoded_chunks)}


def decode_latent_batch_chunked(
    samples: Dict[str, Any],
    vae: Any,
    chunk_size: int,
    clear_cache_between_chunks: bool = False,
) -> torch.Tensor:
    latent_samples = samples["samples"]
    batch_size = _latent_batch_size(latent_samples)
    if batch_size == 0:
        raise ValueError("latent batch must not be empty")

    image_chunks = []
    step = int(chunk_size)
    for start in range(0, batch_size, step):
        end = min(start + step, batch_size)
        decoded = vae.decode(_slice_samples(latent_samples, start, end))
        if len(decoded.shape) == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        image_chunks.append(decoded)
        if clear_cache_between_chunks and end < batch_size:
            soft_clear_cache()

    return torch.cat(image_chunks, dim=0)


def sample_latent_batch_chunked(
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    latent_image: Dict[str, Any],
    config: SampleChunkConfig,
    noise_seed_override: Optional[int] = None,
    forced_batch_index: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    original_latent = latent_image
    full_samples = original_latent["samples"]
    batch_size = _latent_batch_size(full_samples)
    if batch_size == 0:
        raise ValueError("latent batch must not be empty")

    effective_chunk_size = max(1, min(int(config.chunk_size), batch_size))
    min_chunk_size = max(1, min(int(config.min_chunk_size), effective_chunk_size))
    chunking_needed = batch_size > effective_chunk_size

    working_latent, synthetic_batch_index = _append_batch_index_if_needed(
        original_latent,
        chunking_needed,
        forced_batch_index=forced_batch_index,
    )

    sample_chunks = []
    denoised_chunks = []
    x0_presence = []

    start = 0
    while start < batch_size:
        attempt_size = min(effective_chunk_size, batch_size - start)
        end = start + attempt_size
        latent_chunk = _slice_latent_dict(working_latent, start, end, batch_size)
        retry_after_oom = False

        try:
            samples_chunk, denoised_chunk, has_x0 = _run_sampler_chunk(
                noise,
                guider,
                sampler,
                sigmas,
                latent_chunk,
                seed_override=noise_seed_override,
            )
        except Exception as exc:
            if not config.auto_reduce_on_oom or not comfy.model_management.is_oom(exc) or effective_chunk_size <= min_chunk_size:
                raise

            new_chunk_size = max(min_chunk_size, effective_chunk_size // 2)
            if new_chunk_size == effective_chunk_size:
                raise

            effective_chunk_size = new_chunk_size
            if (
                forced_batch_index is None
                and not synthetic_batch_index
                and "batch_index" not in original_latent
                and batch_size > effective_chunk_size
            ):
                working_latent, synthetic_batch_index = _append_batch_index_if_needed(original_latent, True)
            retry_after_oom = True

        if retry_after_oom:
            if config.clear_cache_on_retry:
                soft_clear_cache()
            continue

        sample_chunks.append(samples_chunk)
        x0_presence.append(has_x0)
        if has_x0:
            denoised_chunks.append(denoised_chunk)

        start = end
        if config.clear_cache_between_chunks and start < batch_size:
            soft_clear_cache()

    output_samples = _concat_samples(sample_chunks)
    out = original_latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = output_samples

    if synthetic_batch_index and forced_batch_index is None:
        out.pop("batch_index", None)

    if any(x0_presence):
        if not all(x0_presence):
            raise RuntimeError("inconsistent x0 callback behavior across chunks")
        out_denoised = original_latent.copy()
        out_denoised["samples"] = _concat_samples(denoised_chunks)
        if synthetic_batch_index and forced_batch_index is None:
            out_denoised.pop("batch_index", None)
    else:
        out_denoised = out

    return out, out_denoised


def sample_single_latent_temporal(
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    latent_image: Dict[str, Any],
    noise_seed_override: Optional[int] = None,
    forced_batch_index_value: int = 0,
    clear_cache_between_chunks: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return sample_latent_batch_chunked(
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        config=SampleChunkConfig(
            chunk_size=1,
            auto_reduce_on_oom=False,
            min_chunk_size=1,
            clear_cache_on_retry=True,
            clear_cache_between_chunks=clear_cache_between_chunks,
        ),
        noise_seed_override=noise_seed_override,
        forced_batch_index=[int(forced_batch_index_value)],
    )


__all__ = [
    "LATENT_KEYS_WITH_BROADCAST_BATCH_ONE",
    "SampleChunkConfig",
    "decode_latent_batch_chunked",
    "encode_image_batch_chunked",
    "sample_latent_batch_chunked",
    "sample_single_latent_temporal",
    "soft_clear_cache",
]
