from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from .core_sampling import (
    SampleChunkConfig,
    decode_latent_batch_chunked,
    encode_image_batch_chunked,
    sample_latent_batch_chunked,
    sample_single_latent_temporal,
    soft_clear_cache,
)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _frame_batch_size(images: torch.Tensor) -> int:
    if images.ndim != 4:
        raise ValueError(f"expected IMAGE tensor with shape [B,H,W,C], got {tuple(images.shape)}")
    return int(images.shape[0])


def _single_frame(images: torch.Tensor, index: int) -> torch.Tensor:
    return images[index : index + 1]


def _validate_optional_frame_batch(name: str, value: Optional[torch.Tensor], batch_size: int) -> None:
    if value is None:
        return
    actual = _frame_batch_size(value)
    if actual not in {1, batch_size, max(1, batch_size - 1)}:
        raise ValueError(
            f"{name} batch size must be 1, the full frame count, or frame_count - 1; "
            f"got {actual} for {batch_size} frames"
        )


def _extract_optional_frame(
    name: str,
    value: Optional[torch.Tensor],
    frame_index: int,
    batch_size: int,
) -> Optional[torch.Tensor]:
    if value is None:
        return None

    actual = _frame_batch_size(value)
    if actual == 1:
        return value
    if actual == batch_size:
        return _single_frame(value, frame_index)
    if actual == batch_size - 1:
        if frame_index == 0:
            return None
        return _single_frame(value, frame_index - 1)
    raise ValueError(f"{name} batch size mismatch")


def _mask_batch_size(mask: torch.Tensor) -> int:
    if mask.ndim == 2:
        return 1
    if mask.ndim == 3:
        return int(mask.shape[0])
    if mask.ndim == 4 and mask.shape[-1] == 1:
        return int(mask.shape[0])
    raise ValueError(f"unsupported MASK tensor shape {tuple(mask.shape)}")


def _validate_optional_mask_batch(name: str, value: Optional[torch.Tensor], batch_size: int) -> None:
    if value is None:
        return
    actual = _mask_batch_size(value)
    if actual not in {1, batch_size, max(1, batch_size - 1)}:
        raise ValueError(
            f"{name} batch size must be 1, the full frame count, or frame_count - 1; "
            f"got {actual} for {batch_size} frames"
        )


def _extract_optional_mask(
    name: str,
    value: Optional[torch.Tensor],
    frame_index: int,
    batch_size: int,
) -> Optional[torch.Tensor]:
    if value is None:
        return None

    if value.ndim == 2:
        return value.unsqueeze(0)

    if value.ndim == 3:
        actual = int(value.shape[0])
        if actual == 1:
            return value
        if actual == batch_size:
            return value[frame_index : frame_index + 1]
        if actual == batch_size - 1:
            if frame_index == 0:
                return None
            return value[frame_index - 1 : frame_index]

    if value.ndim == 4 and value.shape[-1] == 1:
        actual = int(value.shape[0])
        if actual == 1:
            return value[..., 0]
        if actual == batch_size:
            return value[frame_index : frame_index + 1, ..., 0]
        if actual == batch_size - 1:
            if frame_index == 0:
                return None
            return value[frame_index - 1 : frame_index, ..., 0]

    raise ValueError(f"{name} batch size mismatch")


def _mask_to_image_weight(mask: Optional[torch.Tensor], reference_image: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(f"expected mask with shape [B,H,W], got {tuple(mask.shape)}")

    mask = mask.to(device=reference_image.device, dtype=reference_image.dtype)
    if mask.shape[0] == 1 and reference_image.shape[0] != 1:
        mask = mask.expand(reference_image.shape[0], -1, -1)

    return mask.unsqueeze(-1).clamp(0.0, 1.0)


def _ensure_same_image_shape(current: torch.Tensor, previous: torch.Tensor, context: str) -> torch.Tensor:
    if previous.shape == current.shape:
        return previous.to(device=current.device, dtype=current.dtype)

    if previous.shape[0] != current.shape[0] or previous.shape[-1] != current.shape[-1]:
        raise ValueError(
            f"{context} has incompatible IMAGE shape {tuple(previous.shape)} "
            f"for current frame {tuple(current.shape)}"
        )

    resized = previous.permute(0, 3, 1, 2)
    resized = F.interpolate(resized, size=current.shape[1:3], mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1).to(device=current.device, dtype=current.dtype)


def _frame_difference_score(previous_frame: torch.Tensor, current_frame: torch.Tensor) -> float:
    previous = _ensure_same_image_shape(current_frame, previous_frame, "previous input")
    return float(torch.mean(torch.abs(current_frame - previous)).item())


def _compute_reset(
    frame_index: int,
    previous_input: Optional[torch.Tensor],
    current_input: torch.Tensor,
    reset_every_n: int,
    scene_cut_threshold: float,
) -> bool:
    if frame_index == 0 or previous_input is None:
        return True
    if reset_every_n > 0 and frame_index % reset_every_n == 0:
        return True
    if scene_cut_threshold > 0.0 and _frame_difference_score(previous_input, current_input) >= scene_cut_threshold:
        return True
    return False


def _blend_images(
    current: torch.Tensor,
    prior: torch.Tensor,
    strength: float,
    confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    prior = _ensure_same_image_shape(current, prior, "temporal prior")
    alpha = torch.full_like(current[..., :1], float(strength))
    if confidence is not None:
        alpha = alpha * _mask_to_image_weight(confidence, current)
    alpha = alpha.clamp(0.0, 1.0)
    return current * (1.0 - alpha) + prior * alpha


def _cv2_require() -> Any:
    if cv2 is None:
        raise RuntimeError("internal_flow_warp requires OpenCV (cv2) in the ComfyUI Python environment")
    return cv2


def _image_to_numpy_uint8(frame: torch.Tensor):
    return frame.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()


def _compute_backward_flow(current_frame: torch.Tensor, previous_frame: torch.Tensor):
    cv2_mod = _cv2_require()
    current_np = _image_to_numpy_uint8(current_frame[0])
    previous_np = _image_to_numpy_uint8(previous_frame[0])
    current_gray = cv2_mod.cvtColor(current_np, cv2_mod.COLOR_RGB2GRAY)
    previous_gray = cv2_mod.cvtColor(previous_np, cv2_mod.COLOR_RGB2GRAY)
    return cv2_mod.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def _warp_with_flow(image: torch.Tensor, backward_flow) -> torch.Tensor:
    cv2_mod = _cv2_require()
    import numpy as np

    image_np = image[0].detach().cpu().numpy().astype("float32", copy=False)
    height, width = backward_flow.shape[:2]
    grid_x, grid_y = np.meshgrid(range(width), range(height))
    map_x = (grid_x.astype("float32") + backward_flow[..., 0]).astype("float32")
    map_y = (grid_y.astype("float32") + backward_flow[..., 1]).astype("float32")
    warped = cv2_mod.remap(
        image_np,
        map_x,
        map_y,
        interpolation=cv2_mod.INTER_LINEAR,
        borderMode=cv2_mod.BORDER_REPLICATE,
    )
    return torch.from_numpy(warped).unsqueeze(0).to(device=image.device, dtype=image.dtype)


def _photometric_flow_confidence(
    current_frame: torch.Tensor,
    previous_input_frame: torch.Tensor,
    backward_flow,
) -> torch.Tensor:
    warped_previous_input = _warp_with_flow(previous_input_frame, backward_flow)
    error = torch.mean(torch.abs(current_frame - warped_previous_input), dim=-1, keepdim=True)
    return (1.0 - error).clamp(0.0, 1.0)


def _resolve_temporal_prior(
    temporal_mode: str,
    frame_index: int,
    current_frame: torch.Tensor,
    previous_input: Optional[torch.Tensor],
    previous_output: Optional[torch.Tensor],
    warped_previous_images: Optional[torch.Tensor],
    flow_confidence_mask: Optional[torch.Tensor],
    flow_confidence_scale: float,
    temporal_strength: float,
    batch_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if temporal_mode == "off" or frame_index == 0 or previous_output is None:
        return current_frame, None

    external_mask = _extract_optional_mask("flow_confidence", flow_confidence_mask, frame_index, batch_size)
    if external_mask is not None:
        external_mask = external_mask * float(flow_confidence_scale)
    elif flow_confidence_scale < 1.0:
        external_mask = torch.full(
            (1, current_frame.shape[1], current_frame.shape[2]),
            float(flow_confidence_scale),
            device=current_frame.device,
            dtype=current_frame.dtype,
        )

    if temporal_mode == "prev_output_blend":
        return _blend_images(current_frame, previous_output, temporal_strength), None

    if temporal_mode == "external_warped_prev":
        warped_frame = _extract_optional_frame(
            "warped_previous_images",
            warped_previous_images,
            frame_index,
            batch_size,
        )
        if warped_frame is None:
            raise ValueError("external_warped_prev requires warped_previous_images after the first frame")
        return _blend_images(current_frame, warped_frame, temporal_strength, confidence=external_mask), warped_frame

    if temporal_mode == "internal_flow_warp":
        if previous_input is None:
            return current_frame, None
        previous_input = _ensure_same_image_shape(current_frame, previous_input, "previous input")
        previous_output = _ensure_same_image_shape(current_frame, previous_output, "previous output")
        backward_flow = _compute_backward_flow(current_frame, previous_input)
        warped_previous_output = _warp_with_flow(previous_output, backward_flow)
        confidence = _photometric_flow_confidence(current_frame, previous_input, backward_flow)
        if external_mask is not None:
            confidence = confidence * _mask_to_image_weight(external_mask, current_frame)
        return (
            _blend_images(current_frame, warped_previous_output, temporal_strength, confidence=confidence),
            warped_previous_output,
        )

    raise ValueError(f"unsupported temporal_mode: {temporal_mode}")


def _seed_for_frame(noise: Any, frame_index: int, lock_seed: bool, seed_stride_mode: str) -> Optional[int]:
    if not lock_seed:
        return None
    if not hasattr(noise, "seed"):
        raise TypeError("lock_seed=True requires a NOISE input with a 'seed' attribute")

    base_seed = int(noise.seed)
    if seed_stride_mode == "fixed":
        return base_seed
    if seed_stride_mode == "sequential":
        return base_seed + frame_index
    raise ValueError(f"unsupported seed_stride_mode: {seed_stride_mode}")


def _batch_index_for_frame(frame_index: int, lock_seed: bool) -> int:
    if lock_seed:
        return 0
    return int(frame_index)


class FluxVideoCleanupTemporalAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "vae": ("VAE",),
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "temporal_mode": (("off", "prev_output_blend", "external_warped_prev", "internal_flow_warp"),),
                "temporal_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reset_every_n": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "scene_cut_threshold": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "flow_confidence_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
                "lock_seed": ("BOOLEAN", {"default": True}),
                "seed_stride_mode": (("fixed", "sequential"),),
                "encode_chunk_size": ("INT", {"default": 8, "min": 1, "max": 4096}),
                "sample_chunk_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "Only used when temporal_mode is off. Temporal recurrence forces per-frame sampling.",
                    },
                ),
                "decode_chunk_size": ("INT", {"default": 8, "min": 1, "max": 4096}),
                "auto_reduce_on_oom": ("BOOLEAN", {"default": True}),
                "min_chunk_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "clear_cache_on_retry": ("BOOLEAN", {"default": True, "advanced": True}),
                "clear_cache_between_chunks": ("BOOLEAN", {"default": False, "advanced": True}),
            },
            "optional": {
                "warped_previous_images": ("IMAGE",),
                "flow_confidence": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "LATENT", "MASK")
    RETURN_NAMES = ("images", "latents", "denoised_latents", "reset_mask")
    FUNCTION = "cleanup"
    CATEGORY = "sampling/custom_sampling"

    def cleanup(
        self,
        images,
        vae,
        noise,
        guider,
        sampler,
        sigmas,
        temporal_mode,
        temporal_strength,
        reset_every_n,
        scene_cut_threshold,
        flow_confidence_scale,
        lock_seed,
        seed_stride_mode,
        encode_chunk_size,
        sample_chunk_size,
        decode_chunk_size,
        auto_reduce_on_oom=True,
        min_chunk_size=1,
        clear_cache_on_retry=True,
        clear_cache_between_chunks=False,
        warped_previous_images=None,
        flow_confidence=None,
    ):
        batch_size = _frame_batch_size(images)
        if batch_size == 0:
            raise ValueError("images batch must not be empty")

        _validate_optional_frame_batch("warped_previous_images", warped_previous_images, batch_size)
        _validate_optional_mask_batch("flow_confidence", flow_confidence, batch_size)

        if temporal_mode == "off":
            latents = encode_image_batch_chunked(
                images,
                vae,
                encode_chunk_size,
                clear_cache_between_chunks=clear_cache_between_chunks,
            )
            sampled, denoised = sample_latent_batch_chunked(
                noise,
                guider,
                sampler,
                sigmas,
                latents,
                config=SampleChunkConfig(
                    chunk_size=sample_chunk_size,
                    auto_reduce_on_oom=auto_reduce_on_oom,
                    min_chunk_size=min_chunk_size,
                    clear_cache_on_retry=clear_cache_on_retry,
                    clear_cache_between_chunks=clear_cache_between_chunks,
                ),
            )
            decoded = decode_latent_batch_chunked(
                sampled,
                vae,
                decode_chunk_size,
                clear_cache_between_chunks=clear_cache_between_chunks,
            )
            reset_mask = torch.ones(
                (batch_size, images.shape[1], images.shape[2]),
                device=images.device,
                dtype=images.dtype,
            )
            return decoded, sampled, denoised, reset_mask

        previous_input = None
        previous_output = None
        output_images = []
        output_latents = []
        output_denoised_latents = []
        reset_values = []

        for frame_index in range(batch_size):
            current_frame = _single_frame(images, frame_index)
            should_reset = _compute_reset(
                frame_index,
                previous_input,
                current_frame,
                int(reset_every_n),
                float(scene_cut_threshold),
            )
            reset_values.append(float(should_reset))

            if should_reset:
                prior_frame = current_frame
            else:
                prior_frame, _ = _resolve_temporal_prior(
                    temporal_mode,
                    frame_index,
                    current_frame,
                    previous_input,
                    previous_output,
                    warped_previous_images,
                    flow_confidence,
                    float(flow_confidence_scale),
                    float(temporal_strength),
                    batch_size,
                )

            encoded = encode_image_batch_chunked(
                prior_frame,
                vae,
                encode_chunk_size,
                clear_cache_between_chunks=False,
            )
            sampled, denoised = sample_single_latent_temporal(
                noise,
                guider,
                sampler,
                sigmas,
                encoded,
                noise_seed_override=_seed_for_frame(noise, frame_index, bool(lock_seed), seed_stride_mode),
                forced_batch_index_value=_batch_index_for_frame(frame_index, bool(lock_seed)),
                clear_cache_between_chunks=clear_cache_between_chunks,
            )
            decoded = decode_latent_batch_chunked(
                sampled,
                vae,
                decode_chunk_size,
                clear_cache_between_chunks=False,
            )

            output_images.append(decoded)
            output_latents.append(sampled["samples"])
            output_denoised_latents.append(denoised["samples"])

            previous_input = current_frame
            previous_output = decoded

            if clear_cache_between_chunks and frame_index + 1 < batch_size:
                soft_clear_cache()

        images_out = torch.cat(output_images, dim=0)
        latents_out = {"samples": torch.cat(output_latents, dim=0)}
        denoised_out = {"samples": torch.cat(output_denoised_latents, dim=0)}
        reset_mask = torch.tensor(reset_values, device=images.device, dtype=images.dtype).view(batch_size, 1, 1)
        reset_mask = reset_mask.expand(batch_size, images.shape[1], images.shape[2])
        return images_out, latents_out, denoised_out, reset_mask


NODE_CLASS_MAPPINGS = {
    "FluxVideoCleanupTemporalAdvanced": FluxVideoCleanupTemporalAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxVideoCleanupTemporalAdvanced": "Flux Video Cleanup Temporal Advanced",
}
