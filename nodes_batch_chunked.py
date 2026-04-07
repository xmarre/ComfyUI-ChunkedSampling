from .core_sampling import (
    SampleChunkConfig,
    decode_latent_batch_chunked,
    encode_image_batch_chunked,
    sample_latent_batch_chunked,
)


class VAEEncodeBatchChunked:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "chunk_size": ("INT", {"default": 8, "min": 1, "max": 4096}),
                "clear_cache_between_chunks": ("BOOLEAN", {"default": False, "advanced": True}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "latent"

    def encode(self, pixels, vae, chunk_size, clear_cache_between_chunks=False):
        return (
            encode_image_batch_chunked(
                pixels,
                vae,
                chunk_size,
                clear_cache_between_chunks=clear_cache_between_chunks,
            ),
        )


class VAEDecodeBatchChunked:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "chunk_size": ("INT", {"default": 8, "min": 1, "max": 4096}),
                "clear_cache_between_chunks": ("BOOLEAN", {"default": False, "advanced": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "latent"

    def decode(self, samples, vae, chunk_size, clear_cache_between_chunks=False):
        return (
            decode_latent_batch_chunked(
                samples,
                vae,
                chunk_size,
                clear_cache_between_chunks=clear_cache_between_chunks,
            ),
        )


class SamplerCustomAdvancedChunked:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "chunk_size": ("INT", {"default": 2, "min": 1, "max": 4096}),
                "auto_reduce_on_oom": ("BOOLEAN", {"default": True}),
                "min_chunk_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "clear_cache_on_retry": ("BOOLEAN", {"default": True, "advanced": True}),
                "clear_cache_between_chunks": ("BOOLEAN", {"default": False, "advanced": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        chunk_size,
        auto_reduce_on_oom=True,
        min_chunk_size=1,
        clear_cache_on_retry=True,
        clear_cache_between_chunks=False,
    ):
        return sample_latent_batch_chunked(
            noise,
            guider,
            sampler,
            sigmas,
            latent_image,
            config=SampleChunkConfig(
                chunk_size=chunk_size,
                auto_reduce_on_oom=auto_reduce_on_oom,
                min_chunk_size=min_chunk_size,
                clear_cache_on_retry=clear_cache_on_retry,
                clear_cache_between_chunks=clear_cache_between_chunks,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "VAEEncodeBatchChunked": VAEEncodeBatchChunked,
    "VAEDecodeBatchChunked": VAEDecodeBatchChunked,
    "SamplerCustomAdvancedChunked": SamplerCustomAdvancedChunked,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEEncodeBatchChunked": "VAE Encode Batch Chunked",
    "VAEDecodeBatchChunked": "VAE Decode Batch Chunked",
    "SamplerCustomAdvancedChunked": "SamplerCustomAdvanced Chunked",
}
