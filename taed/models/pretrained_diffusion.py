"""Load a local or revision-pinned Diffusers image model."""

from __future__ import annotations

import re
from pathlib import Path

from diffusers import DDIMScheduler, DiffusionPipeline

from .diffusion import DiffusionModel


class LatentDiffusersModel(DiffusionModel):
    def __init__(self, pipeline, n_timesteps):
        super().__init__(
            pipeline.unet, DDIMScheduler.from_config(pipeline.scheduler.config), n_timesteps
        )
        self.autoencoder = pipeline.vqvae
        self.latent_channels, self.latent_size, _ = self.sample_shape

    def decode(self, sample):
        result = self.autoencoder.decode(sample)
        return result.sample if hasattr(result, "sample") else result

    def _decode_latents_to_image(self, sample):
        return self.decode(sample)


def create_diffusers_model(model_id, revision=None, device="cpu", n_timesteps=200):
    if not model_id:
        raise ValueError("model_id must name a local directory or a Hugging Face model.")
    if not Path(model_id).is_dir() and (
        not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
    ):
        raise ValueError("Remote models require an immutable 40-character commit revision.")
    pipeline = DiffusionPipeline.from_pretrained(model_id, revision=revision)
    if not hasattr(pipeline, "unet") or not hasattr(pipeline, "scheduler"):
        raise ValueError("The pipeline must expose an image UNet and a scheduler.")
    if hasattr(pipeline, "text_encoder"):
        raise ValueError("Text-conditioned pipelines are not supported.")
    if hasattr(pipeline, "vqvae"):
        model = LatentDiffusersModel(pipeline, n_timesteps)
    else:
        model = DiffusionModel(
            pipeline.unet, DDIMScheduler.from_config(pipeline.scheduler.config), n_timesteps
        )
    return model.to(device).eval().requires_grad_(False)
