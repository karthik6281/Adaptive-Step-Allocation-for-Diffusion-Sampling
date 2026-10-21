"""Adapter for the released unconditional and class-conditional CompVis LDMs."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from models.diffusion import DiffusionModel

LDM_NAMES = {
    "church256": "lsun_churches256",
    "bedroom256": "lsun_beds256",
    "celeb256": "celeba256",
    "ffhq256": "ffhq256",
    "cin256": "cin256",
}


def resolve_paths(dataset: str, config_path=None, checkpoint_path=None, compvis_root=None):
    if dataset not in LDM_NAMES:
        raise ValueError(f"Unsupported CompVis dataset: {dataset}")
    project = Path(__file__).resolve().parent
    repo = (
        Path(
            compvis_root
            or os.environ.get("TAED_COMPVIS_ROOT", project / "external/latent-diffusion")
        )
        .expanduser()
        .resolve()
    )
    checkpoint_root = Path(os.environ.get("TAED_LDM_ROOT", project / "checkpoints"))
    name = LDM_NAMES[dataset]
    config = Path(config_path) if config_path else repo / "models/ldm" / name / "config.yaml"
    checkpoint = Path(checkpoint_path) if checkpoint_path else checkpoint_root / name / "model.ckpt"
    for path in (config.expanduser(), checkpoint.expanduser()):
        if not path.is_file():
            raise FileNotFoundError(f"Required model file not found: {path}")
    return config.expanduser().resolve(), checkpoint.expanduser().resolve(), repo


class CompVisSchedulerAdapter:
    def __init__(self, model):
        self.num_train_timesteps = int(model.num_timesteps)
        self.config = SimpleNamespace(
            num_train_timesteps=self.num_train_timesteps,
            prediction_type="epsilon",
            clip_sample=False,
        )
        self.alphas_cumprod = model.alphas_cumprod.detach().cpu().float()
        self.final_alpha_cumprod = self.alphas_cumprod[0].clone()

    def set_timesteps(self, num_inference_steps: int):
        total = self.num_train_timesteps
        if not isinstance(num_inference_steps, int) or not 1 <= num_inference_steps <= total:
            raise ValueError(f"num_inference_steps must be in [1, {total}]")
        if num_inference_steps == total:
            ascending = torch.arange(total)
        else:
            # Matches CompVis for divisors of T; also gives exactly N valid nodes otherwise.
            ascending = torch.arange(num_inference_steps) * total // num_inference_steps + 1
        self.timesteps = ascending.flip(0)


class CompVisUNetAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        conditional = getattr(model, "cond_stage_model", None)
        if conditional is not None and getattr(model, "cond_stage_key", None) != "class_label":
            raise ValueError("Only unconditional and class-label CompVis models are supported.")
        classes = int(conditional.embedding.num_embeddings) if conditional is not None else None
        self.config = SimpleNamespace(
            in_channels=int(model.channels),
            sample_size=int(model.image_size),
            num_class_embeds=classes,
        )

    def forward(self, sample, timestep, class_labels=None):
        conditioning = None
        if self.config.num_class_embeds is not None:
            if class_labels is None or class_labels.shape != (sample.shape[0],):
                raise ValueError("A class label is required for every sample.")
            if torch.any(class_labels < 0) or torch.any(
                class_labels >= self.config.num_class_embeds
            ):
                raise ValueError("Class labels are outside the model's embedding range.")
            key = self.model.cond_stage_model.key
            conditioning = self.model.get_learned_conditioning({key: class_labels})
        return SimpleNamespace(sample=self.model.apply_model(sample, timestep, conditioning))


class CompVisAdapter(DiffusionModel):
    def __init__(
        self,
        config_path,
        ckpt_path,
        device="cpu",
        n_timesteps=200,
        compvis_root=None,
        trust_checkpoint=False,
    ):
        from omegaconf import OmegaConf

        repo = (
            Path(
                compvis_root
                or os.environ.get(
                    "TAED_COMPVIS_ROOT",
                    Path(__file__).resolve().parent / "external/latent-diffusion",
                )
            )
            .expanduser()
            .resolve()
        )
        if not (repo / "ldm/util.py").is_file():
            raise FileNotFoundError(f"CompVis source is missing at {repo}; set TAED_COMPVIS_ROOT.")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        util = importlib.import_module("ldm.util")
        if not Path(util.__file__).resolve().is_relative_to(repo):
            raise ImportError("A different ldm package is already imported; use a fresh process.")

        config = OmegaConf.load(config_path)
        params = config.model.params
        params.pop("ckpt_path", None)
        first_stage = params.get("first_stage_config", {}).get("params", {})
        first_stage.pop("ckpt_path", None)
        model = util.instantiate_from_config(config.model)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=not trust_checkpoint)
        state = payload.get("state_dict", payload)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint/config mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        model.eval().requires_grad_(False)
        super().__init__(CompVisUNetAdapter(model), CompVisSchedulerAdapter(model), n_timesteps)
        self.latent_channels, self.latent_size, _ = self.sample_shape
        self.to(device)

    @property
    def model(self):
        return self.unet.model

    @torch.no_grad()
    def decode(self, sample):
        return self.model.decode_first_stage(sample)

    def _decode_latents_to_image(self, latents):
        return self.decode(latents)

    @torch.no_grad()
    def _encode_image_to_latents(self, images):
        return self.model.get_first_stage_encoding(self.model.encode_first_stage(images))

    def _denoise_step_latents(self, sample, start, end, eta=0.0, class_labels=None):
        if eta != 0:
            raise ValueError("This adapter supports deterministic DDIM only (eta=0).")
        return self.transition(sample, start, end, class_labels)


def create_compvis_adapter(config_path, ckpt_path, device="cpu", n_timesteps=200, **kwargs):
    return CompVisAdapter(config_path, ckpt_path, device, n_timesteps, **kwargs)
