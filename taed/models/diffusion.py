"""Shared deterministic DDIM transitions for training and evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from utils.reproducibility import require_finite


def ddim_transition(
    sample: torch.Tensor,
    prediction: torch.Tensor,
    alpha_start: torch.Tensor,
    alpha_end: torch.Tensor,
    prediction_type: str = "epsilon",
    clip_sample: bool = False,
    clip_sample_range: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sample.shape != prediction.shape:
        raise ValueError("The model prediction must have the same shape as the sample.")
    if not torch.all((alpha_start > 0) & (alpha_start <= alpha_end) & (alpha_end <= 1)):
        raise ValueError("DDIM transitions require 0 < alpha_start <= alpha_end <= 1.")
    shape = (-1,) + (1,) * (sample.ndim - 1)
    start, end = alpha_start.reshape(shape), alpha_end.reshape(shape)
    if prediction_type == "epsilon":
        epsilon = prediction
        clean = (sample - (1 - start).sqrt() * epsilon) / start.sqrt()
    elif prediction_type == "v_prediction":
        clean = start.sqrt() * sample - (1 - start).sqrt() * prediction
        epsilon = start.sqrt() * prediction + (1 - start).sqrt() * sample
    elif prediction_type == "sample":
        clean = prediction
        epsilon = (sample - start.sqrt() * clean) / (1 - start).sqrt().clamp_min(1e-12)
    else:
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")
    if clip_sample:
        clean = clean.clamp(-clip_sample_range, clip_sample_range)
    result = end.sqrt() * clean + (1 - end).sqrt() * epsilon
    return require_finite(result, "DDIM result"), epsilon, clean


class DiffusionModel(nn.Module):
    def __init__(self, unet: nn.Module, scheduler, n_timesteps: int = 50):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.num_timesteps = int(scheduler.config.num_train_timesteps)
        self.register_buffer("alphas", scheduler.alphas_cumprod.detach().float().clone())
        terminal = getattr(scheduler, "final_alpha_cumprod", torch.tensor(1.0))
        self.register_buffer("terminal_alpha", torch.as_tensor(terminal).float().clone())
        size = unet.config.sample_size
        spatial = (size, size) if isinstance(size, int) else tuple(size)
        self.sample_shape = (int(unet.config.in_channels), *spatial)
        self.num_classes = getattr(unet.config, "num_class_embeds", None)
        self.set_timesteps(n_timesteps)

    @property
    def device(self) -> torch.device:
        return self.alphas.device

    def set_timesteps(self, steps: int) -> torch.Tensor:
        if not isinstance(steps, int) or not 1 <= steps <= self.num_timesteps:
            raise ValueError(f"steps must be an integer in [1, {self.num_timesteps}]")
        self.scheduler.set_timesteps(steps)
        grid = self.scheduler.timesteps.to(self.device)
        if (
            grid.ndim != 1
            or grid.numel() != steps
            or torch.any(grid < 0)
            or torch.any(grid >= self.num_timesteps)
            or torch.any(grid[1:] >= grid[:-1])
        ):
            raise ValueError("The scheduler did not produce a valid descending grid.")
        self.n_timesteps = steps
        return grid

    def predict_noise(self, sample, timestep, class_labels=None):
        if self.num_classes is not None and class_labels is None:
            raise ValueError("This diffusion model requires class labels.")
        if self.num_classes is None and class_labels is not None:
            raise ValueError("Class labels were supplied to an unconditional diffusion model.")
        kwargs = {} if class_labels is None else {"class_labels": class_labels}
        output = self.unet(sample, timestep, **kwargs)
        return output.sample if hasattr(output, "sample") else output

    @torch.no_grad()
    def transition(self, sample, start, end, class_labels=None):
        if start.shape != (sample.shape[0],) or end.shape != start.shape:
            raise ValueError("start and end must have shape [batch]")
        if torch.any(start < 0) or torch.any(start >= self.num_timesteps):
            raise ValueError("Invalid start timestep.")
        if torch.any(end < -1) or torch.any(end >= start):
            raise ValueError("Each destination must precede its source timestep.")
        config = self.scheduler.config
        if getattr(config, "thresholding", False):
            raise ValueError("Dynamic thresholding is not supported by this adapter.")
        end_alpha = self.alphas[end.clamp_min(0)]
        end_alpha = torch.where(end < 0, self.terminal_alpha, end_alpha)
        return ddim_transition(
            sample,
            self.predict_noise(sample, start, class_labels),
            self.alphas[start],
            end_alpha,
            prediction_type=config.prediction_type,
            clip_sample=getattr(config, "clip_sample", False),
            clip_sample_range=getattr(config, "clip_sample_range", 1.0),
        )

    def decode(self, sample: torch.Tensor) -> torch.Tensor:
        return sample

    @torch.no_grad()
    def sample(self, initial_noise: torch.Tensor, steps: int, class_labels=None):
        grid = self.set_timesteps(steps)
        state = initial_noise.clone().to(self.device)
        if tuple(state.shape[1:]) != self.sample_shape:
            raise ValueError(f"Expected initial noise of shape [batch, {self.sample_shape}].")
        destinations = torch.cat((grid[1:], grid.new_tensor([-1])))
        for start, end in zip(grid, destinations):
            state, _, _ = self.transition(
                state, start.expand(state.shape[0]), end.expand(state.shape[0]), class_labels
            )
        return self.decode(state)


class ToyDenoiser(nn.Module):
    """Small analytic denoiser for offline plumbing tests, not an image model."""

    def __init__(self, channels: int = 1, size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(in_channels=channels, sample_size=size, num_class_embeds=None)

    def forward(self, sample, timestep):
        time = timestep.float().reshape(-1, 1, 1, 1) / 100
        return SimpleNamespace(sample=0.15 * sample.tanh() + 0.05 * time)


def create_toy_diffusion(steps: int = 8) -> DiffusionModel:
    from diffusers import DDIMScheduler

    return DiffusionModel(
        ToyDenoiser(),
        DDIMScheduler(num_train_timesteps=100, clip_sample=False, timestep_spacing="linspace"),
        n_timesteps=steps,
    )
