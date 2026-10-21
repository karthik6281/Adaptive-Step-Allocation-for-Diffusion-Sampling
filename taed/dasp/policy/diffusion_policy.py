"""Bounded-logit action diffusion conditioned on image state and remaining budget."""

from __future__ import annotations

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from torch import nn
from torch.nn import functional as F

from dasp.diffusion_transformer.module_attr_mixin import ModuleAttrMixin
from dasp.diffusion_transformer.transformer_for_diffusion import TransformerForDiffusion
from dasp.encoder.custom_encoder import create_taed_visual_encoder
from models.diffusion import ddim_transition
from utils.reproducibility import require_finite

LOGIT_SCALE = 5.0
HORIZON_JUMP_SCALE = 8.0


def prob_to_bounded_logit(p, eps=1e-5, scale=LOGIT_SCALE):
    p = p.clamp(eps, 1 - eps)
    return scale * torch.tanh(torch.logit(p) / scale)


def bounded_logit_to_prob(logit, scale=LOGIT_SCALE):
    return torch.sigmoid(logit)


def bound_trajectory(trajectory, scale=LOGIT_SCALE):
    return scale * torch.tanh(trajectory / scale)


class DiffusionPolicy(ModuleAttrMixin):
    def __init__(
        self,
        dataset="cifar10",
        action_horizon=20,
        planning_interval=10,
        n_obs_steps=1,
        num_inference_steps=10,
        scheduler_type="DDPM",
        obs_feature_dim=256,
        n_layer=6,
        n_head=8,
        n_emb=256,
        p_drop_emb=0.0,
        p_drop_attn=0.1,
        use_ema=False,
        ema_decay=0.995,
        causal_attn=False,
        time_as_cond=True,
        obs_as_cond=True,
        n_cond_layers=2,
        max_diffusion_steps=1000,
        num_classes=None,
        class_embedding_dim=16,
        debug_mode=False,
        budget_aware=True,
        budget_embed_dim=16,
        horizon_aware=True,
        horizon_embed_dim=16,
        encoder_config=None,
        learning_rate=1e-4,
        weight_decay=1e-3,
        batch_size=32,
        prediction_type="epsilon",
        loss_reduction="mean",
    ):
        super().__init__()
        if not 1 <= planning_interval <= action_horizon:
            raise ValueError("Require 1 <= planning_interval <= action_horizon.")
        if obs_feature_dim < 4 or max_diffusion_steps < 1 or not 2 <= num_inference_steps <= 100:
            raise ValueError("Invalid observation dimension, timestep count, or inference steps.")
        if n_obs_steps != 1 or not time_as_cond:
            raise ValueError("TAED policies require one observation and time conditioning.")
        if use_ema:
            raise ValueError("EMA is not supported; use the trained policy weights directly.")
        if scheduler_type != "DDPM" or prediction_type != "epsilon" or loss_reduction != "mean":
            raise ValueError("Action diffusion uses DDPM epsilon training with mean loss.")
        self.dataset = dataset
        self.action_horizon = action_horizon
        self.planning_interval = planning_interval
        self.num_inference_steps = num_inference_steps
        self.obs_feature_dim = obs_feature_dim
        self.max_diffusion_steps = max_diffusion_steps
        self.budget_aware = budget_aware
        self.horizon_aware = horizon_aware
        self.is_conditional = num_classes is not None
        self.num_classes = num_classes
        self.obs_as_cond = obs_as_cond
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.visual_encoder = create_taed_visual_encoder(
            dataset, config=encoder_config, output_dim=obs_feature_dim
        )
        time_dim = obs_feature_dim // 4
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )
        self.class_embedding = (
            nn.Embedding(num_classes, class_embedding_dim) if self.is_conditional else None
        )
        cond_dim = obs_feature_dim + time_dim + (class_embedding_dim if self.is_conditional else 0)
        self.budget_embedding = None
        if budget_aware:
            self.budget_embedding = nn.Sequential(
                nn.Linear(1, budget_embed_dim),
                nn.GELU(),
                nn.Linear(budget_embed_dim, budget_embed_dim),
            )
            cond_dim += budget_embed_dim
        self.horizon_embedding = None
        if horizon_aware:
            self.horizon_embedding = nn.Sequential(
                nn.Linear(3, horizon_embed_dim),
                nn.GELU(),
                nn.Linear(horizon_embed_dim, horizon_embed_dim),
            )
            cond_dim += horizon_embed_dim
        self.cond_dim = cond_dim
        input_dim = 1 if obs_as_cond else 1 + cond_dim
        self.transformer = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=input_dim,
            horizon=action_horizon,
            n_obs_steps=1,
            cond_dim=cond_dim if obs_as_cond else 0,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=True,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
        )
        scheduler_config = {
            "num_train_timesteps": 100,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "epsilon",
            "clip_sample": False,
            "timestep_spacing": "linspace",
        }
        self.noise_scheduler = DDPMScheduler(**scheduler_config)
        self.inference_scheduler = DDIMScheduler(**scheduler_config)
        self.value_head = nn.Sequential(
            nn.Linear(cond_dim, obs_feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(obs_feature_dim, obs_feature_dim // 2),
            nn.GELU(),
            nn.Linear(obs_feature_dim // 2, 1),
        )

    def encode_state(self, obs_dict, class_labels=None, budget_ratio=None):
        image, timestep = obs_dict["image"], obs_dict["timestep"]
        batch = image.shape[0]
        if timestep.shape != (batch,):
            raise ValueError("timestep must have shape [batch]")
        features = [
            self.visual_encoder(image),
            self.time_embedding(timestep.float().unsqueeze(-1) / self.max_diffusion_steps),
        ]
        if self.is_conditional:
            if class_labels is None or class_labels.shape != (batch,):
                raise ValueError("A class label is required for every policy observation.")
            if torch.any(class_labels < 0) or torch.any(class_labels >= self.num_classes):
                raise ValueError("Policy class labels are outside the embedding range.")
            features.append(self.class_embedding(class_labels))
        elif class_labels is not None:
            raise ValueError("This policy is unconditional.")
        if self.budget_embedding is not None:
            budget = budget_ratio if budget_ratio is not None else image.new_ones(batch)
            features.append(self.budget_embedding(budget.float().unsqueeze(-1)))
        if self.horizon_embedding is not None:
            jump = obs_dict.get("jump_ratio", image.new_zeros(batch))
            progress = obs_dict.get("progress_ratio", image.new_zeros(batch))
            rate = obs_dict.get("rate_ratio", image.new_zeros(batch))
            stride = max(1, int(getattr(self, "timestep_stride", 1)))
            jump = jump * self.max_diffusion_steps / (stride * HORIZON_JUMP_SCALE)
            features.append(
                self.horizon_embedding(torch.stack((jump, progress, rate), dim=-1).float())
            )
        return require_finite(torch.cat(features, dim=-1), "policy conditioning")

    def _condition(self, observations):
        return self.encode_state(
            observations, observations.get("class_labels"), observations.get("budget_ratio")
        )

    def _predict_epsilon(self, trajectory, timestep, condition):
        if self.obs_as_cond:
            output = self.transformer(trajectory, timestep, condition.unsqueeze(1))
        else:
            repeated = condition.unsqueeze(1).expand(-1, trajectory.shape[1], -1)
            output = self.transformer(torch.cat((trajectory, repeated), dim=-1), timestep)[..., :1]
        return bound_trajectory(output)

    def predict_action(self, obs_dict):
        condition = self._condition(obs_dict)
        trajectory = torch.randn(
            (condition.shape[0], self.action_horizon, 1),
            device=condition.device,
            dtype=condition.dtype,
            generator=obs_dict.get("generator"),
        ) * (LOGIT_SCALE / 3)
        trajectory = bound_trajectory(trajectory)
        scheduler = self.inference_scheduler
        scheduler.set_timesteps(self.num_inference_steps, device=condition.device)
        alphas = scheduler.alphas_cumprod.to(condition.device)
        ends = torch.cat((scheduler.timesteps[1:], scheduler.timesteps.new_tensor([-1])))
        for timestep, end in zip(scheduler.timesteps, ends):
            prediction = self._predict_epsilon(trajectory, timestep, condition)
            alpha_end = alphas[end] if end >= 0 else alphas.new_tensor(1.0)
            trajectory = bound_trajectory(
                ddim_transition(trajectory, prediction, alphas[timestep], alpha_end)[0]
            )
        probabilities = require_finite(
            torch.sigmoid(trajectory.squeeze(-1)), "action probabilities"
        )
        return {"action": probabilities, "action_prob": probabilities}

    def plan(self, observations):
        probabilities = self.predict_action(observations)["action_prob"]
        values = self.value_head(self._condition(observations)).squeeze(-1)
        return probabilities, values

    def compute_loss(self, batch):
        observations, actions = batch["observations"], batch["actions"]
        if actions.ndim != 2 or actions.shape[1] != self.action_horizon:
            raise ValueError("actions must have shape [batch, action_horizon]")
        if not torch.all((actions >= 0) & (actions <= 1)):
            raise ValueError("actions must be probabilities or binary values")
        mask = batch.get("mask", torch.ones_like(actions, dtype=torch.bool))
        if mask.shape != actions.shape or mask.dtype != torch.bool or not mask.any():
            raise ValueError("mask must match actions and contain a valid action")
        last_valid = int(mask.any(0).nonzero()[-1]) + 1
        actions, mask = actions[:, :last_valid], mask[:, :last_valid]
        trajectory = prob_to_bounded_logit(actions.float()).unsqueeze(-1)
        noise = torch.randn_like(trajectory) * (LOGIT_SCALE / 3)
        timestep = torch.randint(100, (actions.shape[0],), device=actions.device)
        noisy = bound_trajectory(self.noise_scheduler.add_noise(trajectory, noise, timestep))
        prediction = self._predict_epsilon(noisy, timestep, self._condition(observations))
        loss = F.mse_loss(prediction, noise, reduction="none").squeeze(-1)
        return loss[mask].mean()

    def get_optimizer(self, learning_rate=None, weight_decay=None):
        return torch.optim.AdamW(
            [
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad and parameter.numel()
            ],
            lr=self.learning_rate if learning_rate is None else learning_rate,
            weight_decay=self.weight_decay if weight_decay is None else weight_decay,
        )
