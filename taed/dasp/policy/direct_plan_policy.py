"""A deterministic multi-action head using the same observations as DASP."""

from __future__ import annotations

import math

import torch
from torch import nn

from .diffusion_policy import LOGIT_SCALE, DiffusionPolicy


class DirectPlanPolicy(DiffusionPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n_emb, n_head = kwargs.get("n_emb", 256), kwargs.get("n_head", 8)
        self.cond_norm = nn.LayerNorm(self.cond_dim)
        self.cond_proj = nn.Linear(self.cond_dim, n_emb)
        self.slot_emb = nn.Parameter(torch.empty(1, self.action_horizon, n_emb))
        nn.init.normal_(self.slot_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=kwargs.get("p_drop_attn", 0.1),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.plan_encoder = nn.TransformerEncoder(
            layer, num_layers=kwargs.get("n_layer", 6), enable_nested_tensor=False
        )
        self.plan_ln = nn.LayerNorm(n_emb)
        self.plan_head = nn.Linear(n_emb, 1)
        nn.init.normal_(self.plan_head.weight, std=1 / math.sqrt(n_emb))
        nn.init.zeros_(self.plan_head.bias)
        self.transformer.requires_grad_(False)

    def predict_action(self, obs_dict):
        condition = self.cond_proj(self.cond_norm(self._condition(obs_dict))).unsqueeze(1)
        tokens = self.plan_encoder(self.slot_emb + condition)
        logits = self.plan_head(self.plan_ln(tokens)).squeeze(-1)
        probabilities = torch.sigmoid(LOGIT_SCALE * torch.tanh(logits / LOGIT_SCALE))
        return {"action": probabilities, "action_prob": probabilities}

    @torch.no_grad()
    def calibrate_dc(self, obs_dict):
        offset = self.cond_proj(self.cond_norm(self._condition(obs_dict))).mean(0)
        self.cond_proj.bias.sub_(offset)
        return float(offset.abs().mean())

    def compute_loss(self, batch):
        raise ValueError(
            "The direct head has no diffusion auxiliary loss; use diffusion_loss_weight=0."
        )
