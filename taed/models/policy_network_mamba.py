"""Optional single-token Mamba ablation; this variant has no recurrent memory."""

import torch
from torch import nn

from .policy_network import PolicyNetwork


class MambaLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_state=16, d_expand=2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as error:
            raise ImportError(
                "Mamba requires its optional Linux/CUDA installation. No substitute network is used."
            ) from error
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim, d_state=d_state, d_conv=4, expand=d_expand)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features):
        tokens = self.input_proj(features).unsqueeze(1)
        return self.output_proj(self.mamba(tokens).squeeze(1))


class PolicyNetworkMamba(PolicyNetwork):
    def init_hidden(self, batch_size, device):
        return None

    def __init__(
        self,
        input_channels,
        hidden_dim=64,
        feature_dim=32,
        num_classes=None,
        class_embedding_dim=16,
        max_diffusion_steps=1000,
    ):
        super().__init__(
            input_channels,
            hidden_dim,
            feature_dim,
            num_classes,
            class_embedding_dim,
            max_diffusion_steps,
        )
        self.mamba = MambaLayer(self.lstm.input_size, hidden_dim)
        del self.lstm

    def forward(self, x_t, t, hidden=None, class_labels=None):
        if hidden is not None:
            raise ValueError("The single-token Mamba ablation has no recurrent hidden state.")
        features = [
            self.feature_extractor(x_t).flatten(1),
            self.time_embed(t.float()[:, None] / self.max_diffusion_steps),
        ]
        if self.is_conditional:
            if class_labels is None or class_labels.dtype != torch.long:
                raise ValueError("Conditional Mamba requires integer class labels.")
            features.append(self.class_embed(class_labels))
        elif class_labels is not None:
            raise ValueError("This Mamba policy is unconditional.")
        encoded = self.mamba(torch.cat(features, dim=-1))
        return (
            self.skip_head(encoded).sigmoid().flatten(),
            None,
            self.value_head(encoded).flatten(),
            None,
        )
