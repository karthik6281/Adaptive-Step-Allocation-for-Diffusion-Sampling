"""The CIFAR encoder with an optional single-token Mamba head."""

import torch

from .policy_network_cifar import PolicyNetworkCIFAR
from .policy_network_mamba import MambaLayer


class PolicyNetworkMambaCIFAR(PolicyNetworkCIFAR):
    def init_hidden(self, batch_size, device):
        return None

    def __init__(
        self,
        input_channels=3,
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
        if hidden is not None or class_labels is not None:
            raise ValueError("This ablation is unconditional and has no recurrent hidden state.")
        features = torch.cat(
            (
                self.feature_extractor(x_t).flatten(1),
                self.time_embed(t.float()[:, None] / self.max_diffusion_steps),
            ),
            dim=-1,
        )
        encoded = self.mamba(features)
        return (
            self.skip_head(encoded).sigmoid().flatten(),
            None,
            self.value_head(encoded).flatten(),
            None,
        )
