import socket

import pytest
import torch
from torch import nn

from dasp.adapters.policy_adapter import DASPPolicyAdapter
from models.diffusion import create_toy_diffusion
from models.policy_network import PolicyNetwork
from training.advanced_reward import AdvancedRewardV3
from utils.reproducibility import seed_everything


class SquaredDistance(nn.Module):
    def forward(self, first, second):
        return (first - second).square().mean((1, 2, 3), keepdim=True)


@pytest.fixture(autouse=True)
def deterministic_tests(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def deny_network(*args, **kwargs):
        raise AssertionError("Tests must not open network connections.")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    torch.set_num_threads(1)
    seed_everything(17)


@pytest.fixture
def model():
    return create_toy_diffusion(6)


@pytest.fixture
def policy_factory():
    def create(kind="lstm", classes=None):
        if kind == "lstm":
            return PolicyNetwork(
                1, hidden_dim=8, feature_dim=8, max_diffusion_steps=100, num_classes=classes
            )
        return DASPPolicyAdapter(
            dataset="toy",
            head=kind,
            diffusion_policy_config={
                "action_horizon": 4,
                "planning_interval": 2,
                "obs_feature_dim": 8,
                "n_layer": 1,
                "n_head": 2,
                "n_emb": 8,
                "n_cond_layers": 0,
                "num_inference_steps": 4,
                "max_diffusion_steps": 100,
                "num_classes": classes,
                "class_embedding_dim": 4,
                "budget_embed_dim": 4,
                "horizon_embed_dim": 4,
            },
            encoder_config={
                "input_channels": 1,
                "image_size": 8,
                "channels": [4, 8],
                "use_batch_norm": False,
                "dropout_rate": 0.0,
                "output_dim": 8,
            },
        )

    return create


@pytest.fixture
def reward():
    return AdvancedRewardV3(w_color=0.001, w_cost=0.005)


@pytest.fixture
def perceptual_reward():
    return AdvancedRewardV3(w_perceptual=1.0, perceptual_model=SquaredDistance())
