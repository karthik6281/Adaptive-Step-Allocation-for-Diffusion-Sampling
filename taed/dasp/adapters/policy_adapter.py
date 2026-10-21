"""Expose the action-plan policies through a common trainer interface."""

from __future__ import annotations

import copy
import importlib

from torch import nn

from dasp.policy.diffusion_policy import DiffusionPolicy
from dasp.policy.direct_plan_policy import DirectPlanPolicy

DATASETS = ("cifar10", "celeb64", "celeb256", "bedroom256", "church256", "ffhq256", "cin256")


class DASPPolicyAdapter(nn.Module):
    def __init__(
        self,
        dataset="cifar10",
        diffusion_policy_config=None,
        encoder_config=None,
        planning_mode="multi",
        head="diffusion",
        **overrides,
    ):
        super().__init__()
        if planning_mode not in ("single", "multi") or head not in ("diffusion", "direct"):
            raise ValueError("Invalid planning mode or policy head.")
        if diffusion_policy_config is None:
            if dataset not in DATASETS:
                raise ValueError(f"No default policy configuration for {dataset}.")
            module = importlib.import_module(f"dasp.config.config_{dataset}")
            config = copy.deepcopy(module.DIFFUSION_POLICY_CONFIG)
            if encoder_config is None:
                encoder_config = copy.deepcopy(module.ENCODER_CONFIG)
        else:
            config = copy.deepcopy(diffusion_policy_config)
        config.update(overrides)
        config["dataset"] = dataset
        config["encoder_config"] = copy.deepcopy(encoder_config)
        constructor = DirectPlanPolicy if head == "direct" else DiffusionPolicy
        self.policy = constructor(**config)
        self.planning_mode = planning_mode
        self.action_horizon = self.policy.action_horizon
        self.planning_interval = self.policy.planning_interval

    @property
    def encoder(self):
        return self.policy.visual_encoder

    def get_planning_horizon(self):
        return self.planning_interval if self.planning_mode == "multi" else 1

    def plan(self, observations):
        return self.policy.plan(observations)

    def plan_actions(
        self,
        image,
        timestep,
        hidden=None,
        class_labels=None,
        budget_ratio=None,
        jump_ratio=None,
        progress_ratio=None,
        rate_ratio=None,
    ):
        observations = {"image": image, "timestep": timestep}
        for key, value in (
            ("class_labels", class_labels),
            ("budget_ratio", budget_ratio),
            ("jump_ratio", jump_ratio),
            ("progress_ratio", progress_ratio),
            ("rate_ratio", rate_ratio),
        ):
            if value is not None:
                observations[key] = value
        probabilities, values = self.plan(observations)
        return probabilities, values[:, None].expand_as(probabilities), hidden

    def forward(self, image, timestep, hidden=None, class_labels=None, **kwargs):
        probabilities, values, hidden = self.plan_actions(
            image, timestep, hidden, class_labels, **kwargs
        )
        return probabilities[:, 0], hidden, values[:, 0], None

    def init_hidden(self, batch_size, device):
        return None


def create_dasp_policy_for_dataset(dataset="cifar10", **kwargs):
    return DASPPolicyAdapter(dataset=dataset, **kwargs)
