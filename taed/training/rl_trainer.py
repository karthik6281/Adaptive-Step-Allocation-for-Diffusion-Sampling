"""A2C, PPO, and grouped policy-gradient training on shared diffusion rollouts."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from training.advanced_reward import validate_reward_schedule
from training.budget import validate_rollout_settings
from training.rollout import collect_rollout, observation_to, replay_policy
from utils.reproducibility import (
    provenance,
    require_finite,
    require_int,
    restore_rng_state,
    rng_state,
    save_checkpoint,
)


def compute_gae(rewards, values, gamma=1.0, gae_lambda=0.95, dones=None, bootstrap=None):
    if rewards.ndim != 2 or rewards.shape != values.shape or rewards.shape[0] < 1:
        raise ValueError("rewards and values must have matching nonempty [time, batch] shapes.")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    rewards, values = rewards.detach(), values.detach()
    require_finite(rewards, "GAE rewards")
    require_finite(values, "GAE values")
    if dones is None:
        dones = torch.zeros_like(rewards, dtype=torch.bool)
    if dones.shape != rewards.shape or dones.dtype != torch.bool:
        raise ValueError("dones must be a Boolean tensor matching rewards.")
    next_value = torch.zeros_like(rewards[0]) if bootstrap is None else bootstrap.detach()
    if next_value.shape != rewards.shape[1:]:
        raise ValueError("bootstrap must have shape [batch].")
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for step in reversed(range(rewards.shape[0])):
        alive = (~dones[step]).to(rewards.dtype)
        delta = rewards[step] + gamma * alive * next_value - values[step]
        last = delta + gamma * gae_lambda * alive * last
        advantages[step] = last
        next_value = values[step]
    return advantages, advantages + values


def masked_mean(values, mask):
    if values.shape != mask.shape:
        raise ValueError("The reduction mask must match the values.")
    if not mask.any():
        return values.new_zeros(())
    return values[mask].mean()


def normalized_advantages(advantages, mask):
    selected = advantages[mask]
    if selected.numel() < 2 or selected.std(unbiased=False) < 1e-8:
        return advantages
    return (advantages - selected.mean()) / selected.std(unbiased=False)


def grouped_advantages(returns, group_size):
    if returns.ndim != 1 or group_size < 2 or returns.numel() % group_size:
        raise ValueError("GRPO requires at least two trajectories per group and a divisible batch.")
    groups = returns.reshape(-1, group_size)
    centered = groups - groups.mean(1, keepdim=True)
    return (centered / groups.std(1, keepdim=True, unbiased=False).clamp_min(1e-8)).reshape_as(
        returns
    )


def warmup_ratio_at(episode, fixed=0.0, initial=None, final=0.0, anneal_episodes=0):
    if initial is None or anneal_episodes == 0:
        return fixed
    progress = min(1.0, episode / anneal_episodes)
    return initial + progress * (final - initial)


class RLTrainer:
    def __init__(
        self,
        diffusion_model,
        policy_network,
        reward_function,
        device="cpu",
        lr=1e-4,
        weight_decay=None,
        gamma=1.0,
        entropy_weight=0.01,
        algo="a2c_gae",
        gae_lambda=0.95,
        ppo_epochs=2,
        ppo_clip_ratio=0.2,
        grpo_group_size=4,
        grpo_beta=0.01,
        num_episodes=100,
        inference_steps=200,
        reference_steps=None,
        denoising_budget=None,
        budget_mode="cap",
        max_consecutive_skips=None,
        warmup_ratio=0.0,
        warmup_initial_ratio=None,
        warmup_final_ratio=0.0,
        warmup_anneal_episodes=0,
        value_lr_scale=1.0,
        diffusion_loss_weight=0.0,
        max_grad_norm=1.0,
        truncation_steps=None,
        experiment=None,
        reward_schedule=None,
    ):
        if weight_decay is None:
            weight_decay = 1e-3 if hasattr(policy_network, "policy") else 0.0
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if algo not in ("a2c_simple", "a2c_gae", "ppo", "grpo"):
            raise ValueError(f"Unknown training algorithm: {algo}")
        if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must lie in [0, 1]")
        if (
            min(lr, value_lr_scale, max_grad_norm) <= 0
            or min(entropy_weight, diffusion_loss_weight, grpo_beta) < 0
        ):
            raise ValueError("Invalid learning rate, gradient limit, or loss coefficient.")
        if not all(
            math.isfinite(value)
            for value in (
                lr,
                value_lr_scale,
                max_grad_norm,
                entropy_weight,
                diffusion_loss_weight,
                grpo_beta,
            )
        ):
            raise ValueError("Training coefficients must be finite.")
        for name, value in (
            ("num_episodes", num_episodes),
            ("ppo_epochs", ppo_epochs),
            ("grpo_group_size", grpo_group_size),
        ):
            require_int(value, name)
        require_int(inference_steps, "inference_steps", maximum=diffusion_model.num_timesteps)
        reference_steps = inference_steps if reference_steps is None else reference_steps
        require_int(reference_steps, "reference_steps", maximum=diffusion_model.num_timesteps)
        require_int(warmup_anneal_episodes, "warmup_anneal_episodes", minimum=0)
        if not 0 < ppo_clip_ratio < 1:
            raise ValueError("Invalid PPO clipping ratio.")
        if algo == "grpo" and grpo_group_size < 2:
            raise ValueError("GRPO requires grpo_group_size >= 2.")
        for value in (warmup_ratio, warmup_final_ratio, warmup_initial_ratio):
            if value is not None and not 0 <= value <= 1:
                raise ValueError("Warmup ratios must lie in [0, 1].")
        ratios = (
            (warmup_ratio,)
            if warmup_initial_ratio is None or warmup_anneal_episodes == 0
            else (warmup_initial_ratio, warmup_final_ratio)
        )
        for ratio in ratios:
            validate_rollout_settings(
                inference_steps, denoising_budget, budget_mode, ratio, max_consecutive_skips
            )
        reward_schedule = [] if reward_schedule is None else reward_schedule
        validate_reward_schedule(reward_schedule, num_episodes)
        for stage in reward_schedule:
            for name, assets in (
                ("w_perceptual", ("lpips_model",)),
                ("w_object", ("feature_extractor",)),
                ("w_manifold", ("_manifold_enc", "_manifold_bank")),
            ):
                if stage["weights"].get(name, 0) and any(
                    getattr(reward_function, asset, None) is None for asset in assets
                ):
                    raise ValueError(
                        f"Scheduled {name} requires initialized reward assets; use build_experiment "
                        "or initialize those components before constructing the trainer."
                    )
        if diffusion_loss_weight and not hasattr(policy_network, "policy"):
            raise ValueError("Diffusion auxiliary loss requires a diffusion action policy.")
        if diffusion_loss_weight:
            from dasp.policy.direct_plan_policy import DirectPlanPolicy

            if isinstance(policy_network.policy, DirectPlanPolicy):
                raise ValueError("The direct head requires diffusion_loss_weight=0.")
        if truncation_steps is not None:
            require_int(truncation_steps, "truncation_steps")
        self.device = torch.device(device)
        self.diffusion_model = diffusion_model.to(self.device).eval().requires_grad_(False)
        self.policy_network = policy_network.to(self.device)
        self.reward_function = reward_function
        self.algo, self.gamma, self.gae_lambda = algo, gamma, gae_lambda
        self.entropy_weight, self.diffusion_loss_weight = entropy_weight, diffusion_loss_weight
        self.ppo_epochs, self.ppo_clip_ratio = ppo_epochs, ppo_clip_ratio
        self.grpo_group_size, self.grpo_beta = grpo_group_size, grpo_beta
        self.inference_steps, self.reference_steps = inference_steps, reference_steps
        self.denoising_budget, self.budget_mode = denoising_budget, budget_mode
        self.max_consecutive_skips = max_consecutive_skips
        self.warmup_ratio, self.warmup_initial_ratio = warmup_ratio, warmup_initial_ratio
        self.warmup_final_ratio, self.warmup_anneal_episodes = (
            warmup_final_ratio,
            warmup_anneal_episodes,
        )
        self.max_grad_norm, self.truncation_steps = max_grad_norm, truncation_steps
        self.experiment = copy.deepcopy(experiment or {})
        self.reward_schedule = copy.deepcopy(reward_schedule)
        self.num_episodes = num_episodes
        self.episode = 0
        self.history = []
        self.settings = {
            "algo": algo,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "lr": lr,
            "weight_decay": weight_decay,
            "num_episodes": num_episodes,
            "inference_steps": inference_steps,
            "reference_steps": self.reference_steps,
            "budget": denoising_budget,
            "budget_mode": budget_mode,
            "entropy_weight": entropy_weight,
            "diffusion_loss_weight": diffusion_loss_weight,
            "ppo_epochs": ppo_epochs,
            "ppo_clip_ratio": ppo_clip_ratio,
            "grpo_group_size": grpo_group_size,
            "grpo_beta": grpo_beta,
            "truncation_steps": truncation_steps,
            "max_grad_norm": max_grad_norm,
            "value_lr_scale": value_lr_scale,
            "warmup_ratio": warmup_ratio,
            "warmup_initial_ratio": warmup_initial_ratio,
            "warmup_final_ratio": warmup_final_ratio,
            "warmup_anneal_episodes": warmup_anneal_episodes,
            "max_consecutive_skips": max_consecutive_skips,
            "reward_schedule": copy.deepcopy(self.reward_schedule),
        }
        parameters = {"policy": [], "value": []}
        for name, parameter in self.policy_network.named_parameters():
            if parameter.requires_grad and parameter.numel():
                parameters["value" if "value_head" in name else "policy"].append(parameter)
        groups = [{"params": parameters["policy"], "lr": lr}]
        if parameters["value"]:
            groups.append({"params": parameters["value"], "lr": lr * value_lr_scale})
        optimizer = torch.optim.AdamW if hasattr(policy_network, "policy") else torch.optim.Adam
        self.optimizer = optimizer(groups, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_episodes
        )
        self.reference_policy = None
        if algo == "grpo":
            self.reference_policy = copy.deepcopy(self.policy_network).eval().requires_grad_(False)

    compute_gae = staticmethod(compute_gae)

    def get_current_warmup_ratio(self, episode):
        return warmup_ratio_at(
            episode,
            self.warmup_ratio,
            self.warmup_initial_ratio,
            self.warmup_final_ratio,
            self.warmup_anneal_episodes,
        )

    def _apply_reward_schedule(self):
        for stage in self.reward_schedule:
            if self.episode >= stage["episode"]:
                for name, weight in stage["weights"].items():
                    if not hasattr(self.reward_function, name) or not name.startswith("w_"):
                        raise ValueError(f"Unknown scheduled reward weight: {name}")
                    if not math.isfinite(weight) or weight < 0:
                        raise ValueError("Scheduled reward weights must be finite and nonnegative.")
                    setattr(self.reward_function, name, weight)

    def collect(
        self, batch_size, initial_noise=None, class_labels=None, *, capture_trajectory=False
    ):
        require_int(batch_size, "batch_size")
        self._apply_reward_schedule()
        group_size = self.grpo_group_size if self.algo == "grpo" else 1
        if class_labels is None and self.diffusion_model.num_classes is not None:
            if batch_size % group_size:
                raise ValueError("batch_size must be divisible by group_size.")
            labels = torch.randint(
                self.diffusion_model.num_classes, (batch_size // group_size,), device=self.device
            )
            class_labels = labels.repeat_interleave(group_size)
        return collect_rollout(
            self.diffusion_model,
            self.policy_network,
            self.reward_function,
            batch_size,
            self.inference_steps,
            budget=self.denoising_budget,
            budget_mode=self.budget_mode,
            warmup_ratio=self.get_current_warmup_ratio(self.episode),
            max_consecutive_skips=self.max_consecutive_skips,
            initial_noise=initial_noise,
            class_labels=class_labels,
            reference_steps=self.reference_steps,
            group_size=group_size,
            capture_trajectory=capture_trajectory,
        )

    def _auxiliary_loss(self, rollout):
        if not self.diffusion_loss_weight:
            return torch.zeros((), device=self.device)
        policy = self.policy_network.policy
        losses = []
        for index, call in enumerate(rollout.calls):
            stop = (
                rollout.calls[index + 1].first_step
                if index + 1 < len(rollout.calls)
                else len(rollout.actions)
            )
            taken = rollout.actions[call.first_step : stop].T
            valid = ~rollout.forced[call.first_step : stop].T
            if not valid.any():
                continue
            padding = policy.action_horizon - taken.shape[1]
            losses.append(
                policy.compute_loss(
                    {
                        "observations": observation_to(call.observations, self.device),
                        "actions": F.pad(taken, (0, padding)),
                        "mask": F.pad(valid, (0, padding), value=False),
                    }
                )
            )
        return torch.stack(losses).mean() if losses else torch.zeros((), device=self.device)

    def update(self, rollout):
        trainable = ~rollout.forced
        if self.algo == "grpo":
            discounted = (
                rollout.rewards
                * self.gamma ** torch.arange(len(rollout.rewards), device=self.device)[:, None]
            )
            advantages = grouped_advantages(discounted.sum(0), self.grpo_group_size)[
                None
            ].expand_as(rollout.rewards)
            targets = rollout.values
        else:
            advantages, targets = compute_gae(
                rollout.rewards,
                rollout.values,
                self.gamma,
                1.0 if self.algo == "a2c_simple" else self.gae_lambda,
            )
            advantages = normalized_advantages(advantages, trainable)
        reference_probs = None
        if self.reference_policy is not None and self.grpo_beta:
            if hasattr(self.reference_policy, "policy"):
                self.reference_policy.policy.timestep_stride = (
                    self.policy_network.policy.timestep_stride
                )
            with torch.no_grad():
                reference_probs = replay_policy(
                    self.reference_policy, rollout, self.device, self.truncation_steps
                )[3]
        epochs = self.ppo_epochs if self.algo in ("ppo", "grpo") else 1
        result = {}
        for _ in range(epochs):
            log_probs, values, entropies, probabilities = replay_policy(
                self.policy_network, rollout, self.device, self.truncation_steps
            )
            if self.algo in ("ppo", "grpo"):
                ratio = (log_probs - rollout.log_probs).exp()
                clipped = ratio.clamp(1 - self.ppo_clip_ratio, 1 + self.ppo_clip_ratio)
                policy_loss = -masked_mean(
                    torch.minimum(ratio * advantages, clipped * advantages), trainable
                )
            else:
                policy_loss = -masked_mean(log_probs * advantages, trainable)
            value_loss = (
                values.new_zeros(())
                if self.algo == "grpo"
                else F.mse_loss(values, targets.detach())
            )
            entropy = masked_mean(entropies, trainable)
            kl = torch.zeros((), device=self.device)
            if reference_probs is not None:
                kl_values = (
                    probabilities * (probabilities / reference_probs).log()
                    + (1 - probabilities) * ((1 - probabilities) / (1 - reference_probs)).log()
                )
                kl = masked_mean(kl_values, trainable)
            auxiliary = self._auxiliary_loss(rollout)
            loss = (
                policy_loss
                + 0.5 * value_loss
                - self.entropy_weight * entropy
                + self.diffusion_loss_weight * auxiliary
                + self.grpo_beta * kl
            )
            require_finite(loss, "training loss")
            self.optimizer.zero_grad(set_to_none=True)
            gradient_norm = torch.zeros((), device=self.device)
            if loss.requires_grad:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy_network.parameters(), self.max_grad_norm, error_if_nonfinite=True
                )
            self.optimizer.step()
            result = {
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "diffusion_loss": float(auxiliary.detach()),
                "kl": float(kl.detach()),
                "gradient_norm": float(gradient_norm),
            }
        return result

    def train_step(self, batch_size, initial_noise=None, class_labels=None, *, checkpoint_dir=None):
        if self.episode >= self.num_episodes:
            raise ValueError("Training has already reached num_episodes.")
        rollout = self.collect(
            batch_size, initial_noise, class_labels, capture_trajectory=checkpoint_dir is not None
        )
        result = self.update(rollout)
        self.scheduler.step()
        result.update(
            {
                "episode": self.episode,
                "reward": float(rollout.rewards.sum(0).mean()),
                "nfe_mean": float(rollout.nfe.float().mean()),
                "nfe_min": int(rollout.nfe.min()),
                "nfe_max": int(rollout.nfe.max()),
                "forced_fraction": float(rollout.forced.float().mean()),
                "policy_calls": len(rollout.calls),
                "warmup_ratio": self.get_current_warmup_ratio(self.episode),
            }
        )
        for name, values in rollout.components.items():
            if name != "total":
                result[f"reward_{name}"] = float(values.sum(0).mean())
        self.history.append(result)
        self.episode += 1
        if checkpoint_dir is not None:
            from utils.visualization import save_training_artifacts

            self.save_checkpoint(Path(checkpoint_dir) / "latest.pt")
            save_training_artifacts(rollout, self.history, checkpoint_dir)
        return result

    def train(self, num_episodes, batch_size, checkpoint_dir=None, checkpoint_interval=1):
        require_int(num_episodes, "stopping episode", maximum=self.num_episodes)
        require_int(checkpoint_interval, "checkpoint_interval")
        require_int(batch_size, "batch_size")
        if num_episodes <= self.episode:
            raise ValueError("The stopping episode must be greater than the current episode.")
        while self.episode < num_episodes:
            saving = (
                self.episode + 1
            ) % checkpoint_interval == 0 or self.episode + 1 == num_episodes
            metrics = self.train_step(batch_size, checkpoint_dir=checkpoint_dir if saving else None)
            print(
                f"episode={self.episode} reward={metrics['reward']:.6f} "
                f"nfe={metrics['nfe_mean']:.2f} loss={metrics['loss']:.6f}"
            )
        return self.history

    def save_checkpoint(self, path):
        payload = {
            "format_version": 1,
            "episode": self.episode,
            "history": self.history,
            "experiment": self.experiment,
            "settings": self.settings,
            "policy_network_state_dict": self.policy_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "reward_state_dict": self.reward_function.state_dict(),
            "reward_configuration": self.reward_function.configuration,
            "provenance": provenance(),
            "reward_weights": {
                key: value
                for key, value in vars(self.reward_function).items()
                if key.startswith("w_")
            },
            "rng_state": rng_state(),
            "reference_policy_state_dict": None
            if self.reference_policy is None
            else self.reference_policy.state_dict(),
        }
        save_checkpoint(payload, path)

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=True)
        current = provenance()
        if (
            payload.get("reward_configuration") != self.reward_function.configuration
            or payload.get("provenance") != current
        ):
            raise ValueError(
                "Exact resume requires matching reward assets, source, and environment."
            )
        if (
            payload["format_version"] != 1
            or payload["settings"] != self.settings
            or payload["experiment"] != self.experiment
        ):
            raise ValueError(
                "Exact resume requires the same format, experiment, and trainer settings."
            )
        self.policy_network.load_state_dict(payload["policy_network_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.reward_function.load_state_dict(payload["reward_state_dict"])
        for name, value in payload["reward_weights"].items():
            setattr(self.reward_function, name, value)
        if self.reference_policy is not None:
            self.reference_policy.load_state_dict(
                payload["reference_policy_state_dict"], strict=True
            )
        self.episode, self.history = payload["episode"], payload["history"]
        restore_rng_state(payload["rng_state"])
        return self.episode, self.history
