"""One rollout implementation for training and evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch.distributions import Bernoulli

from training.budget import enforce_budget, validate_rollout_settings
from utils.reproducibility import require_finite, require_int


@contextmanager
def policy_randomness(seed, device):
    device = torch.device(device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        yield


def observation_to(observations, device):
    return {name: value.to(device) for name, value in observations.items()}


def store_observation(observations):
    return {name: value.detach().cpu().clone() for name, value in observations.items()}


def detach_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    if isinstance(hidden, tuple):
        return tuple(detach_hidden(value) for value in hidden)
    if hidden is not None:
        raise TypeError(f"Unsupported recurrent state: {type(hidden).__name__}")
    return None


@dataclass
class PolicyCall:
    observations: dict[str, torch.Tensor]
    seed: int
    first_step: int


@dataclass
class Rollout:
    calls: list[PolicyCall]
    observations: list[dict[str, torch.Tensor]]
    call_indices: list[int]
    slots: list[int]
    actions: torch.Tensor
    forced: torch.Tensor
    rewards: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    nfe: torch.Tensor
    images: torch.Tensor
    reference_images: torch.Tensor | None
    initial_noise: torch.Tensor
    components: dict[str, torch.Tensor]
    class_labels: torch.Tensor | None = None
    trajectory: list[dict] = field(default_factory=list)

    def tensor_payload(self, max_samples=None):
        count = len(self.images) if max_samples is None else min(len(self.images), max_samples)
        require_int(count, "max_samples")
        return {
            "images": self.images[:count].detach().cpu(),
            "reference_images": None
            if self.reference_images is None
            else self.reference_images[:count].detach().cpu(),
            "nfe": self.nfe[:count].cpu(),
            "actions": self.actions[:, :count].cpu(),
            "forced": self.forced[:, :count].cpu(),
            "initial_noise": self.initial_noise[:count].detach().cpu(),
            "class_labels": None if self.class_labels is None else self.class_labels[:count].cpu(),
            "trajectory": self.trajectory,
        }


def invoke_policy(policy, observations, seed, hidden=None):
    with policy_randomness(seed, observations["image"].device):
        if hasattr(policy, "plan"):
            probabilities, values = policy.plan(observations)
            return probabilities, values, hidden
        probability, hidden, values, _ = policy(
            observations["image"],
            observations["timestep"],
            hidden,
            observations.get("class_labels"),
        )
        return probability[:, None], values, hidden


def critic_value(policy, observations, planned_values):
    if hasattr(policy, "policy"):
        core = policy.policy
        return core.value_head(core._condition(observations)).squeeze(-1)
    return planned_values


def distribution(probabilities):
    if probabilities.ndim != 1 or not torch.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("The policy must return finite [batch] probabilities in [0, 1].")
    return Bernoulli(probs=probabilities.clamp(1e-6, 1 - 1e-6))


@torch.no_grad()
def collect_rollout(
    diffusion_model,
    policy,
    reward,
    batch_size,
    steps,
    budget=None,
    budget_mode="cap",
    warmup_ratio=0.0,
    max_consecutive_skips=None,
    initial_noise=None,
    class_labels=None,
    reference_steps=None,
    stochastic=True,
    group_size=1,
    compute_reference=False,
    capture_trajectory=False,
    record_observations=True,
):
    require_int(batch_size, "batch_size")
    require_int(steps, "steps", maximum=diffusion_model.num_timesteps)
    require_int(group_size, "group_size")
    if batch_size % group_size:
        raise ValueError("batch_size must be divisible by group_size.")
    if reference_steps is not None:
        require_int(reference_steps, "reference_steps", maximum=diffusion_model.num_timesteps)
    allowance, warmup_steps = validate_rollout_settings(
        steps, budget, budget_mode, warmup_ratio, max_consecutive_skips
    )
    device = diffusion_model.device
    policy.eval()
    diffusion_model.eval()
    if initial_noise is None:
        noise = torch.randn(batch_size // group_size, *diffusion_model.sample_shape, device=device)
        initial_noise = noise.repeat_interleave(group_size, dim=0)
    if tuple(initial_noise.shape) != (batch_size, *diffusion_model.sample_shape):
        raise ValueError("Initial noise does not match the diffusion model's sample shape.")
    initial_noise = require_finite(initial_noise.to(device), "initial noise")
    if class_labels is not None:
        if class_labels.shape != (batch_size,):
            raise ValueError("class_labels must have shape [batch]")
        class_labels = class_labels.to(device)
    if group_size > 1:
        grouped_noise = initial_noise.reshape(-1, group_size, *diffusion_model.sample_shape)
        if not torch.equal(grouped_noise, grouped_noise[:, :1].expand_as(grouped_noise)):
            raise ValueError("GRPO trajectories in each group must share the same initial noise.")
        if class_labels is not None:
            grouped_labels = class_labels.reshape(-1, group_size)
            if not torch.equal(grouped_labels, grouped_labels[:, :1].expand_as(grouped_labels)):
                raise ValueError("GRPO trajectories in each group must share the same class label.")
    reference = None
    if compute_reference or (reward is not None and reward.needs_reference):
        reference = require_finite(
            diffusion_model.sample(
                initial_noise, steps if reference_steps is None else reference_steps, class_labels
            ),
            "reference images",
        )
    grid = diffusion_model.set_timesteps(steps)
    destinations = torch.cat((grid[1:], grid.new_tensor([-1])))
    state = initial_noise.clone()
    current_time = grid[0].expand(batch_size).clone()
    used = torch.zeros(batch_size, dtype=torch.long, device=device)
    consecutive_skips = torch.zeros_like(used)
    planned = planned_values = hidden = None
    horizon = policy.get_planning_horizon() if hasattr(policy, "get_planning_horizon") else 1
    require_int(horizon, "policy planning horizon")
    if hasattr(policy, "policy"):
        policy.policy.timestep_stride = max(1, diffusion_model.num_timesteps // steps)
    calls, observations, indices, slots = [], [], [], []
    actions, forced, rewards, log_probs, values = [], [], [], [], []
    components = {}
    trajectory = []
    if capture_trajectory:
        preview_count = min(4, batch_size)
        frame_steps = set(torch.linspace(0, steps - 1, min(8, steps)).round().long().tolist())
        latest_clean = state.new_zeros((preview_count, *state.shape[1:]))
        available = torch.zeros(preview_count, device=device, dtype=torch.bool)
    for index, destination in enumerate(destinations):
        observation = {
            "image": state,
            "timestep": current_time,
            "budget_ratio": (allowance - used).float() / allowance,
            "jump_ratio": (current_time - destination).float() / diffusion_model.num_timesteps,
            "progress_ratio": state.new_full((batch_size,), index / steps),
            "rate_ratio": ((allowance - used).float() / (steps - index)).clamp(0, 1),
        }
        if class_labels is not None:
            observation["class_labels"] = class_labels
        stored = store_observation(observation) if record_observations else {}
        if record_observations:
            observations.append(stored)
        slot = index % horizon
        if slot == 0:
            seed = int(torch.randint(0, 2**31 - 1, ()).item())
            calls.append(PolicyCall(stored, seed, index))
            planned, planned_values, hidden = invoke_policy(policy, observation, seed, hidden)
            if planned.ndim != 2 or planned.shape[0] != batch_size or planned.shape[1] < horizon:
                raise ValueError("The policy plan is shorter than its execution horizon.")
        indices.append(len(calls) - 1)
        slots.append(slot)
        action_distribution = distribution(planned[:, slot])
        proposed = action_distribution.sample() if stochastic else (planned[:, slot] >= 0.5).float()
        force = torch.full((batch_size,), index < warmup_steps, device=device, dtype=torch.bool)
        action, shielded = enforce_budget(
            proposed,
            used,
            allowance,
            steps - index,
            budget_mode,
            force,
            max_consecutive_skips=max_consecutive_skips,
            consecutive_skips=consecutive_skips,
        )
        value = require_finite(critic_value(policy, observation, planned_values), "critic values")
        if value.shape != (batch_size,):
            raise ValueError("The critic must return shape [batch].")
        before = (
            diffusion_model.decode(state).clone()
            if reward is not None and reward.needs_images
            else state
        )
        execute = action == 0
        if execute.any():
            labels = None if class_labels is None else class_labels[execute]
            next_state, _, predicted_clean = diffusion_model.transition(
                state[execute],
                current_time[execute],
                destination.expand(int(execute.sum())),
                labels,
            )
            state[execute] = next_state
            current_time[execute] = destination
            if capture_trajectory:
                selected = execute.nonzero(as_tuple=True)[0]
                visible = selected < preview_count
                latest_clean[selected[visible]] = predicted_clean[visible]
                available[selected[visible]] = True
        used += execute.long()
        consecutive_skips = torch.where(execute, 0, consecutive_skips + 1)
        if capture_trajectory and index in frame_steps and index < steps - 1:
            frame = None
            if available.any():
                decoded = require_finite(
                    diffusion_model.decode(latest_clean[available]), "trajectory images"
                ).cpu()
                frame = decoded.new_zeros((preview_count, *decoded.shape[1:]))
                frame[available.cpu()] = decoded
            trajectory.append(
                {
                    "step": index + 1,
                    "kind": "predicted_clean",
                    "images": frame,
                    "available": available.cpu().clone(),
                    "actions": action[:preview_count].cpu().clone(),
                    "forced": shielded[:preview_count].cpu().clone(),
                    "nfe": used[:preview_count].cpu().clone(),
                }
            )
        if reward is None:
            result = {"total": state.new_zeros(batch_size)}
        else:
            after = diffusion_model.decode(state) if reward.needs_images else state
            result = reward.compute_comprehensive_rewards(
                after,
                reference,
                action,
                x_pre=before,
                target_class=class_labels,
                timestep_ratio=max(0, int(destination)) / diffusion_model.num_timesteps,
            )
        for name, value_component in result.items():
            components.setdefault(name, []).append(value_component.detach())
        actions.append(action)
        forced.append(shielded)
        rewards.append(result["total"])
        log_probs.append(action_distribution.log_prob(action))
        values.append(value)
    if budget_mode == "exact" and not torch.all(used == allowance):
        raise AssertionError("Exact-budget rollout did not spend its budget.")
    if torch.any(used > allowance) or not torch.all(current_time == -1):
        raise AssertionError("Rollout violated its budget or omitted the terminal denoising call.")
    images = require_finite(diffusion_model.decode(state), "generated images")
    if capture_trajectory:
        trajectory.append(
            {
                "step": steps,
                "kind": "final_sample",
                "images": images[:preview_count].detach().cpu().clone(),
                "available": torch.ones(preview_count, dtype=torch.bool),
                "actions": actions[-1][:preview_count].cpu().clone(),
                "forced": forced[-1][:preview_count].cpu().clone(),
                "nfe": used[:preview_count].cpu().clone(),
            }
        )
    return Rollout(
        calls,
        observations,
        indices,
        slots,
        torch.stack(actions),
        torch.stack(forced),
        torch.stack(rewards),
        torch.stack(log_probs),
        torch.stack(values),
        used,
        images,
        reference,
        initial_noise.detach().clone(),
        {name: torch.stack(parts) for name, parts in components.items()},
        None if class_labels is None else class_labels.detach().clone(),
        trajectory,
    )


def replay_policy(policy, rollout, device, truncation_steps=None):
    if len(rollout.observations) != len(rollout.actions):
        raise ValueError(
            "Policy replay requires a rollout collected with record_observations=True."
        )
    policy.eval()
    hidden = None
    planned = planned_values = None
    active_call = -1
    log_probs, values, entropies, probabilities = [], [], [], []
    for step, observation in enumerate(rollout.observations):
        if truncation_steps and step % truncation_steps == 0:
            hidden = detach_hidden(hidden)
        call_index = rollout.call_indices[step]
        if call_index != active_call:
            call = rollout.calls[call_index]
            planned, planned_values, hidden = invoke_policy(
                policy, observation_to(call.observations, device), call.seed, hidden
            )
            active_call = call_index
        probability = planned[:, rollout.slots[step]]
        action_distribution = distribution(probability)
        log_probs.append(action_distribution.log_prob(rollout.actions[step]))
        entropies.append(action_distribution.entropy())
        probabilities.append(action_distribution.probs)
        values.append(critic_value(policy, observation_to(observation, device), planned_values))
    return tuple(torch.stack(parts) for parts in (log_probs, values, entropies, probabilities))
