import pytest
import torch

from training.budget import enforce_budget
from training.rollout import collect_rollout


@pytest.mark.parametrize("steps", range(1, 9))
@pytest.mark.parametrize("mode", ["cap", "exact"])
def test_all_binary_action_sequences_respect_every_budget(steps, mode):
    proposals = ((torch.arange(2**steps)[:, None] >> torch.arange(steps)) & 1).float()
    for budget in range(1, steps + 1):
        used = torch.zeros(2**steps, dtype=torch.long)
        for index in range(steps):
            actions, forced = enforce_budget(proposals[:, index], used, budget, steps - index, mode)
            torch.testing.assert_close(actions[~forced], proposals[:, index][~forced])
            used += (actions == 0).long()
            assert torch.all(used <= budget)
        assert torch.all(actions == 0)
        assert torch.all(used >= 1)
        if mode == "exact":
            assert torch.all(used == budget)


@pytest.mark.parametrize(
    "budget, remaining, used, mode",
    [
        (0, 3, 0, "cap"),
        (-1, 3, 0, "cap"),
        (3, 0, 0, "cap"),
        (3, 2, 0, "exact"),
        (3, 1, 3, "cap"),
        (3, 2, -1, "cap"),
        (3, 2, 0, "unknown"),
    ],
)
def test_invalid_budget_state_is_rejected(budget, remaining, used, mode):
    with pytest.raises(ValueError):
        enforce_budget(torch.ones(1), torch.tensor([used]), budget, remaining, mode)


@pytest.mark.parametrize("mode", ["cap", "exact"])
@pytest.mark.parametrize("budget", [1, 2, 5, 6])
def test_actual_unet_calls_equal_reported_nfe(
    model, policy_factory, reward, monkeypatch, mode, budget
):
    calls = []
    original = model.predict_noise

    def counted(state, timestep, class_labels=None):
        calls.append(len(state))
        return original(state, timestep, class_labels)

    monkeypatch.setattr(model, "predict_noise", counted)
    rollout = collect_rollout(
        model, policy_factory(), reward, 4, 6, budget=budget, budget_mode=mode
    )
    assert sum(calls) == int(rollout.nfe.sum())
    assert torch.all(rollout.actions[-1] == 0)
    torch.testing.assert_close(rollout.components["cost"].sum(0), rollout.nfe.float())
    if mode == "exact":
        assert torch.all(rollout.nfe == budget)


def test_warmup_cannot_bypass_budget(model, policy_factory, reward):
    with pytest.raises(ValueError, match="warmup"):
        collect_rollout(model, policy_factory(), reward, 2, 6, budget=2, warmup_ratio=0.5)


def test_unconstrained_and_full_budget_match(model, policy_factory, reward):
    policy = policy_factory()
    noise = torch.randn(2, *model.sample_shape)
    first = collect_rollout(model, policy, reward, 2, 6, initial_noise=noise, stochastic=False)
    second = collect_rollout(
        model, policy, reward, 2, 6, budget=6, initial_noise=noise, stochastic=False
    )
    torch.testing.assert_close(first.images, second.images, rtol=0, atol=0)


def test_perceptual_deltas_use_pre_action_copy(model, policy_factory, perceptual_reward):
    noise = torch.randn(2, *model.sample_shape)
    untouched = noise.clone()
    rollout = collect_rollout(
        model, policy_factory(), perceptual_reward, 2, 6, budget=3, initial_noise=noise
    )
    before = perceptual_reward._compute_perceptual_reward(noise, rollout.reference_images)
    after = perceptual_reward._compute_perceptual_reward(rollout.images, rollout.reference_images)
    torch.testing.assert_close(rollout.components["perceptual"].sum(0), after - before)
    assert (after - before).abs().max() > 0
    torch.testing.assert_close(noise, untouched, rtol=0, atol=0)


@pytest.mark.parametrize("steps", range(1, 9))
@pytest.mark.parametrize("mode", ["cap", "exact"])
def test_every_feasible_budget_warmup_and_skip_limit(steps, mode):
    proposals = ((torch.arange(2**steps)[:, None] >> torch.arange(steps)) & 1).float()
    for limit in range(1, steps + 1):
        for warmup in range(steps + 1):
            minimum = warmup + (steps - warmup + limit) // (limit + 1)
            for budget in range(minimum, steps + 1):
                used = torch.zeros(2**steps, dtype=torch.long)
                streak = torch.zeros_like(used)
                for index in range(steps):
                    actions, forced = enforce_budget(
                        proposals[:, index],
                        used,
                        budget,
                        steps - index,
                        mode,
                        force_denoise=torch.full_like(used, index < warmup, dtype=torch.bool),
                        max_consecutive_skips=limit,
                        consecutive_skips=streak,
                    )
                    torch.testing.assert_close(actions[~forced], proposals[:, index][~forced])
                    used += (actions == 0).long()
                    streak = torch.where(actions == 0, 0, streak + 1)
                    assert torch.all(used <= budget)
                    assert torch.all(streak <= limit)
                    if index < warmup:
                        assert torch.all(actions == 0)
                assert torch.all(actions == 0)
                if mode == "exact":
                    assert torch.all(used == budget)


def test_conflicting_forced_denoise_does_not_silently_become_skip():
    with pytest.raises(ValueError, match="conflicts"):
        enforce_budget(
            torch.zeros(1),
            torch.tensor([2]),
            3,
            3,
            force_denoise=torch.tensor([True]),
        )


@pytest.mark.parametrize("warmup, budget, skip_limit", [(0, 2, 1), (0.5, 4, 1), (0.5, 3, None)])
def test_impossible_joint_constraints_fail_before_any_model_calls(
    model, policy_factory, monkeypatch, warmup, budget, skip_limit
):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid constraints must fail before reference or policy sampling.")

    policy = policy_factory()
    monkeypatch.setattr(model, "sample", unexpected)
    monkeypatch.setattr(policy, "forward", unexpected)
    with pytest.raises(ValueError, match="budget"):
        collect_rollout(
            model,
            policy,
            None,
            2,
            6,
            budget=budget,
            warmup_ratio=warmup,
            max_consecutive_skips=skip_limit,
            compute_reference=True,
        )
