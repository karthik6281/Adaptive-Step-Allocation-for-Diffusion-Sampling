import copy

import pytest
import torch

from training.rl_trainer import RLTrainer, compute_gae, grouped_advantages, normalized_advantages
from training.rollout import collect_rollout, detach_hidden, replay_policy
from training.tbptt_trainer import TBPTTRLTrainer
from utils.reproducibility import seed_everything


def test_gae_matches_hand_computed_returns_and_detaches_targets():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [1.0], [1.5]], requires_grad=True)
    advantages, returns = compute_gae(rewards, values, gamma=1, gae_lambda=1)
    torch.testing.assert_close(returns, torch.tensor([[6.0], [5.0], [3.0]]))
    torch.testing.assert_close(advantages, torch.tensor([[5.5], [4.0], [1.5]]))
    assert not advantages.requires_grad
    assert not returns.requires_grad
    dones = torch.tensor([[False], [True], [False]])
    _, truncated = compute_gae(rewards, values, 1, 1, dones)
    torch.testing.assert_close(truncated, torch.tensor([[3.0], [2.0], [3.0]]))


def test_gae_bootstrap_and_zero_lambda():
    rewards, values = torch.tensor([[1.0], [2.0]]), torch.tensor([[0.5], [1.0]])
    advantage, _ = compute_gae(rewards, values, 0.9, 0, bootstrap=torch.tensor([4.0]))
    torch.testing.assert_close(advantage, torch.tensor([[1.4], [4.6]]))


def test_singleton_advantage_is_not_nan_or_erased():
    advantage = torch.tensor([[3.0]])
    torch.testing.assert_close(
        normalized_advantages(advantage, torch.ones_like(advantage, dtype=torch.bool)), advantage
    )
    torch.testing.assert_close(
        grouped_advantages(torch.tensor([1.0, 3.0, 5.0, 5.0]), 2),
        torch.tensor([-1.0, 1.0, 0.0, 0.0]),
    )


@pytest.mark.parametrize("kind", ["lstm", "direct", "diffusion"])
def test_replayed_likelihoods_match_collection(model, policy_factory, reward, kind):
    policy = policy_factory(kind)
    rollout = collect_rollout(model, policy, reward, 3, 6, budget=3)
    log_probs, values, _, _ = replay_policy(policy, rollout, "cpu")
    torch.testing.assert_close(log_probs, rollout.log_probs, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(values, rollout.values, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("kind", ["lstm", "direct", "diffusion"])
@pytest.mark.parametrize("algorithm", ["a2c_simple", "a2c_gae", "ppo", "grpo"])
def test_each_algorithm_updates_policy_with_finite_gradients(
    model, policy_factory, reward, kind, algorithm
):
    policy = policy_factory(kind)
    before = {
        name: value.detach().clone()
        for name, value in policy.named_parameters()
        if "value_head" not in name
    }
    trainer = RLTrainer(
        model,
        policy,
        reward,
        algo=algorithm,
        inference_steps=6,
        num_episodes=3,
        denoising_budget=3,
        budget_mode="exact",
        grpo_group_size=2,
        diffusion_loss_weight=0.01 if kind == "diffusion" else 0,
    )
    metrics = trainer.train_step(4)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["nfe_min"] == metrics["nfe_max"] == 3
    assert any(
        not torch.equal(before[name], value)
        for name, value in policy.named_parameters()
        if name in before
    )
    assert all(not parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("kind", ["lstm", "direct", "diffusion"])
@pytest.mark.parametrize("algorithm", ["a2c_gae", "ppo", "grpo"])
def test_checkpoint_resume_matches_uninterrupted_training(
    tmp_path, model, policy_factory, reward, kind, algorithm
):
    def create():
        seed_everything(81)
        return RLTrainer(
            model,
            policy_factory(kind),
            copy.deepcopy(reward),
            algo=algorithm,
            inference_steps=6,
            num_episodes=3,
            denoising_budget=3,
            budget_mode="cap",
            grpo_group_size=2,
        )

    uninterrupted = create()
    uninterrupted.train(3, 4)
    expected = copy.deepcopy(uninterrupted.policy_network.state_dict())
    interrupted = create()
    interrupted.train(1, 4)
    checkpoint = tmp_path / "resume.pt"
    interrupted.save_checkpoint(checkpoint)
    resumed = create()
    resumed.load_checkpoint(checkpoint)
    resumed.train(3, 4)
    assert resumed.history == uninterrupted.history
    for name, value in resumed.policy_network.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)


def test_grouped_rollouts_share_noise_and_reject_unpaired_input(model, policy_factory, reward):
    trainer = RLTrainer(
        model, policy_factory(), reward, algo="grpo", grpo_group_size=2, inference_steps=6
    )
    rollout = trainer.collect(4)
    torch.testing.assert_close(rollout.initial_noise[0], rollout.initial_noise[1], atol=0, rtol=0)
    torch.testing.assert_close(rollout.initial_noise[2], rollout.initial_noise[3], atol=0, rtol=0)
    with pytest.raises(ValueError, match="same initial noise"):
        trainer.collect(4, initial_noise=torch.randn(4, 1, 8, 8))


def test_tbptt_truncates_hidden_state_and_trains(model, policy_factory, reward):
    hidden = (torch.ones(1, requires_grad=True) * 2, torch.ones(1, requires_grad=True) * 3)
    detached = detach_hidden(hidden)
    assert all(not value.requires_grad for value in detached)
    trainer = TBPTTRLTrainer(
        model,
        policy_factory(),
        reward,
        tbptt_chunk_size=2,
        inference_steps=6,
        num_episodes=2,
        denoising_budget=3,
        budget_mode="exact",
    )
    assert trainer.truncation_steps == 2
    assert trainer.train_step(2)["nfe_mean"] == 3


def test_forced_actions_do_not_train_actor(model, policy_factory, reward):
    policy = policy_factory()
    trainer = RLTrainer(
        model, policy, reward, inference_steps=6, denoising_budget=1, entropy_weight=0
    )
    before = copy.deepcopy(policy.skip_head.state_dict())
    rollout = trainer.collect(2)
    assert rollout.forced.all()
    trainer.update(rollout)
    for name, value in policy.skip_head.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_incompatible_resume_is_not_a_fresh_optimizer_fallback(
    tmp_path, model, policy_factory, reward
):
    trainer = RLTrainer(model, policy_factory(), reward, inference_steps=6)
    checkpoint = tmp_path / "state.pt"
    trainer.save_checkpoint(checkpoint)
    incompatible = RLTrainer(model, policy_factory(), reward, inference_steps=6, gamma=0.9)
    with pytest.raises(ValueError, match="Exact resume"):
        incompatible.load_checkpoint(checkpoint)
