import copy
import json

import pytest
import torch

from config.experiment import build_experiment, load_experiment
from training.rl_trainer import RLTrainer
from training.rollout import collect_rollout
from utils.evaluation import evaluate
from utils.reproducibility import save_json


def test_budget_does_not_override_maximum_skip_streak(model, policy_factory):
    policy = policy_factory()
    with torch.no_grad():
        policy.skip_head.weight.zero_()
        policy.skip_head.bias.fill_(-20)
    result = collect_rollout(
        model,
        policy,
        None,
        2,
        6,
        budget=3,
        budget_mode="exact",
        max_consecutive_skips=1,
        stochastic=False,
    )
    streak = torch.zeros(2, dtype=torch.long)
    for actions in result.actions:
        streak = torch.where(actions == 1, streak + 1, 0)
        assert torch.all(streak <= 1)
    assert torch.all(result.nfe == 3)


@pytest.mark.parametrize("kind", ["direct", "diffusion"])
def test_all_forced_actions_do_not_decay_actor_only_parameters(model, policy_factory, reward, kind):
    policy = policy_factory(kind)
    head = policy.policy.plan_head if kind == "direct" else policy.policy.transformer.head
    before = copy.deepcopy(head.state_dict())
    trainer = RLTrainer(
        model,
        policy,
        reward,
        inference_steps=6,
        denoising_budget=1,
        entropy_weight=0,
    )
    rollout = trainer.collect(2)
    assert rollout.forced.all()
    trainer.update(rollout)
    for name, value in head.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_invalid_reference_count_is_not_silently_replaced(model, policy_factory):
    with pytest.raises(ValueError, match="reference_steps"):
        collect_rollout(model, policy_factory(), None, 2, 6, reference_steps=0)


def test_failed_json_write_preserves_previous_valid_file(tmp_path):
    path = tmp_path / "history.json"
    save_json({"episode": 1}, path)
    with pytest.raises(ValueError):
        save_json({"episode": 2, "loss": float("nan")}, path)
    assert json.loads(path.read_text()) == {"episode": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_evaluation_produces_viewable_images_and_plots(tmp_path, model, policy_factory):
    output = tmp_path / "evaluation"
    evaluate(
        model,
        policy_factory(),
        num_samples=3,
        batch_size=2,
        steps=6,
        budget=3,
        metric="mse",
        output_dir=output,
    )
    assert len(list((output / "images").glob("*.png"))) == 3
    for name in (
        "samples.png",
        "comparison.png",
        "trajectory.png",
        "action_schedule.png",
        "nfe_histogram.png",
        "quality_vs_nfe.png",
    ):
        assert (output / name).is_file(), name


@pytest.mark.parametrize("kind", ["lstm", "direct", "diffusion"])
@pytest.mark.parametrize("algorithm", ["a2c_gae", "ppo", "grpo"])
def test_forced_actor_keeps_existing_optimizer_momentum(
    model, policy_factory, reward, kind, algorithm
):
    policy = policy_factory(kind)
    trainer = RLTrainer(
        model,
        policy,
        reward,
        algo=algorithm,
        grpo_group_size=2,
        inference_steps=6,
        denoising_budget=3,
        weight_decay=0.01,
    )
    trainer.train_step(4)
    head = (
        policy.skip_head
        if kind == "lstm"
        else policy.policy.plan_head
        if kind == "direct"
        else policy.policy.transformer.head
    )
    before = copy.deepcopy(head.state_dict())
    all_before = copy.deepcopy(policy.state_dict())
    trainer.denoising_budget = 1
    rollout = trainer.collect(4)
    assert rollout.forced.all()
    trainer.update(rollout)
    for name, value in head.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    if algorithm == "grpo":
        for name, value in policy.state_dict().items():
            torch.testing.assert_close(value, all_before[name], atol=0, rtol=0)


@pytest.mark.parametrize(
    "setting, value",
    [
        ("num_episodes", 2.5),
        ("num_episodes", True),
        ("ppo_epochs", 1.5),
        ("grpo_group_size", False),
        ("inference_steps", 6.0),
        ("reference_steps", 0),
        ("reference_steps", True),
        ("warmup_anneal_episodes", -1),
        ("warmup_anneal_episodes", 0.5),
        ("truncation_steps", 2.5),
        ("denoising_budget", True),
        ("max_consecutive_skips", 1.5),
    ],
)
def test_invalid_integer_training_settings_fail_early(
    model, policy_factory, reward, setting, value
):
    settings = {"inference_steps": 6, setting: value}
    with pytest.raises(ValueError):
        RLTrainer(model, policy_factory(), reward, **settings)


@pytest.mark.parametrize("reference_steps", [0, -1, 3.5, True, 101])
def test_bad_reference_count_does_not_create_output(
    tmp_path, model, policy_factory, reference_steps
):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="reference_steps"):
        evaluate(
            model, policy_factory(), steps=6, reference_steps=reference_steps, output_dir=output
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "schedule",
    [
        [{"episode": 1, "weights": {"w_color": float("nan")}}],
        [{"episode": 1, "weights": {"w_color": -0.1}}],
        [{"episode": 1, "weights": {"w_colr": 0.1}}],
        [{"episode": 1.5, "weights": {"w_color": 0.1}}],
        [{"episode": True, "weights": {"w_color": 0.1}}],
        [{"episode": 0, "weights": {"w_color": True}}],
        [{"episode": 3, "weights": {"w_color": 0.1}}],
        [{"episode": 1, "weights": {"w_color": 0.1}}, {"episode": 0, "weights": {}}],
        [{"episode": 1, "weights": []}],
        {"episode": 1, "weights": {}},
    ],
)
def test_curriculum_is_validated_before_training(
    model, policy_factory, reward, schedule, monkeypatch
):
    with pytest.raises(ValueError):
        RLTrainer(
            model,
            policy_factory(),
            reward,
            inference_steps=6,
            num_episodes=3,
            reward_schedule=schedule,
        )
    from pathlib import Path

    configuration = load_experiment(Path(__file__).resolve().parents[1] / "examples/toy.yaml")
    configuration["training"].update(num_episodes=3, reward_schedule=schedule)

    def unexpected(*args, **kwargs):
        pytest.fail("An invalid curriculum must fail before loading the backbone.")

    monkeypatch.setattr("config.experiment.create_model", unexpected)
    with pytest.raises(ValueError):
        build_experiment(configuration)


def test_direct_curriculum_is_part_of_exact_resume(tmp_path, model, policy_factory, reward):
    schedule = [{"episode": 1, "weights": {"w_color": 0.01}}]
    trainer = RLTrainer(
        model,
        policy_factory(),
        reward,
        inference_steps=6,
        num_episodes=3,
        reward_schedule=schedule,
    )
    trainer.train_step(2)
    checkpoint = tmp_path / "state.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["settings"]["reward_schedule"] == schedule
    changed = RLTrainer(model, policy_factory(), reward, inference_steps=6, num_episodes=3)
    with pytest.raises(ValueError, match="Exact resume"):
        changed.load_checkpoint(checkpoint)


def test_uninitialized_scheduled_reward_fails_at_construction(model, policy_factory, reward):
    with pytest.raises(ValueError, match="initialized reward assets"):
        RLTrainer(
            model,
            policy_factory(),
            reward,
            inference_steps=6,
            reward_schedule=[{"episode": 2, "weights": {"w_perceptual": 1.0}}],
        )


def test_no_work_and_beyond_end_training_are_rejected(tmp_path, model, policy_factory, reward):
    trainer = RLTrainer(model, policy_factory(), reward, inference_steps=6, num_episodes=1)
    with pytest.raises(ValueError, match="stopping episode"):
        trainer.train(0, 2, checkpoint_dir=tmp_path / "unused")
    assert not (tmp_path / "unused").exists()
    trainer.train_step(2)
    with pytest.raises(ValueError, match="already reached"):
        trainer.train_step(2)
    with pytest.raises(ValueError, match="greater than"):
        trainer.train(1, 2)
    assert len(trainer.history) == 1
