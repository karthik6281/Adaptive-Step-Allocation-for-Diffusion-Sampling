import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import training.rollout as rollout_module
import utils.evaluation as evaluation_module
import utils.visualization as visualization
from models.diffusion import DiffusionModel, ToyDenoiser
from models.policy_network import PolicyNetwork
from training.rl_trainer import RLTrainer
from training.rollout import collect_rollout
from training.tbptt_trainer import TBPTTRLTrainer
from utils.reproducibility import restore_rng_state, rng_state, seed_everything


def assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_equal(left, right)
    else:
        assert actual == expected


def expected_pixels(images):
    values = images.detach().cpu().numpy().astype(np.float32)
    pixels = np.rint((np.clip(values, -1, 1) + 1) * 127.5).astype(np.uint8)
    return pixels[:, 0] if pixels.shape[1] == 1 else pixels.transpose(0, 2, 3, 1)


def assert_png(path, expected=None):
    with Image.open(path) as image:
        assert image.format == "PNG"
        image.load()
        if expected is not None:
            assert image.mode == ("L" if expected.ndim == 2 else "RGB")
            np.testing.assert_array_equal(np.asarray(image), expected)
        else:
            assert min(image.size) > 20
            assert np.ptp(np.asarray(image.convert("RGB"))) > 0


def assert_same_rollout(actual, expected):
    for name in (
        "actions",
        "forced",
        "rewards",
        "log_probs",
        "values",
        "nfe",
        "images",
        "reference_images",
        "initial_noise",
        "class_labels",
        "components",
        "call_indices",
        "slots",
    ):
        assert_equal(getattr(actual, name), getattr(expected, name))
    assert [(call.seed, call.first_step) for call in actual.calls] == [
        (call.seed, call.first_step) for call in expected.calls
    ]


def count_predictions(monkeypatch, model):
    original = model.predict_noise
    counts = []

    def predict(sample, timestep, class_labels=None):
        counts.append(len(sample))
        return original(sample, timestep, class_labels)

    monkeypatch.setattr(model, "predict_noise", predict)
    return counts


def expected_prediction_batches(rollouts, reference_steps=0):
    counts = []
    for rollout in rollouts:
        counts.extend([len(rollout.images)] * reference_steps)
        counts.extend(
            int((actions == 0).sum()) for actions in rollout.actions if (actions == 0).any()
        )
    return counts


def mock_training_plots(monkeypatch):
    for name in (
        "save_training_curves",
        "save_image_grid",
        "save_action_schedule",
        "save_trajectory",
        "save_comparison",
    ):
        monkeypatch.setattr(visualization, name, Mock())


@pytest.mark.parametrize("channels,batch", [(1, 1), (3, 1), (1, 3), (3, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sample_pngs_have_exact_pixels_without_mutating_tensors(tmp_path, channels, batch, dtype):
    values = torch.tensor([-2, -1, -0.5, 0, 0.5, 1, 2, 0.25], dtype=dtype)
    images = torch.stack(
        [
            torch.stack(
                [values.roll(sample + channel).reshape(2, 4) for channel in range(channels)]
            )
            for sample in range(batch)
        ]
    ).transpose(-1, -2)
    images.requires_grad_(True)
    original = images.detach().clone()
    expected = expected_pixels(original)
    np.testing.assert_array_equal(
        expected[0, :, :, 0] if channels == 3 else expected[0],
        np.array([[0, 191], [0, 255], [64, 255], [128, 159]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(visualization.image_pixels(images), expected)
    directory = tmp_path / "images"
    visualization.save_sample_images(images, directory, first_sample=7)
    assert sorted(path.name for path in directory.iterdir()) == [
        f"{index:06d}.png" for index in range(7, 7 + batch)
    ]
    for index, pixels in enumerate(expected, start=7):
        assert_png(directory / f"{index:06d}.png", pixels)
    assert_equal(images.detach(), original)
    assert images.requires_grad and images.grad is None


@pytest.mark.parametrize(
    "shape",
    [(), (2,), (1, 8, 8), (1, 2, 4, 4), (1, 4, 4, 4), (0, 3, 4, 4), (1, 3, 0, 4), (1, 1, 4, 0)],
)
def test_invalid_image_shapes_are_rejected_before_writing(tmp_path, shape):
    images = torch.zeros(shape)
    with pytest.raises(ValueError, match="nonempty"):
        visualization.image_pixels(images)
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError, match="nonempty"):
        visualization.save_sample_images(images, directory)
    assert not directory.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_images_are_rejected_before_writing(tmp_path, value):
    images = torch.zeros(1, 3, 2, 2)
    images[0, 1, 0, 0] = value
    with pytest.raises(FloatingPointError, match="display images"):
        visualization.image_pixels(images)
    directory = tmp_path / "invalid"
    with pytest.raises(FloatingPointError, match="display images"):
        visualization.save_sample_images(images, directory)
    assert not directory.exists()


@pytest.mark.parametrize("kind", ["lstm", "direct", "diffusion"])
@pytest.mark.parametrize("capture,record", [(True, True), (False, False), (True, False)])
def test_capture_and_observation_elision_are_numerically_and_rng_neutral(
    monkeypatch, model, policy_factory, perceptual_reward, kind, capture, record
):
    policy = policy_factory(kind)
    initial = torch.randn(5, *model.sample_shape)
    original_noise = initial.clone()
    predictions = count_predictions(monkeypatch, model)
    before = rng_state()
    baseline = collect_rollout(
        model, policy, perceptual_reward, 5, 6, budget=3, initial_noise=initial
    )
    baseline_rng = rng_state()
    baseline_predictions = predictions.copy()
    assert baseline_predictions == expected_prediction_batches([baseline], reference_steps=6)
    restore_rng_state(before)
    predictions.clear()
    if not record:
        monkeypatch.setattr(
            rollout_module,
            "store_observation",
            Mock(side_effect=AssertionError("stored an observation")),
        )
    observed = collect_rollout(
        model,
        policy,
        perceptual_reward,
        5,
        6,
        budget=3,
        initial_noise=initial,
        capture_trajectory=capture,
        record_observations=record,
    )
    assert_same_rollout(observed, baseline)
    assert_equal(rng_state(), baseline_rng)
    assert predictions == baseline_predictions
    assert_equal(initial, original_noise)
    assert bool(observed.trajectory) == capture
    if record:
        assert_equal(observed.observations, baseline.observations)
    else:
        assert observed.observations == []
        assert all(call.observations == {} for call in observed.calls)


@pytest.mark.parametrize("steps,batch", [(1, 1), (6, 1), (6, 5), (20, 7)])
def test_trajectory_bounds_include_the_actual_final_sample(model, policy_factory, steps, batch):
    rollout = collect_rollout(
        model, policy_factory(), None, batch, steps, budget=min(3, steps), capture_trajectory=True
    )
    frames = rollout.trajectory
    assert len(frames) == min(8, steps)
    assert [frame["step"] for frame in frames] == sorted({frame["step"] for frame in frames})
    assert frames[0]["step"] == 1
    assert frames[-1]["step"] == steps
    for frame in frames:
        count = min(4, batch)
        for name in ("available", "actions", "forced", "nfe"):
            assert frame[name].shape == (count,)
        assert frame["available"].dtype == torch.bool
        if frame["images"] is not None:
            assert frame["images"].shape == (count, *rollout.images.shape[1:])
            assert torch.isfinite(frame["images"]).all()
    final = frames[-1]
    assert final["kind"] == "final_sample" and final["available"].all()
    assert_equal(final["images"], rollout.images[:4])
    assert_equal(final["nfe"], rollout.nfe[:4])
    assert_equal(final["actions"], rollout.actions[-1, :4])
    assert_equal(final["forced"], rollout.forced[-1, :4])
    assert all(frame["kind"] == "predicted_clean" for frame in frames[:-1])


class ScheduledPolicy(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.register_buffer("probabilities", actions.T.float())

    def get_planning_horizon(self):
        return self.probabilities.shape[1]

    def plan(self, observations):
        return self.probabilities, observations["image"].new_zeros(len(self.probabilities))


def test_trajectory_marks_missing_predictions_and_holds_clean_estimates_on_skips(
    monkeypatch, tmp_path, model
):
    requested = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1],
            [0, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    original_transition = model.transition
    predictions = []

    def record_transition(*args, **kwargs):
        result = original_transition(*args, **kwargs)
        predictions.append(result[2].clone())
        return result

    monkeypatch.setattr(model, "transition", record_transition)
    rollout = collect_rollout(
        model,
        ScheduledPolicy(requested),
        None,
        5,
        6,
        budget=3,
        stochastic=False,
        capture_trajectory=True,
    )
    assert_equal(rollout.actions[:-1], requested[:-1].float())
    assert (rollout.actions[-1] == 0).all()
    latest = torch.zeros_like(rollout.initial_noise)
    available = torch.zeros(5, dtype=torch.bool)
    prediction = iter(predictions)
    for index, frame in enumerate(rollout.trajectory[:-1]):
        execute = rollout.actions[index] == 0
        if execute.any():
            latest[execute] = next(prediction)
            available |= execute
        assert_equal(frame["available"], available[:4])
        assert_equal(frame["nfe"], (rollout.actions[: index + 1, :4] == 0).sum(0))
        if available[:4].any():
            assert_equal(frame["images"][available[:4]], model.decode(latest[:4][available[:4]]))
        else:
            assert frame["images"] is None
    assert_equal(rollout.trajectory[2]["images"][[0, 2]], rollout.trajectory[1]["images"][[0, 2]])
    assert_equal(rollout.trajectory[3]["images"][:3], rollout.trajectory[2]["images"][:3])
    assert_equal(rollout.trajectory[-1]["images"], rollout.images[:4])
    save_figure = Mock()
    monkeypatch.setattr(visualization, "_save_figure", save_figure)
    visualization.save_trajectory(rollout.trajectory, tmp_path / "trajectory.png")
    figure = save_figure.call_args.args[0]
    try:
        missing = sum(int((~frame["available"]).sum()) for frame in rollout.trajectory)
        assert (
            sum(
                text.get_text() == "No prediction yet"
                for axis in figure.axes
                for text in axis.texts
            )
            == missing
        )
        assert all(figure.axes[row * 6 + 5].get_title().startswith("Final:") for row in range(4))
        assert "skips hold the last estimate" in figure._supxlabel.get_text()
    finally:
        figure.clear()


class OfflinePairedDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0))
        self.inputs = []

    def forward(self, generated, reference):
        self.inputs.append((generated.clone(), reference.clone()))
        return (generated - reference).abs().mean((1, 2, 3), keepdim=True) * self.scale


@pytest.mark.parametrize(
    "metric,samples,batch_size,channels,render",
    [
        ("none", 17, 8, 1, True),
        ("none", 1, 1, 3, False),
        ("mse", 5, 2, 1, True),
        ("lpips", 3, 2, 1, False),
        ("lpips", 3, 2, 3, False),
    ],
)
def test_evaluation_saves_exact_pairs_pixels_and_metrics_without_extra_denoising(
    tmp_path, monkeypatch, model, metric, samples, batch_size, channels, render
):
    model = DiffusionModel(ToyDenoiser(channels), copy.deepcopy(model.scheduler), n_timesteps=6)
    policy = PolicyNetwork(channels, hidden_dim=8, feature_dim=8, max_diffusion_steps=100)
    lpips_model = OfflinePairedDistance()
    lpips_factory = Mock(return_value=lpips_model)
    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=lpips_factory))
    reference_sample = model.sample
    references = []

    def sample(initial, steps, labels=None):
        assert metric != "none", "metric=none must not run the reference sampler"
        result = reference_sample(initial, steps, labels)
        references.append((initial.clone(), result.clone()))
        return result

    monkeypatch.setattr(model, "sample", sample)
    predictions = count_predictions(monkeypatch, model)
    collected = []

    def collect(*args, **kwargs):
        assert kwargs["record_observations"] is False
        rollout = collect_rollout(*args, **kwargs)
        collected.append(rollout)
        return rollout

    monkeypatch.setattr(evaluation_module, "collect_rollout", collect)
    plots = {}
    for name in (
        "save_image_grid",
        "save_action_schedule",
        "save_nfe_histogram",
        "save_trajectory",
        "save_comparison",
        "save_quality_plot",
    ):
        plots[name] = Mock(wraps=getattr(evaluation_module, name) if render else None)
        monkeypatch.setattr(evaluation_module, name, plots[name])
    arguments = dict(
        num_samples=samples,
        batch_size=batch_size,
        steps=6,
        budget=3,
        budget_mode="exact",
        reference_steps=6,
        seed=91,
        metric=metric,
    )
    baseline = evaluation_module.evaluate(model, policy, **arguments)
    baseline_rollouts = collected.copy()
    baseline_predictions = predictions.copy()
    baseline_rng = rng_state()
    collected.clear()
    predictions.clear()
    references.clear()
    lpips_model.inputs.clear()
    output = tmp_path / "evaluation"
    summary = evaluation_module.evaluate(model, policy, **arguments, output_dir=output)
    assert {
        key: value for key, value in summary.items() if key != "wall_seconds_including_reference"
    } == {
        key: value for key, value in baseline.items() if key != "wall_seconds_including_reference"
    }
    assert_equal(rng_state(), baseline_rng)
    assert predictions == baseline_predictions
    assert predictions == expected_prediction_batches(collected, 0 if metric == "none" else 6)
    for actual, expected in zip(collected, baseline_rollouts):
        assert_same_rollout(actual, expected)
    assert bool(collected[0].trajectory)
    assert all(not rollout.trajectory for rollout in collected[1:])
    files = sorted(output.glob("samples_*.pt"))
    assert [path.name for path in files] == [
        f"samples_{index:06d}.pt" for index in range(0, samples, batch_size)
    ]
    assert sorted(path.name for path in (output / "images").iterdir()) == [
        f"{index:06d}.png" for index in range(samples)
    ]
    distances = []
    for batch_index, (path, rollout) in enumerate(zip(files, collected)):
        data = torch.load(path, weights_only=True)
        first = batch_index * batch_size
        count = min(batch_size, samples - first)
        assert data["first_sample"] == first and data["metric"] == metric
        assert data["images"].shape == (count, channels, 8, 8)
        assert_equal({key: data[key] for key in rollout.tensor_payload()}, rollout.tensor_payload())
        expected_noise = torch.stack(
            [
                torch.randn(model.sample_shape, generator=torch.Generator().manual_seed(91 + index))
                for index in range(first, first + count)
            ]
        )
        assert_equal(data["initial_noise"], expected_noise)
        assert_equal(data["nfe"], (data["actions"] == 0).sum(0))
        for index, pixels in enumerate(expected_pixels(data["images"]), start=first):
            assert_png(output / "images" / f"{index:06d}.png", pixels)
        if metric == "none":
            assert data["reference_images"] is None and data["distance"] is None
        else:
            assert_equal(references[batch_index][0], expected_noise)
            assert_equal(data["reference_images"], references[batch_index][1])
            if metric == "mse":
                distance = (data["images"] - data["reference_images"]).square().flatten(1).mean(1)
            else:
                generated = data["images"].clamp(-1, 1).expand(-1, 3, -1, -1)
                reference = data["reference_images"].clamp(-1, 1).expand(-1, 3, -1, -1)
                assert_equal(lpips_model.inputs[batch_index], (generated, reference))
                distance = (generated - reference).abs().mean((1, 2, 3))
            assert data["distance"].shape == (count,)
            assert_equal(data["distance"], distance)
            distances.append(distance)
    assert any((rollout.images.abs() > 1).any() for rollout in collected)
    assert len(references) == (0 if metric == "none" else len(files))
    assert json.loads((output / "metrics.json").read_text()) == summary
    if metric == "none":
        assert summary["distance_mean"] is None
        assert summary["metric_model_sha256"] is None
        assert summary["reference_nfe_per_sample"] == 0
        lpips_factory.assert_not_called()
        for name, filename in (
            ("save_comparison", "comparison.png"),
            ("save_quality_plot", "quality_vs_nfe.png"),
        ):
            plots[name].assert_not_called()
            assert not (output / filename).exists()
    else:
        assert summary["distance_mean"] == float(torch.cat(distances).mean())
        assert summary["reference_nfe_per_sample"] == 6
        plots["save_comparison"].assert_called_once()
        plots["save_quality_plot"].assert_called_once()
        assert_equal(plots["save_quality_plot"].call_args.args[1], torch.cat(distances))
        if metric == "lpips":
            assert len(summary["metric_model_sha256"]) == 64
            assert lpips_factory.call_count == 2
            assert lpips_factory.call_args.kwargs == {"net": "alex", "verbose": False}
        else:
            lpips_factory.assert_not_called()
            assert summary["metric_model_sha256"] is None
    for name in (
        "save_image_grid",
        "save_action_schedule",
        "save_nfe_histogram",
        "save_trajectory",
    ):
        plots[name].assert_called_once()
    if render:
        expected_plots = {
            "samples.png",
            "action_schedule.png",
            "nfe_histogram.png",
            "trajectory.png",
        }
        if metric != "none":
            expected_plots |= {"comparison.png", "quality_vs_nfe.png"}
        assert {path.name for path in output.glob("*.png")} == expected_plots
        for filename in expected_plots:
            assert_png(output / filename)


def test_plot_axes_show_recorded_data_integer_ticks_and_nonduplicate_reward_legend(
    monkeypatch, tmp_path
):
    save_figure = Mock()
    monkeypatch.setattr(visualization, "_save_figure", save_figure)
    actions = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    forced = torch.tensor([[False, True], [False, False], [True, True]])
    visualization.save_action_schedule(actions, forced, tmp_path / "actions.png")
    action_figure = save_figure.call_args.args[0]
    np.testing.assert_array_equal(action_figure.axes[0].images[0].get_array(), actions.T.numpy())
    step, sample = forced.nonzero(as_tuple=True)
    np.testing.assert_array_equal(
        action_figure.axes[0].collections[0].get_offsets(),
        np.column_stack((step.numpy() + 1, sample.numpy())),
    )
    action_figure.clear()
    nfe = torch.tensor([3.0, 3.0, 3.0])
    distance = torch.tensor([0.25, 0.5, 0.75])
    visualization.save_nfe_histogram(nfe, tmp_path / "nfe.png")
    histogram = save_figure.call_args.args[0]
    assert [patch.get_height() for patch in histogram.axes[0].patches] == [3]
    visualization.save_quality_plot(nfe, distance, "mse", tmp_path / "quality.png")
    quality = save_figure.call_args.args[0]
    np.testing.assert_array_equal(
        quality.axes[0].collections[0].get_offsets(), np.column_stack((nfe, distance))
    )
    assert quality.axes[0].get_ylabel() == "Paired MSE distance"
    history = [
        {
            "episode": 0,
            "reward": 1.0,
            "loss": 0.2,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "nfe_mean": 3.0,
            "nfe_min": 3,
            "nfe_max": 3,
            "entropy": 0.3,
            "kl": 0.0,
            "diffusion_loss": 0.0,
            "gradient_norm": 0.5,
            "forced_fraction": 0.5,
            "reward_color": 0.25,
            "reward_raw_color": 0.25,
            "reward_cost": 0.0,
            "reward_raw_cost": 0.0,
        }
    ]
    original = copy.deepcopy(history)
    visualization.save_training_curves(history, tmp_path / "curves.png")
    curves = save_figure.call_args.args[0]
    try:
        reward_axis = curves.axes[-1]
        assert "unweighted" in reward_axis.get_title().lower()
        assert reward_axis.get_legend_handles_labels()[1] == ["color"]
        assert_equal(history, original)
        for figure, point in ((histogram, 3), (quality, 3), (curves, 1)):
            for axis in figure.axes:
                low, high = axis.get_xlim()
                assert low < point < high
                np.testing.assert_array_equal(axis.get_xticks(), np.rint(axis.get_xticks()))
    finally:
        for figure in (histogram, quality, curves):
            figure.clear()


TRAINING_CASES = [
    pytest.param(algorithm, kind, False, id=f"{algorithm}-{kind}")
    for algorithm in ("a2c_simple", "a2c_gae", "ppo", "grpo")
    for kind in ("lstm", "direct", "diffusion")
] + [
    pytest.param(algorithm, "lstm", True, id=f"{algorithm}-tbptt")
    for algorithm in ("a2c_simple", "a2c_gae", "ppo", "grpo")
]


@pytest.mark.parametrize("algorithm,kind,tbptt", TRAINING_CASES)
def test_training_output_and_checkpoint_resume_are_exactly_neutral(
    tmp_path, monkeypatch, model, policy_factory, reward, algorithm, kind, tbptt
):
    mock_training_plots(monkeypatch)

    def create():
        seed_everything(81)
        trainer_type = TBPTTRLTrainer if tbptt else RLTrainer
        trainer = trainer_type(
            copy.deepcopy(model),
            policy_factory(kind),
            copy.deepcopy(reward),
            algo=algorithm,
            inference_steps=6,
            num_episodes=3,
            denoising_budget=3,
            grpo_group_size=2,
            diffusion_loss_weight=0.01 if kind == "diffusion" else 0,
            **({"tbptt_chunk_size": 2} if tbptt else {}),
        )
        return trainer, count_predictions(monkeypatch, trainer.diffusion_model)

    baseline, baseline_calls = create()
    initial_weights = copy.deepcopy(baseline.policy_network.state_dict())
    baseline.train_step(4)
    first_weights = copy.deepcopy(baseline.policy_network.state_dict())
    first_rng = rng_state()
    baseline.train_step(4)
    baseline.train_step(4)
    expected_rng = rng_state()
    assert any(
        not torch.equal(value, initial_weights[name])
        for name, value in baseline.policy_network.state_dict().items()
        if "value_head" not in name
    )
    output = tmp_path / "training"
    interrupted, interrupted_calls = create()
    interrupted.train_step(4, checkpoint_dir=output)
    assert interrupted.history == baseline.history[:1]
    assert_equal(interrupted.policy_network.state_dict(), first_weights)
    assert_equal(rng_state(), first_rng)
    first_checkpoint = torch.load(output / "latest.pt", weights_only=True)
    assert_equal(first_checkpoint["rng_state"], first_rng)
    resumed, resumed_calls = create()
    resumed.load_checkpoint(output / "latest.pt")
    assert_equal(rng_state(), first_rng)
    resumed.train_step(4, checkpoint_dir=output)
    resumed.train_step(4, checkpoint_dir=output)
    assert resumed.history == baseline.history
    assert_equal(resumed.policy_network.state_dict(), baseline.policy_network.state_dict())
    assert_equal(resumed.optimizer.state_dict(), baseline.optimizer.state_dict())
    assert_equal(resumed.scheduler.state_dict(), baseline.scheduler.state_dict())
    assert_equal(resumed.reward_function.state_dict(), baseline.reward_function.state_dict())
    assert_equal(rng_state(), expected_rng)
    assert interrupted_calls + resumed_calls == baseline_calls
    assert json.loads((output / "history.json").read_text()) == {"episodes": baseline.history}
    checkpoint = torch.load(output / "latest.pt", weights_only=True)
    assert checkpoint["episode"] == 3 and checkpoint["history"] == baseline.history
    assert_equal(checkpoint["rng_state"], expected_rng)
    assert_equal(checkpoint["policy_network_state_dict"], baseline.policy_network.state_dict())
    if resumed.reference_policy is not None:
        assert_equal(resumed.reference_policy.state_dict(), baseline.reference_policy.state_dict())
    for name in (
        "save_training_curves",
        "save_image_grid",
        "save_action_schedule",
        "save_trajectory",
    ):
        assert getattr(visualization, name).call_count == 3


def test_real_training_render_preserves_updates_rng_and_pre_update_raw_pairs(
    tmp_path, monkeypatch, model, policy_factory, perceptual_reward
):
    def create():
        seed_everything(19)
        trainer = RLTrainer(
            copy.deepcopy(model),
            policy_factory(),
            copy.deepcopy(perceptual_reward),
            inference_steps=6,
            denoising_budget=3,
            num_episodes=2,
        )
        return trainer, count_predictions(monkeypatch, trainer.diffusion_model)

    baseline, baseline_calls = create()
    baseline.train_step(4)
    expected_rng = rng_state()
    rendered, render_calls = create()
    collect = rendered.collect
    captured = []

    def record(*args, **kwargs):
        result = collect(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(rendered, "collect", record)
    output = tmp_path / "training"
    rendered.train_step(4, checkpoint_dir=output)
    assert rendered.history == baseline.history
    assert_equal(rendered.policy_network.state_dict(), baseline.policy_network.state_dict())
    assert_equal(rendered.optimizer.state_dict(), baseline.optimizer.state_dict())
    assert_equal(rng_state(), expected_rng)
    assert render_calls == baseline_calls
    assert render_calls == expected_prediction_batches(captured, reference_steps=6)
    data = torch.load(output / "training_samples.pt", weights_only=True)
    assert data["phase"] == "pre_update_rollout" and data["episode"] == 0
    assert_equal(
        {key: data[key] for key in captured[0].tensor_payload()}, captured[0].tensor_payload()
    )
    assert_equal(data["trajectory"][-1]["images"], data["images"][:4])
    assert data["reference_images"] is not None
    checkpoint = torch.load(output / "latest.pt", weights_only=True)
    assert_equal(checkpoint["rng_state"], expected_rng)
    assert_equal(checkpoint["policy_network_state_dict"], rendered.policy_network.state_dict())
    expected_plots = {
        "training_curves.png",
        "training_samples.png",
        "training_actions.png",
        "training_trajectory.png",
        "training_comparison.png",
    }
    assert {path.name for path in output.glob("*.png")} == expected_plots
    for filename in expected_plots:
        assert_png(output / filename)
