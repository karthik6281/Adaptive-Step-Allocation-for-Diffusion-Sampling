import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from config.experiment import build_experiment, create_model, create_policy, load_experiment
from models.pretrained_diffusion import create_diffusers_model
from training.rollout import collect_rollout
from utils.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_cli_train_resume_and_evaluate_outside_repository(tmp_path):
    training = tmp_path / "training"
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    command = [
        sys.executable,
        str(ROOT / "run_rl.py"),
        "--config",
        str(ROOT / "examples/toy.yaml"),
        "--output-dir",
        str(training),
    ]
    first = subprocess.run(
        command + ["--stop-after", "2"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    checkpoint = training / "latest.pt"
    first_state = torch.load(checkpoint, weights_only=True)
    assert first_state["episode"] == 2
    resumed = subprocess.run(
        command + ["--resume", str(checkpoint)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert torch.load(checkpoint, weights_only=True)["episode"] == 4
    output = tmp_path / "evaluation"
    evaluated = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_eval.py"),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--num-samples",
            "5",
            "--batch-size",
            "2",
            "--metric",
            "mse",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    summary = json.loads((output / "metrics.json").read_text())
    provenance = json.loads((output / "run.json").read_text())
    assert len(provenance["policy_checkpoint_sha256"]) == 64
    assert summary["num_samples"] == 5
    assert summary["nfe_min"] == summary["nfe_max"] == 4
    assert summary["reference_nfe_per_sample"] == 8
    assert summary["distance_mean"] >= 0
    batches = [torch.load(path, weights_only=True) for path in sorted(output.glob("samples_*.pt"))]
    assert [len(batch["images"]) for batch in batches] == [2, 2, 1]
    overwrite = subprocess.run(
        command, cwd=tmp_path, env=environment, capture_output=True, text=True
    )
    assert overwrite.returncode != 0
    assert "Refusing to overwrite" in overwrite.stderr
    finished_files = {
        name: (training / name).read_bytes() for name in ("run.json", "config.json", "latest.pt")
    }
    from run_rl import main as train_cli

    with pytest.raises(ValueError, match="exceed the current episode"):
        train_cli(command[2:] + ["--resume", str(checkpoint)])
    for name, contents in finished_files.items():
        assert (training / name).read_bytes() == contents
    foreign_run = tmp_path / "foreign"
    foreign_run.mkdir()
    (foreign_run / "run.json").write_bytes(finished_files["run.json"])
    with pytest.raises(ValueError, match="from that run directory"):
        train_cli(
            [
                "--config",
                str(ROOT / "examples/toy.yaml"),
                "--output-dir",
                str(foreign_run),
                "--resume",
                str(checkpoint),
            ]
        )
    assert list(foreign_run.iterdir()) == [foreign_run / "run.json"]
    assert (foreign_run / "run.json").read_bytes() == finished_files["run.json"]


def test_cli_rejects_empty_run_before_creating_output(tmp_path):
    from run_rl import main

    output = tmp_path / "empty"
    with pytest.raises(ValueError, match="positive episode"):
        main(
            [
                "--config",
                str(ROOT / "examples/toy.yaml"),
                "--output-dir",
                str(output),
                "--stop-after",
                "0",
            ]
        )
    assert not output.exists()


@pytest.mark.parametrize("script", ["run_rl.py", "run_eval.py"])
def test_cli_help_has_no_model_or_gpu_requirements(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--output-dir" in result.stdout


@pytest.mark.parametrize("kind", ["lstm", "direct", "diffusion"])
def test_resolved_configuration_round_trips_without_changing_policy(kind):
    source = load_experiment(ROOT / "examples/toy.yaml")
    source["policy"]["type"] = kind
    if kind != "lstm":
        source["policy"].pop("hidden_dim")
        source["policy"].pop("feature_dim")
    resolved, model, first, _ = build_experiment(source)
    resolved_again, _, second, _ = build_experiment(resolved, with_reward=False)
    assert resolved["policy"] == resolved_again["policy"]
    assert resolved["model"] == resolved_again["model"]
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], atol=0, rtol=0)
    assert model.sample_shape == (1, 8, 8)


def test_local_diffusers_checkpoint_loads_and_samples_without_network(tmp_path):
    unet = UNet2DModel(
        sample_size=8,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(8, 8),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=4,
    )
    directory = tmp_path / "local-model"
    DDPMPipeline(unet=unet, scheduler=DDPMScheduler(num_train_timesteps=100)).save_pretrained(
        directory
    )
    model = create_diffusers_model(str(directory), n_timesteps=4)
    images = model.sample(torch.randn(2, *model.sample_shape), 4)
    assert images.shape == (2, 1, 8, 8)
    assert torch.isfinite(images).all()
    _, recorded = create_model({"kind": "diffusers", "model_id": str(directory)}, "mnist", "cpu", 4)
    assert recorded["asset_sha256"]
    recorded["asset_sha256"]["model_index.json"] = "not-the-real-hash"
    with pytest.raises(ValueError, match="checksums"):
        create_model(recorded, "mnist", "cpu", 4)


@pytest.mark.parametrize("revision", [None, "main", "g" * 40])
def test_remote_model_requires_real_commit_hash(revision):
    with pytest.raises(ValueError, match="revision"):
        create_diffusers_model("example/not-downloaded", revision=revision)


def test_mistyped_configuration_is_not_ignored(tmp_path):
    source = load_experiment(ROOT / "examples/toy.yaml")
    source["training"]["denoising_budegt"] = 2
    with pytest.raises(ValueError, match="Unknown training"):
        build_experiment(source)
    source = load_experiment(ROOT / "examples/toy.yaml")
    source["unexpected"] = 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Unknown experiment"):
        load_experiment(path)


def test_evaluation_backbone_identity_ignores_paths_but_not_contents():
    from run_eval import model_identity

    first = {
        "kind": "compvis",
        "checkpoint_path": "/first/model.ckpt",
        "asset_sha256": {"checkpoint": "a"},
    }
    relocated = {**first, "checkpoint_path": "/second/model.ckpt"}
    changed = {**first, "asset_sha256": {"checkpoint": "b"}}
    assert model_identity(first) == model_identity(relocated)
    assert model_identity(first) != model_identity(changed)


def test_inconsistent_model_policy_shapes_are_rejected(model):
    with pytest.raises(ValueError, match="input_channels"):
        create_policy({"type": "lstm", "input_channels": 3}, "toy", model)
    with pytest.raises(ValueError, match="input_channels"):
        create_policy({"type": "diffusion", "encoder_config": {"input_channels": 3}}, "toy", model)


def test_metric_none_is_not_reported_as_zero_lpips(model, policy_factory):
    summary = evaluate(
        model, policy_factory(), num_samples=3, batch_size=2, steps=6, budget=3, metric="none"
    )
    assert summary["distance_mean"] is None
    assert summary["reference_nfe_per_sample"] == 0


@pytest.mark.parametrize("kind", ["lstm", "direct", "diffusion"])
def test_evaluation_seed_is_self_contained(model, policy_factory, kind):
    policy = policy_factory(kind)
    first = evaluate(
        model, policy, num_samples=5, batch_size=2, steps=6, budget=3, seed=87, metric="mse"
    )
    torch.randn(100)
    second = evaluate(
        model, policy, num_samples=5, batch_size=2, steps=6, budget=3, seed=87, metric="mse"
    )
    first.pop("wall_seconds_including_reference")
    second.pop("wall_seconds_including_reference")
    assert first == second


def test_training_and_evaluation_share_warmup_and_skip_constraints(model, policy_factory):
    policy = policy_factory()
    initial = torch.randn(2, *model.sample_shape)
    trained = collect_rollout(
        model,
        policy,
        None,
        2,
        6,
        budget=5,
        warmup_ratio=0.3,
        max_consecutive_skips=1,
        initial_noise=initial,
        stochastic=False,
    )
    evaluated = collect_rollout(
        model,
        policy,
        None,
        2,
        6,
        budget=5,
        warmup_ratio=0.3,
        max_consecutive_skips=1,
        initial_noise=initial,
        stochastic=False,
    )
    torch.testing.assert_close(trained.actions, evaluated.actions, atol=0, rtol=0)
    torch.testing.assert_close(trained.images, evaluated.images, atol=0, rtol=0)


def test_reward_curriculum_initializes_future_components_without_turning_them_on():
    source = load_experiment(ROOT / "examples/toy.yaml")
    source["training"]["reward_schedule"] = [{"episode": 2, "weights": {"w_quality": 0.5}}]
    _, _, _, reward = build_experiment(copy.deepcopy(source))
    assert reward.w_quality == 0
