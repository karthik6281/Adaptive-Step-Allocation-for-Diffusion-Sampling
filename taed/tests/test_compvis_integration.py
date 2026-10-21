import importlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from adapter_compvis import CompVisAdapter
from models.policy_network import PolicyNetwork
from training.rollout import collect_rollout
from utils.visualization import save_sample_images, save_trajectory


@pytest.mark.compvis
@pytest.mark.parametrize("conditional", [False, True])
def test_tiny_compvis_checkpoint_loads_and_samples(tmp_path, monkeypatch, conditional):
    configured = os.environ.get("TAED_COMPVIS_ROOT")
    if not configured or not (Path(configured) / "ldm/util.py").is_file():
        pytest.skip("Set TAED_COMPVIS_ROOT to run the optional CompVis integration tests.")
    monkeypatch.syspath_prepend(configured)
    instantiate = importlib.import_module("ldm.util").instantiate_from_config
    configuration = {
        "model": {
            "target": "ldm.models.diffusion.ddpm.LatentDiffusion",
            "params": {
                "timesteps": 20,
                "linear_start": 0.001,
                "linear_end": 0.02,
                "image_size": 8,
                "channels": 4,
                "first_stage_key": "image",
                "cond_stage_key": "class_label" if conditional else "image",
                "conditioning_key": "crossattn" if conditional else None,
                "cond_stage_trainable": conditional,
                "use_ema": False,
                "unet_config": {
                    "target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                    "params": {
                        "image_size": 8,
                        "in_channels": 4,
                        "out_channels": 4,
                        "model_channels": 32,
                        "attention_resolutions": [],
                        "num_res_blocks": 1,
                        "channel_mult": [1],
                        "num_heads": 1,
                        "use_spatial_transformer": conditional,
                        "context_dim": 8 if conditional else None,
                    },
                },
                "first_stage_config": {
                    "target": "ldm.models.autoencoder.AutoencoderKL",
                    "params": {
                        "embed_dim": 4,
                        "lossconfig": {"target": "torch.nn.Identity"},
                        "ddconfig": {
                            "double_z": True,
                            "z_channels": 4,
                            "resolution": 8,
                            "in_channels": 3,
                            "out_ch": 3,
                            "ch": 32,
                            "ch_mult": [1],
                            "num_res_blocks": 1,
                            "attn_resolutions": [],
                            "dropout": 0.0,
                        },
                    },
                },
                "cond_stage_config": {
                    "target": "ldm.modules.encoders.modules.ClassEmbedder",
                    "params": {"embed_dim": 8, "n_classes": 3, "key": "class_label"},
                }
                if conditional
                else "__is_unconditional__",
            },
        }
    }
    config = OmegaConf.create(configuration)
    original = instantiate(config.model)
    checkpoint = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": original.state_dict()}, checkpoint)
    path = tmp_path / "model.yaml"
    OmegaConf.save(config, path)
    adapter = CompVisAdapter(path, checkpoint, n_timesteps=4, compvis_root=configured)
    noise = torch.randn(2, 4, 8, 8)
    labels = torch.tensor([0, 2]) if conditional else None
    images = adapter.sample(noise, 4, labels)
    assert images.shape == (2, 3, 8, 8)
    assert torch.isfinite(images).all()
    torch.testing.assert_close(adapter.alphas, original.alphas_cumprod)
    assert adapter.sample_shape == (4, 8, 8)
    assert adapter.terminal_alpha < 1
    predict_noise = adapter.predict_noise
    transition = adapter.transition
    prediction_batches, terminal_states = [], []

    def count_prediction(sample, timestep, class_labels=None):
        prediction_batches.append(len(sample))
        return predict_noise(sample, timestep, class_labels)

    def record_terminal(*args, **kwargs):
        result = transition(*args, **kwargs)
        if (args[2] == -1).all():
            terminal_states.append((result[0].clone(), result[2].clone()))
        return result

    monkeypatch.setattr(adapter, "predict_noise", count_prediction)
    monkeypatch.setattr(adapter, "transition", record_terminal)
    policy = PolicyNetwork(
        4,
        hidden_dim=8,
        feature_dim=8,
        max_diffusion_steps=20,
        num_classes=3 if conditional else None,
    )
    original_noise = noise.clone()
    rollout = collect_rollout(
        adapter,
        policy,
        None,
        2,
        4,
        initial_noise=noise,
        class_labels=labels,
        budget=4,
        budget_mode="exact",
        stochastic=False,
        capture_trajectory=True,
    )
    assert prediction_batches == [2] * 4
    torch.testing.assert_close(noise, original_noise, rtol=0, atol=0)
    torch.testing.assert_close(rollout.images, images, rtol=0, atol=0)
    assert len(terminal_states) == 1
    final_latents, final_clean_estimate = terminal_states[0]
    decoded_sample = adapter.decode(final_latents)
    decoded_estimate = adapter.decode(final_clean_estimate)
    assert not torch.equal(final_latents, final_clean_estimate)
    assert not torch.allclose(decoded_sample, decoded_estimate, rtol=0, atol=1e-7)
    assert len(rollout.trajectory) == 4
    for frame in rollout.trajectory:
        assert frame["images"].shape == (2, 3, 8, 8)
        assert frame["available"].all()
    final = rollout.trajectory[-1]
    assert final["kind"] == "final_sample"
    torch.testing.assert_close(final["images"], decoded_sample, rtol=0, atol=0)
    torch.testing.assert_close(final["images"], rollout.images, rtol=0, atol=0)
    save_sample_images(rollout.images, tmp_path / "images")
    assert sorted(path.name for path in (tmp_path / "images").iterdir()) == [
        "000000.png",
        "000001.png",
    ]
    expected_pixels = (
        (rollout.images.clamp(-1, 1) + 1).mul(127.5).round().to(torch.uint8).permute(0, 2, 3, 1)
    ).numpy()
    for index, expected in enumerate(expected_pixels):
        with Image.open(tmp_path / "images" / f"{index:06d}.png") as image:
            assert image.mode == "RGB" and image.size == (8, 8)
            np.testing.assert_array_equal(np.asarray(image), expected)
    trajectory_path = tmp_path / "trajectory.png"
    save_trajectory(rollout.trajectory, trajectory_path)
    with Image.open(trajectory_path) as image:
        image.load()
        assert image.format == "PNG" and min(image.size) > 20
        assert np.ptp(np.asarray(image.convert("RGB"))) > 0
    assert prediction_batches == [2] * 4
    broken = original.state_dict()
    broken.pop(next(name for name in broken if "diffusion_model" in name))
    torch.save({"state_dict": broken}, checkpoint)
    with pytest.raises(RuntimeError, match="Checkpoint/config mismatch"):
        CompVisAdapter(path, checkpoint, n_timesteps=4, compvis_root=configured)
