import copy

import pytest
import torch
from PIL import Image
from torch import nn

from tools.build_manifold_bank import build_bank
from training.advanced_reward import AdvancedRewardV3
from training.rl_trainer import RLTrainer
from utils.reproducibility import model_sha256


class MeanEncoder(nn.Module):
    def forward(self, images):
        return images.mean((-1, -2))


@pytest.mark.parametrize("conditional", [False, True])
def test_feature_bank_builds_deterministically_and_round_trips(tmp_path, conditional):
    image_root = tmp_path / "images"
    for label, color in enumerate(("red", "blue")):
        folder = image_root / str(label) if conditional else image_root
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10 + label, 12), color=color).save(folder / f"{label}.png")
    encoder = MeanEncoder()
    bank = build_bank(
        image_root, "resnet18", batch_size=1, class_folders=conditional, encoder=encoder
    )
    second = build_bank(
        image_root, "resnet18", batch_size=2, class_folders=conditional, encoder=encoder
    )
    assert bank["encoder_state_sha256"] == model_sha256(encoder)
    assert bank["images_sha256"] == second["images_sha256"]
    if conditional:
        assert sorted(bank["class_banks"]) == [0, 1]
        for label in (0, 1):
            torch.testing.assert_close(bank["class_banks"][label], second["class_banks"][label])
    else:
        assert bank["embeds"].shape == (2, 3)
        torch.testing.assert_close(bank["embeds"], second["embeds"])
    path = tmp_path / "bank.pt"
    torch.save(bank, path)
    reward = AdvancedRewardV3(
        w_manifold=1,
        manifold_bank_path=path,
        manifold_encoder="resnet18",
        manifold_model=MeanEncoder(),
    )
    assert reward.configuration["manifold_bank_sha256"]


def test_feature_bank_rejects_empty_dataset_and_wrong_encoder_weights(tmp_path):
    with pytest.raises(ValueError, match="No supported images"):
        build_bank(tmp_path, encoder=MeanEncoder())
    path = tmp_path / "bad-bank.pt"
    torch.save(
        {"encoder": "resnet18", "embeds": torch.eye(3), "encoder_state_sha256": "incorrect"}, path
    )
    with pytest.raises(ValueError, match="different encoder weights"):
        AdvancedRewardV3(
            w_manifold=1,
            manifold_bank_path=path,
            manifold_encoder="resnet18",
            manifold_model=MeanEncoder(),
        )


def test_resume_rejects_changed_reward_configuration(tmp_path, model, policy_factory, reward):
    trainer = RLTrainer(model, policy_factory(), reward, inference_steps=6)
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path)
    changed = AdvancedRewardV3(w_color=0.001, w_cost=0.1)
    resumed = RLTrainer(model, policy_factory(), changed, inference_steps=6)
    with pytest.raises(ValueError, match="reward assets"):
        resumed.load_checkpoint(path)


def test_resume_rejects_changed_source_fingerprint(tmp_path, model, policy_factory, reward):
    trainer = RLTrainer(model, policy_factory(), reward, inference_steps=6)
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    payload["provenance"]["source_sha256"]["run_rl.py"] = "changed"
    torch.save(payload, path)
    resumed = RLTrainer(model, policy_factory(), copy.deepcopy(reward), inference_steps=6)
    with pytest.raises(ValueError, match="source"):
        resumed.load_checkpoint(path)
