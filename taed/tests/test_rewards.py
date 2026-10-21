import pytest
import torch
from torch import nn

from training.advanced_reward import (
    AdvancedRewardV3,
    GPURunningStats,
    _ClassConditionalManifoldBank,
    _ManifoldBank,
)


def test_population_statistics_merge_and_freeze():
    stats = GPURunningStats(warmup_samples=4)
    stats.update_batch(torch.tensor([1.0]))
    stats.update_batch(torch.tensor([2.0]))
    stats.update_batch(torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(stats.mean, torch.tensor(2.5))
    torch.testing.assert_close(stats.variance, torch.tensor(1.25))
    stats.update_batch(torch.tensor([1000.0]))
    assert stats.count == 4
    torch.testing.assert_close(stats.mean, torch.tensor(2.5))
    assert torch.isfinite(stats.normalize_batch(torch.tensor([1.0]))).all()


@pytest.mark.parametrize("batch", [1, 3])
def test_perceptual_delta_telescopes_and_preserves_batch_shape(perceptual_reward, batch):
    first, middle, last, reference = [torch.randn(batch, 1, 8, 8).clamp(-1, 1) for _ in range(4)]
    actions = torch.zeros(batch)
    a = perceptual_reward.compute_comprehensive_rewards(middle, reference, actions, x_pre=first)
    b = perceptual_reward.compute_comprehensive_rewards(last, reference, actions, x_pre=middle)
    expected = perceptual_reward._compute_perceptual_reward(
        last, reference
    ) - perceptual_reward._compute_perceptual_reward(first, reference)
    assert a["total"].shape == (batch,)
    torch.testing.assert_close(a["total"] + b["total"], expected)


def test_reference_and_shape_errors_are_not_zero_rewards(perceptual_reward):
    images = torch.zeros(1, 1, 8, 8)
    for reference in (None, torch.zeros(1, 3, 8, 8)):
        with pytest.raises(ValueError, match="reference"):
            perceptual_reward.compute_comprehensive_rewards(
                images, reference, torch.zeros(1), x_pre=images
            )


def test_reward_nan_raises_instead_of_sanitizing(reward):
    images = torch.zeros(1, 1, 8, 8)
    images[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        reward.compute_comprehensive_rewards(images, None, torch.zeros(1))


def test_cost_is_per_sample_and_not_normalized():
    reward = AdvancedRewardV3(w_cost=0.2, use_normalization=True)
    result = reward.compute_comprehensive_rewards(
        torch.zeros(3, 1, 8, 8), None, torch.tensor([0.0, 1.0, 0.0])
    )
    torch.testing.assert_close(result["total"], torch.tensor([-0.2, 0.0, -0.2]))


def test_tv_is_per_image_not_per_batch():
    flat = torch.zeros(1, 1, 4, 4)
    stripes = flat.clone()
    stripes[..., ::2] = 1
    reward = AdvancedRewardV3(w_color=1)
    result = reward.compute_comprehensive_rewards(torch.cat((flat, stripes)), None, torch.ones(2))
    torch.testing.assert_close(result["color"], torch.tensor([0.0, -12.0]))


@pytest.mark.parametrize("size", [8, 28, 32, 64])
def test_quality_svd_is_active_and_finite_for_small_images(size):
    reward = AdvancedRewardV3(w_quality=1)
    image = torch.zeros(1, 1, size, size)
    actual = reward.compute_sparse_quality(image, timestep_ratio=0.0, return_raw=True)
    assert actual.shape == (1,)
    assert torch.isfinite(actual).all()
    assert actual.item() > 40


def test_quality_handles_per_sample_noise_ratios():
    reward = AdvancedRewardV3(w_quality=1)
    images = torch.randn(2, 3, 8, 8)
    ratios = torch.tensor([0.0, 0.8])
    batched = reward.compute_sparse_quality(images, timestep_ratio=ratios)
    individual = torch.cat(
        [
            reward.compute_sparse_quality(images[i : i + 1], timestep_ratio=float(ratios[i]))
            for i in range(2)
        ]
    )
    torch.testing.assert_close(batched, individual)


def test_focus_does_not_load_efficientnet():
    reward = AdvancedRewardV3(w_focus=1)
    assert reward.feature_extractor is None
    result = reward.compute_comprehensive_rewards(torch.zeros(1, 1, 8, 8), None, torch.ones(1))
    assert torch.isfinite(result["focus"]).all()


def test_object_confidence_increases_as_distribution_sharpens():
    reward = AdvancedRewardV3()
    uniform = reward._compute_object_reward(torch.zeros(1, 10))
    logits = torch.zeros(1, 10)
    logits[:, 0] = 100
    certain = reward._compute_object_reward(logits)
    assert 0 <= uniform.item() < certain.item() <= 1


def test_manifold_bank_exact_neighbors_and_class_selection():
    bank = _ManifoldBank(torch.tensor([[1.0, 0], [0, 1], [-1, 0]]))
    torch.testing.assert_close(bank.topk_mean_cosine(torch.tensor([[1.0, 0]]), 1), torch.ones(1))
    torch.testing.assert_close(
        bank.topk_mean_cosine(torch.tensor([[1.0, 0]]), 2), torch.tensor([0.5])
    )
    conditional = _ClassConditionalManifoldBank(
        {0: torch.tensor([[1.0, 0]]), 1: torch.tensor([[-1.0, 0]])}
    )
    query = torch.tensor([[1.0, 0], [1, 0]])
    torch.testing.assert_close(
        conditional.topk_mean_cosine(query, 1, torch.tensor([0, 1])), torch.tensor([1.0, -1])
    )
    with pytest.raises(ValueError, match="target labels"):
        conditional.topk_mean_cosine(query, 1)
    with pytest.raises(ValueError, match="No manifold bank"):
        conditional.topk_mean_cosine(query, 1, 3)


def test_manifold_loading_failure_is_fatal(tmp_path):
    with pytest.raises(FileNotFoundError):
        AdvancedRewardV3(w_manifold=1, manifold_bank_path=tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="manifold_bank_path"):
        AdvancedRewardV3(w_manifold=1)


def test_manifold_delta_is_batch_local_without_stale_cache(tmp_path):
    class Encoder(nn.Module):
        def forward(self, images):
            return images.mean((-1, -2))

    bank_path = tmp_path / "bank.pt"
    torch.save({"embeds": torch.eye(3), "encoder": "resnet18"}, bank_path)
    reward = AdvancedRewardV3(
        w_manifold=1,
        manifold_bank_path=bank_path,
        manifold_encoder="resnet18",
        manifold_model=Encoder(),
        manifold_k=1,
    )
    before, after = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    batch = reward._compute_manifold_delta(before, after)
    separate = torch.cat(
        [reward._compute_manifold_delta(before[i : i + 1], after[i : i + 1]) for i in range(2)]
    )
    torch.testing.assert_close(batch, separate)
