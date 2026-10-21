import importlib

import pytest
import torch

from dasp.adapters.policy_adapter import DATASETS
from dasp.diffusion_transformer.transformer_for_diffusion import TransformerForDiffusion
from dasp.encoder.custom_encoder import CustomEncoder
from dasp.policy.diffusion_policy import prob_to_bounded_logit


@pytest.mark.parametrize("kind", ["direct", "diffusion"])
def test_plan_shapes_gradient_flow_and_class_conditioning(policy_factory, kind):
    policy = policy_factory(kind, classes=3).eval()
    observations = {
        "image": torch.randn(2, 1, 8, 8),
        "timestep": torch.tensor([90, 70]),
        "class_labels": torch.tensor([0, 1]),
        "budget_ratio": torch.tensor([0.3, 0.9]),
        "jump_ratio": torch.tensor([0.1, 0.2]),
        "progress_ratio": torch.tensor([0.1, 0.7]),
        "rate_ratio": torch.tensor([0.2, 0.5]),
    }
    probabilities, values = policy.plan(observations)
    assert probabilities.shape == (2, 4)
    assert values.shape == (2,)
    assert torch.all((probabilities > 0) & (probabilities < 1))
    (probabilities.sum() + values.sum()).backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in policy.encoder.parameters()
    )
    with pytest.raises(ValueError, match="class label"):
        policy.plan({"image": observations["image"], "timestep": observations["timestep"]})


def test_direct_head_is_repeatable_and_uses_images(policy_factory):
    policy = policy_factory("direct").eval()
    observation = {"image": torch.randn(2, 1, 8, 8), "timestep": torch.tensor([50, 50])}
    first = policy.plan(observation)[0]
    second = policy.plan(observation)[0]
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not torch.equal(first[0], first[1])


def test_action_diffusion_generator_controls_randomness(policy_factory):
    policy = policy_factory("diffusion").eval()
    base = {"image": torch.randn(2, 1, 8, 8), "timestep": torch.tensor([50, 50])}
    first = policy.plan({**base, "generator": torch.Generator().manual_seed(2)})[0]
    second = policy.plan({**base, "generator": torch.Generator().manual_seed(2)})[0]
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_auxiliary_loss_masks_padding_and_checks_inputs(policy_factory):
    policy = policy_factory("diffusion").policy
    observations = {"image": torch.randn(2, 1, 8, 8), "timestep": torch.tensor([50, 50])}
    actions = torch.zeros(2, 4)
    mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
    torch.manual_seed(5)
    first = policy.compute_loss({"observations": observations, "actions": actions, "mask": mask})
    actions[:, 2:] = 1
    torch.manual_seed(5)
    second = policy.compute_loss({"observations": observations, "actions": actions, "mask": mask})
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    # Invalid masks must fail rather than return a trainable scalar zero.
    with pytest.raises(ValueError, match="valid action"):
        policy.compute_loss(
            {"observations": observations, "actions": actions, "mask": torch.zeros_like(mask)}
        )
    assert torch.isfinite(first) and first.requires_grad
    with pytest.raises(ValueError, match="action_horizon"):
        policy.compute_loss({"observations": observations, "actions": actions[:, :1]})


@pytest.mark.parametrize("channels", [[9, 25], [4, 8]])
@pytest.mark.parametrize("adaptive", [True, False])
def test_encoder_projection_and_group_norm_for_nonstandard_shapes(channels, adaptive):
    encoder = CustomEncoder(
        {
            "input_channels": 1,
            "image_size": 15,
            "channels": channels,
            "strides": [2, 2],
            "use_group_norm": True,
            "output_dim": 7,
            "use_adaptive_pool": adaptive,
            "dropout_rate": 0,
        }
    )
    assert encoder(torch.randn(2, 1, 15, 15)).shape == (2, 7)


def test_mismatched_encoder_configuration_is_rejected():
    with pytest.raises(ValueError, match="equally sized"):
        CustomEncoder({"channels": [4, 8], "strides": [2]})
    with pytest.raises(ValueError, match="Unknown activation"):
        CustomEncoder({"channels": [4], "activation": "misspelled"})


@pytest.mark.parametrize("dataset", DATASETS)
def test_all_retained_dataset_builders_construct_matching_encoders(dataset):
    module = importlib.import_module(f"dasp.config.config_{dataset}")
    encoder, policy = module.build_complete_system(
        obs_feature_dim=8,
        n_emb=8,
        n_head=2,
        n_layer=1,
        n_cond_layers=0,
        action_horizon=4,
        planning_interval=2,
        num_inference_steps=4,
    )
    assert policy.visual_encoder is encoder
    configuration = module.ENCODER_CONFIG
    inputs = torch.randn(
        2, configuration["input_channels"], configuration["image_size"], configuration["image_size"]
    )
    assert encoder(inputs).shape == (2, 8)


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("time_as_cond", [False, True])
def test_transformer_variable_horizon_and_optimizer_coverage(conditioned, time_as_cond):
    if conditioned and not time_as_cond:
        with pytest.raises(ValueError, match="time token"):
            TransformerForDiffusion(1, 1, horizon=5, cond_dim=4, time_as_cond=False)
        return
    model = TransformerForDiffusion(
        1,
        1,
        horizon=5,
        n_obs_steps=1,
        cond_dim=4 if conditioned else 0,
        n_layer=1,
        n_head=2,
        n_emb=8,
        causal_attn=True,
        time_as_cond=time_as_cond,
    )
    condition = torch.randn(2, 1, 4) if conditioned else None
    result = model(torch.randn(2, 3, 1), 2, condition)
    assert result.shape == (2, 3, 1)
    groups = model.get_optim_groups()
    flattened = [id(value) for group in groups for value in group["params"]]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == {id(value) for value in model.parameters()}


def test_bounded_logits_have_finite_endpoint_values():
    result = prob_to_bounded_logit(torch.tensor([0.0, 0.5, 1.0]))
    assert torch.isfinite(result).all()
    assert -5 < result[0] < 0 < result[2] < 5
    assert result[1] == 0
