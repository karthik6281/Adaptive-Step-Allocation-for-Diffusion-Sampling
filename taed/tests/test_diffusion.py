from types import SimpleNamespace

import pytest
import torch
from diffusers import DDIMScheduler
from torch import nn

from adapter_compvis import CompVisSchedulerAdapter, CompVisUNetAdapter, resolve_paths
from models.diffusion import DiffusionModel, ToyDenoiser, ddim_transition


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "v_prediction"])
@pytest.mark.parametrize("clip", [False, True])
def test_ddim_matches_diffusers_on_uniform_grid(prediction_type, clip):
    scheduler = DDIMScheduler(
        num_train_timesteps=100, prediction_type=prediction_type, clip_sample=clip
    )
    scheduler.set_timesteps(10)
    sample = torch.randn(2, 1, 8, 8)
    prediction = torch.randn_like(sample)
    for time in scheduler.timesteps:
        previous = int(time) - 10
        end = scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.final_alpha_cumprod
        actual = ddim_transition(
            sample, prediction, scheduler.alphas_cumprod[time], end, prediction_type, clip
        )[0]
        expected = scheduler.step(prediction, time, sample).prev_sample
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("start, end", [(0.2, 0.7), (0.01, 0.99), (0.2, 1.0)])
def test_ddim_arbitrary_jump_recovers_known_clean_signal(start, end):
    clean, noise = (
        torch.randn(3, 1, 8, 8, dtype=torch.float64),
        torch.randn(3, 1, 8, 8, dtype=torch.float64),
    )
    alpha_start, alpha_end = (
        torch.tensor(start, dtype=torch.float64),
        torch.tensor(end, dtype=torch.float64),
    )
    state = alpha_start.sqrt() * clean + (1 - alpha_start).sqrt() * noise
    moved, predicted_noise, predicted_clean = ddim_transition(state, noise, alpha_start, alpha_end)
    expected = alpha_end.sqrt() * clean + (1 - alpha_end).sqrt() * noise
    torch.testing.assert_close(moved, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(predicted_clean, clean, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(predicted_noise, noise)


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 50, 200, 400, 667, 999, 1000])
def test_compvis_grid_has_exact_count_and_valid_indices(steps):
    model = SimpleNamespace(num_timesteps=1000, alphas_cumprod=torch.linspace(0.99, 0.01, 1000))
    scheduler = CompVisSchedulerAdapter(model)
    scheduler.set_timesteps(steps)
    assert scheduler.timesteps.numel() == steps
    assert scheduler.timesteps.unique().numel() == steps
    assert torch.all(scheduler.timesteps[1:] < scheduler.timesteps[:-1])
    assert int(scheduler.timesteps.min()) >= 0
    assert int(scheduler.timesteps.max()) < 1000
    if steps == 200:
        torch.testing.assert_close(scheduler.timesteps, (torch.arange(0, 1000, 5) + 1).flip(0))
    torch.testing.assert_close(scheduler.final_alpha_cumprod, model.alphas_cumprod[0])


@pytest.mark.parametrize("steps", [0, -1, 1001, 2.5])
def test_invalid_compvis_grid_is_rejected(steps):
    scheduler = CompVisSchedulerAdapter(
        SimpleNamespace(num_timesteps=1000, alphas_cumprod=torch.ones(1000))
    )
    with pytest.raises(ValueError):
        scheduler.set_timesteps(steps)


def test_nonuniform_sampling_uses_actual_destinations(model, monkeypatch):
    seen = []
    transition = model.transition

    def record(state, start, end, class_labels=None):
        seen.append((int(start[0]), int(end[0])))
        return transition(state, start, end, class_labels)

    monkeypatch.setattr(model, "transition", record)
    noise = torch.randn(2, *model.sample_shape)
    output = model.sample(noise, 7)
    grid = model.set_timesteps(7).tolist()
    assert seen == list(zip(grid, grid[1:] + [-1]))
    assert output.shape == noise.shape


def test_batched_jumps_match_individual_jumps(model):
    noise = torch.randn(2, *model.sample_shape)
    starts, ends = torch.tensor([99, 70]), torch.tensor([50, -1])
    batched = model.transition(noise, starts, ends)[0]
    separate = torch.cat(
        [
            model.transition(noise[i : i + 1], starts[i : i + 1], ends[i : i + 1])[0]
            for i in range(2)
        ]
    )
    torch.testing.assert_close(batched, separate)


def test_conditional_compvis_does_not_default_to_class_zero():
    class Conditional(nn.Module):
        def __init__(self):
            super().__init__()
            self.channels, self.image_size = 4, 8
            self.cond_stage_key = "class_label"
            self.cond_stage_model = nn.Module()
            self.cond_stage_model.key = "class_label"
            self.cond_stage_model.embedding = nn.Embedding(3, 4)

        def get_learned_conditioning(self, values):
            return values["class_label"]

        def apply_model(self, sample, timestep, conditioning):
            return conditioning[:, None, None, None].expand_as(sample).float()

    adapter = CompVisUNetAdapter(Conditional())
    sample, timestep = torch.zeros(2, 4, 8, 8), torch.tensor([99, 99])
    with pytest.raises(ValueError, match="class label"):
        adapter(sample, timestep)
    result = adapter(sample, timestep, torch.tensor([1, 2])).sample
    assert torch.all(result[0] == 1)
    assert torch.all(result[1] == 2)


def test_missing_assets_have_no_machine_specific_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required model file"):
        resolve_paths("church256", compvis_root=tmp_path)


def test_diffusion_model_rejects_unconditional_labels():
    model = DiffusionModel(ToyDenoiser(), DDIMScheduler(num_train_timesteps=100), 6)
    with pytest.raises(ValueError, match="unconditional"):
        model.transition(
            torch.zeros(1, 1, 8, 8), torch.tensor([99]), torch.tensor([-1]), torch.tensor([0])
        )
