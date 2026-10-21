"""Paired evaluation with per-sample NFE and explicitly selected quality metrics."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from training.budget import validate_rollout_settings
from training.rollout import collect_rollout
from utils.reproducibility import (
    model_sha256,
    require_finite,
    require_int,
    save_checkpoint,
    save_json,
    seed_everything,
)
from utils.visualization import (
    save_action_schedule,
    save_comparison,
    save_image_grid,
    save_nfe_histogram,
    save_quality_plot,
    save_sample_images,
    save_trajectory,
)


@torch.no_grad()
def evaluate(
    model,
    policy,
    num_samples=64,
    batch_size=16,
    steps=200,
    budget=None,
    budget_mode="cap",
    reference_steps=None,
    seed=42,
    metric="none",
    stochastic=True,
    output_dir=None,
    warmup_ratio=0.0,
    max_consecutive_skips=None,
):
    require_int(num_samples, "num_samples")
    require_int(batch_size, "batch_size")
    require_int(steps, "steps", maximum=model.num_timesteps)
    reference_steps = steps if reference_steps is None else reference_steps
    require_int(reference_steps, "reference_steps", maximum=model.num_timesteps)
    validate_rollout_settings(steps, budget, budget_mode, warmup_ratio, max_consecutive_skips)
    if metric not in ("none", "mse", "lpips"):
        raise ValueError("metric must be 'none', 'mse', or 'lpips'")
    output = Path(output_dir) if output_dir is not None else None
    if output is not None and output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new evaluation directory.")
    seed_everything(seed, torch.are_deterministic_algorithms_enabled())
    lpips_model = None
    if metric == "lpips":
        import lpips

        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(model.device).eval()
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    all_nfe, distances = [], []
    previews = {key: [] for key in ("images", "reference_images", "nfe", "actions", "forced")}
    policy_calls = 0
    started = time.perf_counter()
    for first in range(0, num_samples, batch_size):
        count = min(batch_size, num_samples - first)
        noise, labels = [], []
        for index in range(first, first + count):
            generator = torch.Generator().manual_seed(seed + index)
            noise.append(torch.randn(model.sample_shape, generator=generator))
            if model.num_classes is not None:
                labels.append(int(torch.randint(model.num_classes, (), generator=generator)))
        initial = torch.stack(noise).to(model.device)
        class_labels = torch.tensor(labels, device=model.device) if labels else None
        rollout = collect_rollout(
            model,
            policy,
            None,
            count,
            steps,
            budget=budget,
            budget_mode=budget_mode,
            initial_noise=initial,
            class_labels=class_labels,
            reference_steps=reference_steps,
            compute_reference=metric != "none",
            stochastic=stochastic,
            warmup_ratio=warmup_ratio,
            max_consecutive_skips=max_consecutive_skips,
            capture_trajectory=output is not None and first == 0,
            record_observations=False,
        )
        all_nfe.append(rollout.nfe.cpu())
        policy_calls += len(rollout.calls) * count
        distance = None
        if metric == "mse":
            distance = (rollout.images - rollout.reference_images).square().flatten(1).mean(1)
            distances.append(require_finite(distance, "MSE metric").cpu())
        elif metric == "lpips":
            generated, reference = (
                rollout.images.clamp(-1, 1),
                rollout.reference_images.clamp(-1, 1),
            )
            if generated.shape[1] == 1:
                generated, reference = (
                    generated.expand(-1, 3, -1, -1),
                    reference.expand(-1, 3, -1, -1),
                )
            distance = lpips_model(generated, reference).reshape(count)
            distances.append(require_finite(distance, "LPIPS metric").cpu())
        if output is not None:
            data = rollout.tensor_payload()
            data.update(
                first_sample=first,
                metric=metric,
                distance=None if distance is None else distance.cpu(),
            )
            save_checkpoint(
                data,
                output / f"samples_{first:06d}.pt",
            )
            save_sample_images(data["images"], output / "images", first)
            visible = min(count, max(0, 16 - first))
            if visible:
                for name in previews:
                    if data[name] is not None:
                        value = (
                            data[name][:, :visible]
                            if name in ("actions", "forced")
                            else data[name][:visible]
                        )
                        previews[name].append(value.clone())
            if first == 0:
                save_trajectory(data["trajectory"], output / "trajectory.png")
    nfe = torch.cat(all_nfe).float()
    if output is not None:
        images = torch.cat(previews["images"])
        preview_nfe = torch.cat(previews["nfe"])
        save_image_grid(images, preview_nfe, output / "samples.png")
        save_action_schedule(
            torch.cat(previews["actions"], dim=1),
            torch.cat(previews["forced"], dim=1),
            output / "action_schedule.png",
        )
        save_nfe_histogram(nfe, output / "nfe_histogram.png")
        if previews["reference_images"]:
            save_comparison(
                images,
                torch.cat(previews["reference_images"]),
                preview_nfe,
                output / "comparison.png",
            )
            save_quality_plot(nfe, torch.cat(distances), metric, output / "quality_vs_nfe.png")
    summary = {
        "num_samples": num_samples,
        "seed": seed,
        "batch_size": batch_size,
        "inference_steps": steps,
        "budget": budget,
        "budget_mode": budget_mode,
        "warmup_ratio": warmup_ratio,
        "max_consecutive_skips": max_consecutive_skips,
        "nfe_mean": float(nfe.mean()),
        "nfe_min": int(nfe.min()),
        "nfe_max": int(nfe.max()),
        "nfe_std": float(nfe.std(unbiased=False)),
        "steps_saved_fraction": float(1 - nfe.mean() / steps),
        "metric": metric,
        "metric_model_sha256": model_sha256(lpips_model) if lpips_model is not None else None,
        "distance_mean": float(torch.cat(distances).mean()) if distances else None,
        "reference_nfe_per_sample": reference_steps if metric != "none" else 0,
        "policy_calls_per_sample": policy_calls / num_samples,
        "wall_seconds_including_reference": time.perf_counter() - started,
    }
    if output is not None:
        save_json(summary, output / "metrics.json")
    return summary
