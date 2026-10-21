"""Headless previews from collected samples, without additional denoising calls."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image

from utils.reproducibility import require_finite, require_int, save_checkpoint, save_json


def image_pixels(images):
    if images.ndim != 4 or min(images.shape) < 1 or images.shape[1] not in (1, 3):
        raise ValueError("Images must be a nonempty [batch, 1 or 3, height, width] tensor.")
    require_finite(images, "display images")
    pixels = (
        images.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    return pixels[..., 0] if images.shape[1] == 1 else pixels


def save_sample_images(images, directory, first_sample=0):
    require_int(first_sample, "first_sample", minimum=0)
    pixels = image_pixels(images)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for offset, image in enumerate(pixels):
        Image.fromarray(image).save(directory / f"{first_sample + offset:06d}.png")


def _save_figure(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        FigureCanvasAgg(figure)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as stream:
            temporary = Path(stream.name)
            figure.savefig(stream, format="png", dpi=120)
        temporary.replace(path)
    finally:
        figure.clear()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _show_image(axis, pixels, title):
    axis.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()


def save_image_grid(images, nfe, path, title="Generated samples"):
    pixels = image_pixels(images[:16])
    columns = min(4, len(pixels))
    rows = (len(pixels) + columns - 1) // columns
    figure = Figure(figsize=(2.2 * columns, 2.3 * rows), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index < len(pixels):
            _show_image(axis, pixels[index], f"Sample {index} | NFE {int(nfe[index])}")
        else:
            axis.set_axis_off()
    figure.suptitle(title)
    _save_figure(figure, path)


def save_comparison(images, reference, nfe, path, title="Paired comparison"):
    if images.shape != reference.shape:
        raise ValueError("Comparison images and references must have the same shape.")
    generated, paired = image_pixels(images[:4]), image_pixels(reference[:4])
    figure = Figure(figsize=(5, 2.1 * len(generated)), layout="constrained")
    axes = figure.subplots(len(generated), 2, squeeze=False)
    for index, (left, right) in enumerate(axes):
        _show_image(left, generated[index], f"Sample {index} | policy NFE {int(nfe[index])}")
        _show_image(right, paired[index], "Paired reference")
    figure.suptitle(title)
    _save_figure(figure, path)


def save_action_schedule(actions, forced, path, title="Executed action schedule"):
    if (
        actions.ndim != 2
        or min(actions.shape) < 1
        or actions.shape != forced.shape
        or not torch.all((actions == 0) | (actions == 1))
        or forced.dtype != torch.bool
    ):
        raise ValueError(
            "Actions and Boolean forced flags must have matching [step, batch] shapes."
        )
    actions, forced = actions[:, :16].cpu(), forced[:, :16].cpu()
    steps, count = actions.shape
    figure = Figure(figsize=(10, max(2.5, 0.3 * count + 1.5)), layout="constrained")
    axis = figure.subplots()
    axis.imshow(
        actions.T.numpy(),
        cmap=ListedColormap(["#2878b5", "#eeeeee"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
        extent=(0.5, steps + 0.5, count - 0.5, -0.5),
    )
    step, sample = forced.nonzero(as_tuple=True)
    axis.scatter(step.numpy() + 1, sample.numpy(), marker="x", c="#b22222", s=14, linewidths=0.8)
    axis.set(xlabel="Decision (1-based)", ylabel="Sample", title=title)
    axis.set_yticks(range(count))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    axis.legend(
        handles=[
            Patch(color="#2878b5", label="Denoise"),
            Patch(color="#eeeeee", label="Skip"),
            Line2D([], [], color="#b22222", marker="x", linestyle="none", label="Forced"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
    )
    _save_figure(figure, path)


def save_trajectory(frames, path, title="Sampling trajectory"):
    if not frames:
        raise ValueError("A trajectory preview requires captured frames.")
    count = len(frames[0]["available"])
    figure = Figure(figsize=(1.8 * len(frames), 2.1 * count + 0.5), layout="constrained")
    axes = figure.subplots(count, len(frames), squeeze=False)
    for column, frame in enumerate(frames):
        available = frame["available"]
        if len(available) != count or (frame["images"] is None and available.any()):
            raise ValueError("Trajectory images do not match their availability mask.")
        pixels = None if frame["images"] is None else image_pixels(frame["images"])
        for row in range(count):
            axis = axes[row, column]
            kind = "Final" if frame["kind"] == "final_sample" else f"Step {frame['step']}"
            action = "denoise" if frame["actions"][row] == 0 else "skip"
            marker = "*" if frame["forced"][row] else ""
            label = f"{kind}: {action}{marker}\nNFE {int(frame['nfe'][row])}"
            if available[row]:
                _show_image(axis, pixels[row], label)
            else:
                axis.text(0.5, 0.5, "No prediction yet", ha="center", va="center", fontsize=8)
                axis.set_title(label, fontsize=9)
                axis.set_axis_off()
    figure.suptitle(title)
    figure.supxlabel(
        "Predicted-clean estimates; skips hold the last estimate. Final = decoded sample. * Forced",
        fontsize=9,
    )
    _save_figure(figure, path)


def save_training_curves(history, path):
    if not history:
        raise ValueError("Training curves require at least one completed episode.")
    components = sorted(
        key
        for key in history[0]
        if key.startswith("reward_")
        and not key.startswith("reward_raw_")
        and any(item[key] != 0 for item in history)
    )
    panels = [
        ("Episode return", ["reward"]),
        ("Losses", ["loss", "policy_loss", "value_loss"]),
        ("Denoiser calls per sample", ["nfe_mean", "nfe_min", "nfe_max"]),
        ("Entropy and KL", ["entropy", "kl"]),
        ("Auxiliary diffusion loss", ["diffusion_loss"]),
        ("Gradient norm", ["gradient_norm"]),
        ("Forced action fraction", ["forced_fraction"]),
        ("Reward components (unweighted)", components),
    ]
    figure = Figure(figsize=(12, 11), layout="constrained")
    episodes = [item["episode"] + 1 for item in history]
    for axis, (title, keys) in zip(figure.subplots(4, 2).flat, panels):
        for key in keys:
            values = [item[key] for item in history]
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Training history {key} contains NaN or infinity.")
            axis.plot(episodes, values, marker=".", markersize=3, label=key.removeprefix("reward_"))
        axis.set(title=title, xlabel="Completed episode")
        axis.set_xlim(0.5, episodes[-1] + 0.5)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        axis.grid(alpha=0.25)
        if keys:
            axis.legend(fontsize=7, ncol=min(3, len(keys)))
    _save_figure(figure, path)


def save_nfe_histogram(nfe, path):
    counts = require_finite(nfe.detach().cpu(), "NFE histogram")
    if (
        counts.ndim != 1
        or counts.numel() < 1
        or not torch.all((counts >= 1) & (counts == counts.round()))
    ):
        raise ValueError("The NFE histogram requires positive per-sample counts.")
    figure = Figure(figsize=(6, 4), layout="constrained")
    axis = figure.subplots()
    axis.hist(counts.numpy(), bins=np.arange(int(counts.min()) - 0.5, int(counts.max()) + 1.5))
    axis.set(xlabel="Denoiser calls per sample", ylabel="Samples", title="NFE distribution")
    axis.set_xlim(int(counts.min()) - 0.5, int(counts.max()) + 0.5)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    _save_figure(figure, path)


def save_quality_plot(nfe, distances, metric, path):
    if nfe.shape != distances.shape or nfe.ndim != 1 or metric not in ("mse", "lpips"):
        raise ValueError("Per-sample distances must match NFE counts and name the measured metric.")
    require_finite(distances, "paired distances")
    figure = Figure(figsize=(6, 4), layout="constrained")
    axis = figure.subplots()
    axis.scatter(nfe.detach().cpu().numpy(), distances.detach().cpu().numpy(), s=15, alpha=0.7)
    axis.set(
        xlabel="Denoiser calls",
        ylabel=f"Paired {metric.upper()} distance",
        title="Distance to paired reference",
    )
    axis.set_xlim(float(nfe.min()) - 0.5, float(nfe.max()) + 0.5)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    _save_figure(figure, path)


def save_training_artifacts(rollout, history, directory):
    directory = Path(directory)
    data = rollout.tensor_payload(max_samples=16)
    episode = history[-1]["episode"]
    data.update(episode=episode, phase="pre_update_rollout")
    title = f"Episode {episode + 1}: pre-update rollout"
    save_checkpoint(data, directory / "training_samples.pt")
    save_json({"episodes": history}, directory / "history.json")
    save_training_curves(history, directory / "training_curves.png")
    save_image_grid(data["images"], data["nfe"], directory / "training_samples.png", title)
    save_action_schedule(data["actions"], data["forced"], directory / "training_actions.png", title)
    save_trajectory(data["trajectory"], directory / "training_trajectory.png", title)
    comparison = directory / "training_comparison.png"
    if data["reference_images"] is not None:
        save_comparison(data["images"], data["reference_images"], data["nfe"], comparison, title)
    else:
        comparison.unlink(missing_ok=True)
