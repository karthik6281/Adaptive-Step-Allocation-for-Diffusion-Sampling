"""Conditional bootstrap for the per-image LPIPS advantage."""

import numpy as np

from .config import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED


def interval(shared, per_image, step_value, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS):
    """95% interval for the mean LPIPS gap, in evaluation-equivalent units.

    Conditional: the searched schedules and the N+1 denominator are held fixed,
    so this measures spread across the evaluation images, not search variability
    and not out-of-sample generalisation. The two arrays are resampled together
    so the per-image pairing survives.

    Endpoints round outward. Across seeds the raw endpoints move by ~2e-3, enough
    to flip a second decimal, so an inward-rounded interval would not reproduce.
    """
    if shared.shape != per_image.shape:
        raise ValueError("shared and per_image must be aligned per image")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), size=(draws, len(shared)))
    ratios = (shared[idx].mean(1) - per_image[idx].mean(1)) / step_value
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    return float(np.floor(lo * 100) / 100), float(np.ceil(hi * 100) / 100)
