"""Loading the shipped arrays."""

from pathlib import Path
from typing import NamedTuple

import numpy as np

from .config import MEASUREMENT_STEPS, PROFILE_EXPONENT, SOLVER_KEYS
from .grid import retained_columns

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "data" / "profiles"
RUNS = ROOT / "data" / "runs"


class MissingData(FileNotFoundError):
    pass


def _path(path):
    if not path.exists():
        raise MissingData(
            f"{path} not found. The arrays ship with the repository; rebuilding "
            f"one needs a GPU and the matching checkpoint (see generators/)."
        )
    return path


def raw_profile(dataset):
    with np.load(_path(PROFILES / f"curv_{dataset}.npz")) as z:
        return z["curv"]


def profile(dataset):
    """Retained profiles and their node positions.

    Node positions are indices in the untrimmed grid, so a step across the
    excised gap is charged its full three-index length. Excision removes
    candidate nodes; it does not contract the domain.
    """
    raw = raw_profile(dataset)
    keep = retained_columns(MEASUREMENT_STEPS, raw.shape[1])
    return raw[:, keep], np.asarray(keep, dtype=float)


class Search(NamedTuple):
    shared: np.ndarray  # per-image LPIPS under the shared schedule
    per_image: np.ndarray
    gain: float
    step_value: float  # improvement from a separately searched N+1 schedule
    improved: int
    distinct: int


def search(dataset, budget):
    """The 256-image, ten-sweep search for one model and budget."""
    stem = RUNS / f"search_{dataset}_B{budget}_n256_sw10"
    with np.load(_path(stem.with_suffix(".npz"))) as z:
        arrays = {k: z[k] for k in z.files}

    count_fields = ("n_improved", "uniq_schedules")
    missing = [field for field in count_fields if field not in arrays]
    if missing:
        raise ValueError(f"{stem.name}.npz is missing metadata: {', '.join(missing)}")

    return Search(
        np.asarray(arrays["D_fixed"], float),
        np.asarray(arrays["per_img"], float),
        float(arrays["gain"]),
        float(arrays["step_value"]),
        *(int(arrays[field]) for field in count_fields),
    )


def solver(dataset):
    """Measured solver error and the two difference proxies, on a common index set.

    The measured error is divided by its expected h^p factor so it is comparable
    with the proxies column by column.
    """
    key = SOLVER_KEYS[dataset]
    with np.load(_path(RUNS / f"curv_{key}_100.npz")) as z:
        second, third = z["curv"], z["curv3"]
    with np.load(_path(RUNS / f"dpm_{key}.npz")) as z:
        raw, hlam = z["curv_raw"], z["hlam"]

    n = min(second.shape[1], third.shape[1], raw.shape[1])
    measured = raw[:, :n] / hlam[None, :n] ** PROFILE_EXPONENT
    return measured, second[:, :n], third[:, :n]
