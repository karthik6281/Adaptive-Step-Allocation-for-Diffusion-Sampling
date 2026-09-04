"""The four result tables, recomputed from the shipped arrays."""

from typing import NamedTuple

import numpy as np

from . import boot, load, shape
from .config import (
    BUDGETS,
    CALIBRATION_SAMPLE,
    CALIBRATION_SEED,
    DATASETS,
    SOLVER_STENCIL_WIDTH,
    SOLVER_STEPS,
    SOLVER_TRIM,
)
from .grid import retained_columns
from .mesh import best_mesh, descend, energy, excess, normalise


class Cost(NamedTuple):
    dataset: str
    budget: int
    median: float
    p90: float
    p99: float
    worst: float
    distinct: int
    matching_shared: int


class Shape(NamedTuple):
    dataset: str
    magnitude: float
    mean_rho: float
    tail: float
    minimum: float


class Advantage(NamedTuple):
    dataset: str
    budget: int
    gain: float
    recovered: float
    interval: tuple
    improved: int
    distinct: int


class Proxy(NamedTuple):
    dataset: str
    raw_second: float
    raw_third: float
    excised_second: float
    excised_third: float
    third_wins: int


def cost(dataset, budget):
    profiles, nodes = load.profile(dataset)
    shapes = normalise(profiles)

    ex = excess(shapes, nodes, budget)
    shared = tuple(best_mesh(nodes, shapes.mean(axis=0), budget))
    meshes = [tuple(best_mesh(nodes, s, budget)) for s in shapes]

    return Cost(
        dataset,
        budget,
        float(np.median(ex)),
        float(np.quantile(ex, 0.90)),
        float(np.quantile(ex, 0.99)),
        float(ex.max()),
        len(set(meshes)),
        sum(m == shared for m in meshes),
    )


def agreement(dataset):
    profiles, _ = load.profile(dataset)
    return Shape(dataset, shape.magnitude(profiles), *shape.summary(profiles))


def advantage(dataset, budget):
    s = load.search(dataset, budget)
    return Advantage(
        dataset,
        budget,
        s.gain,
        s.gain / s.step_value,
        boot.interval(s.shared, s.per_image, s.step_value),
        s.improved,
        s.distinct,
    )


def proxy(dataset):
    """Correlate measured solver error with both finite-difference proxies.

    Scored twice on the same trajectories: once boundary-trimmed, once after also
    excising the rounding-contaminated columns. The contrast is the point.
    """
    measured, second, third = load.solver(dataset)
    n = measured.shape[1]
    column_sets = (
        list(range(n - SOLVER_TRIM)),
        retained_columns(SOLVER_STEPS, n, SOLVER_STENCIL_WIDTH, SOLVER_TRIM),
    )

    means, wins = [], 0
    for cols in column_sets:
        target = shape.unit_shapes(measured[:, cols])
        vs_second = (target * shape.unit_shapes(second[:, cols])).sum(axis=1)
        vs_third = (target * shape.unit_shapes(third[:, cols])).sum(axis=1)
        means += [float(vs_second.mean()), float(vs_third.mean())]
        wins = int((vs_third > vs_second).sum())

    return Proxy(dataset, *means, wins)


def calibration(datasets=DATASETS, budgets=BUDGETS):
    """How far coordinate descent lands from the exact optimum.

    Returns (lowest cell mean, highest cell mean, worst single sample), in
    percent. Samples whose shared mesh is already their own optimum are skipped:
    the search starts there and scores zero on them by construction, which would
    only dilute the figure.
    """
    cell_means, worst = [], 0.0
    for dataset in datasets:
        profiles, nodes = load.profile(dataset)
        shapes = normalise(profiles)
        rng = np.random.default_rng(CALIBRATION_SEED)
        drawn = shapes[rng.choice(len(shapes), CALIBRATION_SAMPLE, replace=False)]

        for budget in budgets:
            shared = best_mesh(nodes, shapes.mean(axis=0), budget)
            gaps = []
            for s in drawn:
                opt = best_mesh(nodes, s, budget)
                if energy(s, nodes, shared) - energy(s, nodes, opt) <= 1e-15:
                    continue
                found = descend(s, nodes, budget, shared)
                gaps.append(100 * (energy(s, nodes, found) / energy(s, nodes, opt) - 1))
            cell_means.append(float(np.mean(gaps)))
            worst = max(worst, max(gaps))
    return min(cell_means), max(cell_means), worst


def cost_table(datasets=DATASETS, budgets=BUDGETS):
    return [cost(d, b) for d in datasets for b in budgets]


def shape_table(datasets=DATASETS):
    return [agreement(d) for d in datasets]


def advantage_table(datasets=("church256", "bedroom256"), budgets=BUDGETS):
    return [advantage(d, b) for d in datasets for b in budgets]


def proxy_table(datasets=DATASETS):
    return [proxy(d) for d in datasets]
