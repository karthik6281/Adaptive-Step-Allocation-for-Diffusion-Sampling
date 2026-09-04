"""The fixed-budget placement objective and its exact optimiser.

For a solver of order p the leading local error on step i is c_k(t_i) h_i^(p+1),
and the surrogate sums those over a mesh M = {t_0, ..., t_B}:

    E_k(M) = sum_i c_k(t_i) h_i^(p+1),   h_i = |t_{i+1} - t_i|.

DDIM is first order, so the exponent is 2. E is linear in the profile, so
E_{ac} = a E_c for a > 0 and positive scaling cannot change which mesh wins:
only the shape of a profile over time matters.

The objective sums local error magnitudes. It does not model propagation or
cancellation through later steps, and its value is neither an image-quality
metric nor the exact global integration error.
"""

import numpy as np

from .config import SURROGATE_EXPONENT


def normalise(profiles):
    """Divide each profile by its own mass. Leaves every sample's optimum alone."""
    return profiles / profiles.sum(axis=1, keepdims=True)


def energy(profile, nodes, mesh):
    return float((profile[mesh[:-1]] * np.diff(nodes[mesh]) ** SURROGATE_EXPONENT).sum())


def best_mesh(nodes, profile, budget):
    """Exact argmin over meshes with `budget` intervals, by dynamic programming.

    Endpoints are pinned to the first and last retained node so shared and
    per-sample optima are compared on the same feasible set. Exact rather than
    the equidistribution formula, which is a many-step approximation and can
    misprice a mesh badly at these budgets.
    """
    n = len(nodes)
    spans = nodes[None, :] - nodes[:, None]
    cost = np.where(spans > 0, profile[:, None] * spans**SURROGATE_EXPONENT, np.inf)

    best = np.full((budget + 1, n), np.inf)
    prev = np.zeros((budget + 1, n), dtype=int)
    best[0, 0] = 0.0
    for k in range(1, budget + 1):
        cand = best[k - 1][:, None] + cost
        prev[k] = np.argmin(cand, axis=0)
        best[k] = cand[prev[k], np.arange(n)]

    mesh, j = [n - 1], n - 1
    for k in range(budget, 0, -1):
        j = prev[k, j]
        mesh.append(j)
    return np.array(mesh[::-1])


def excess(profiles, nodes, budget):
    """Percent by which the shared mesh exceeds each sample's own optimum.

    With equal sample weights the shared mesh is the optimum for the mean profile.
    """
    shared = best_mesh(nodes, profiles.mean(axis=0), budget)
    return (
        np.array(
            [
                energy(c, nodes, shared) / energy(c, nodes, best_mesh(nodes, c, budget))
                - 1.0
                for c in profiles
            ]
        )
        * 100
    )


def descend(profile, nodes, budget, start):
    """Cyclic coordinate descent over every legal interior position.

    Used to calibrate search error against the exact optimum. The surrogate has
    no interaction beyond adjacent nodes, so this flatters a coordinate method
    rather than estimating its error on a full rollout.
    """
    mesh = np.array(start)
    while True:
        moved = False
        for k in range(1, budget):
            lo, hi = mesh[k - 1] + 1, mesh[k + 1] - 1
            if lo > hi:
                continue
            best_val, best_pos = energy(profile, nodes, mesh), mesh[k]
            for pos in range(lo, hi + 1):
                trial = mesh.copy()
                trial[k] = pos
                val = energy(profile, nodes, trial)
                if val < best_val - 1e-15:
                    best_val, best_pos = val, pos
            if best_pos != mesh[k]:
                mesh[k] = best_pos
                moved = True
        if not moved:
            return mesh
