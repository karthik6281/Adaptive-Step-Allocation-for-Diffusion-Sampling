"""How much sample profiles agree in shape.

Reported separately from mesh cost throughout, because the two do not rank
perturbations consistently: a random time shift leaves correlation high yet
costs more than the real profiles, while reweighting the outer thirds lowers
correlation and costs less.
"""

import numpy as np


def unit_shapes(profiles):
    """Centred, unit-norm shapes whose Gram matrix is the Pearson matrix."""
    shares = profiles / profiles.sum(axis=1, keepdims=True)
    centred = shares - shares.mean(axis=1, keepdims=True)
    return centred / (np.linalg.norm(centred, axis=1, keepdims=True) + 1e-30)


def correlations(profiles):
    g = unit_shapes(profiles)
    gram = g @ g.T
    return gram[np.triu_indices(len(gram), 1)]


def summary(profiles):
    """Mean, 1st percentile and minimum of the pairwise correlations."""
    off = correlations(profiles)
    return float(off.mean()), float(np.quantile(off, 0.01)), float(off.min())


def magnitude(profiles):
    """Largest retained profile mass over the smallest, before normalising."""
    mass = profiles.sum(axis=1)
    return float(mass.max() / mass.min())


def mean_shape_share(profiles):
    """Fraction of total squared norm carried by the mean centred shape."""
    shares = profiles / profiles.sum(axis=1, keepdims=True)
    centred = shares - shares.mean(axis=1, keepdims=True)
    return float(len(centred) * (centred.mean(axis=0) ** 2).sum() / (centred**2).sum())
