"""Which grid columns actually measure curvature.

The grid is round(linspace(999, 0, n+1)), which is not uniform: at 200 steps one
step has size 4 and the rest 5. Writing h_j for the step ending at j,

    x_j - 2x_{j-1} + x_{j-2} = (h_j - h_{j-1}) x' + (h_j^2 + h_{j-1}^2) x''/2 + ...

so the first-derivative term vanishes only where adjacent steps match. At the
short step it survives, and since ||x'|| >> h||x''|| it swamps the curvature.
Columns whose stencil straddles such a step are dropped.
"""

import numpy as np

from .config import MAX_T, STENCIL_WIDTH, TERMINAL_TRIM


def step_sizes(steps):
    nodes = np.linspace(MAX_T - 1, 0, steps + 1).round().astype(np.int64)
    return -np.diff(nodes)


def irregular_steps(steps):
    """Steps whose size differs from the modal one."""
    sizes = step_sizes(steps)
    return np.where(sizes != np.bincount(sizes).argmax())[0]


def retained_columns(steps, n_columns, width=STENCIL_WIDTH, trim=TERMINAL_TRIM):
    bad = {
        col
        for step in irregular_steps(steps)
        for col in range(step - width + 1, step + 1)
        if 0 <= col < n_columns
    }
    return [j for j in range(n_columns - trim) if j not in bad]
