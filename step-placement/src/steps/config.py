"""Protocol constants and the published values we check against."""

MAX_T = 1000
MEASUREMENT_STEPS = 200

# The last DDIM update lands on t=0 and spikes the second difference in every
# sample at the same place, so those columns say nothing about where a sample
# curves. Averaged per model they hold 36-63% of the full-grid mass.
TERMINAL_TRIM = 9

# A second difference spans two steps; three for a third difference.
STENCIL_WIDTH = 2

# DDIM is first order, so local error is O(h^2). The solver diagnostics are
# second order, hence h^3.
SURROGATE_EXPONENT = 2
PROFILE_EXPONENT = 3.0

SOLVER_STEPS = 100
SOLVER_TRIM = 10
SOLVER_STENCIL_WIDTH = 3

BUDGETS = (4, 8)
DATASETS = ("church256", "bedroom256", "celeb256", "ffhq256", "cin256")

LABELS = {
    "church256": "LSUN-church",
    "bedroom256": "LSUN-bedroom",
    "celeb256": "CelebA-HQ",
    "ffhq256": "FFHQ",
    "cin256": "ImageNet",
}

# The solver runs were written under a shorter stem for church.
SOLVER_KEYS = dict(zip(DATASETS, ("church",) + DATASETS[1:]))

BOOTSTRAP_SEED = 0
BOOTSTRAP_DRAWS = 200_000

CALIBRATION_SAMPLE = 32
CALIBRATION_SEED = 0


# --- published values, recomputed by scripts/check.py ------------------------

# median, p90, p99, max, distinct meshes, meshes equal to the shared one
COST = {
    ("church256", 4): (2.11, 8.12, 26.22, 28.30, 66, 15),
    ("church256", 8): (3.13, 8.74, 19.12, 24.45, 165, 10),
    ("bedroom256", 4): (0.21, 7.83, 36.73, 72.35, 14, 38),
    ("bedroom256", 8): (0.32, 5.96, 30.50, 60.85, 47, 14),
    ("celeb256", 4): (0.44, 2.77, 11.43, 21.09, 32, 20),
    ("celeb256", 8): (1.30, 5.82, 11.87, 19.33, 118, 19),
    ("ffhq256", 4): (0.02, 0.25, 0.70, 13.77, 12, 105),
    ("ffhq256", 8): (0.01, 0.37, 1.58, 14.11, 33, 122),
    ("cin256", 4): (0.00, 0.45, 18.57, 94.47, 17, 206),
    ("cin256", 8): (0.34, 1.63, 15.41, 83.52, 37, 41),
}

# magnitude, mean rho, 1% tail, min
SHAPE = {
    "church256": (1.70, 0.977, 0.910, 0.820),
    "bedroom256": (1.56, 0.975, 0.926, 0.882),
    "celeb256": (1.58, 0.972, 0.905, 0.828),
    "ffhq256": (1.56, 0.986, 0.951, 0.903),
    "cin256": (2.11, 0.993, 0.952, 0.907),
}

# raw LPIPS gap, recovered evaluations, interval, improved, distinct schedules
ADVANTAGE = {
    ("church256", 4): (0.0218, 0.255, (0.22, 0.29), 238, 176),
    ("church256", 8): (0.0199, 0.713, (0.60, 0.83), 255, 255),
    ("bedroom256", 4): (0.0173, 0.247, (0.20, 0.30), 205, 124),
    ("bedroom256", 8): (0.0196, 0.691, (0.61, 0.78), 252, 246),
}

# raw vs 2nd, raw vs 3rd, excised vs 2nd, excised vs 3rd, trajectories 3rd wins
PROXY = {
    "church256": (0.481, 0.160, 0.558, 0.647, 60),
    "bedroom256": (0.629, 0.241, 0.700, 0.825, 63),
    "celeb256": (0.594, 0.162, 0.728, 0.688, 12),
    "ffhq256": (0.524, 0.112, 0.673, 0.753, 63),
    "cin256": (0.711, 0.257, 0.750, 0.764, 47),
}

MEAN_SHAPE_SHARE = (96.4, 99.0)

# lowest cell mean, highest cell mean, worst single sample
CALIBRATION = (0.04, 1.56, 6.21)
