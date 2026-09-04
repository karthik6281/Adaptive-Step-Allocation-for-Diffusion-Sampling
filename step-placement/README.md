# Step placement at a fixed budget

Code and data for measuring how much a single shared timestep schedule costs,
relative to giving every sample its own, at a fixed number of denoising
evaluations.

The four result tables and two derived summaries regenerate from the arrays in
`data/`, on CPU, in a few seconds, with no downloads or model checkpoints:

```bash
pip install -e .
python scripts/check.py
```

That recomputes 130 published values — the four result tables plus the
mean-shape share and the coordinate-descent calibration — and exits non-zero if
any of them disagrees with what the paper prints.

## Layout

```
src/steps/
    config.py    constants, and the published values check.py compares against
    grid.py      which grid columns actually measure curvature
    load.py      reading the arrays
    mesh.py      the objective, its exact DP optimiser, coordinate descent
    shape.py     pairwise agreement between profiles
    boot.py      conditional bootstrap for the LPIPS advantage
    results.py   the four tables
    plots.py     pgfplots fragments
scripts/         check.py, tables.py, figures.py
generators/      GPU stage that produced data/ -- optional, see its README
data/            profiles/ (5 arrays) and runs/ (searches, solver diagnostics)
```

`make tables` prints the tables, `make latex` emits LaTeX bodies, `make figures`
writes the two figure fragments.

## Using it directly

```python
import numpy as np
from steps import load, mesh

profiles, nodes = load.profile("church256")   # (256, 188) and node positions
shapes = mesh.normalise(profiles)             # scale cancels; only shape matters

shared = mesh.best_mesh(nodes, shapes.mean(0), budget=8)
ex = mesh.excess(shapes, nodes, budget=8)

print(shared.tolist())
# [0, 74, 94, 112, 127, 143, 157, 173, 187]
print(f"median {np.median(ex):.2f}%  p90 {np.quantile(ex, 0.9):.2f}%")
# median 3.13%  p90 8.74%
```

## Method

The surrogate models the leading local truncation error on step $i$ of an
order-$p$ solver as $c_k(t_i)h_i^{p+1}$ and sums it over a mesh. Because it is
linear in the profile, scaling a profile cannot change which mesh minimises it —
only the temporal shape can. Profiles come from the Euclidean norm of the second
difference of 256 deterministic DDIM trajectories per model on a 200-step grid.
After dropping the terminal boundary spike and the columns contaminated by grid
rounding, 188 candidate nodes remain, spanning timesteps 994 to 50. Shared and
per-sample optima are solved exactly by dynamic programming rather than with the
equidistribution formula, which is a many-step approximation and misprices
meshes badly at these budgets.

## Caveats

- The surrogate sums local error magnitudes. It ignores propagation and
  cancellation through later steps, and is not an image-quality metric.
- The per-image LPIPS searches score against the same reference they optimise
  toward, which a deployable predictor would not have.
- Coordinate descent can miss better schedules, so the reported advantages
  bound neither the global optimum nor what a predictor could recover.
- Converting LPIPS gaps to evaluation-equivalents assumes local linearity.
- All five checkpoints come from one source and share a noise schedule, so the
  high cross-model shape agreement may be partly a property of that schedule.
- Stochastic sampling ($\eta > 0$) and text-conditioned models are untested.

## Data

The 6.4 MB data bundle contains 33 NPZ archives and no run logs or local paths.

- `profiles/curv_<model>.npz` — key `curv`, shape `(256, 199)`: untrimmed
  curvature proxy per sample. `load.profile` trims it to 188 columns.
- `runs/search_<model>_B<n>_n256_sw10.npz` — per-image LPIPS searches, including
  the reported counts and run metadata.
- `runs/curv_<model>_100.npz`, `runs/dpm_<model>.npz` — solver diagnostics.

Nineteen archives feed `check.py`. The remaining archives support appendix
sensitivity checks: 64-image searches, 50/400-step grid measurements, and the
excluded stochastic runs.

Model keys are `church256`, `bedroom256`, `celeb256`, `ffhq256`, `cin256`; the
solver runs use `church` for LSUN-church.

## License

MIT. The diffusion checkpoints measured in the generator stage are not
redistributed and carry their own terms.
