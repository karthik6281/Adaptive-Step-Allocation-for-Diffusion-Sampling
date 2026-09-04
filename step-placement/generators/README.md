# Stage 1 generators

These four scripts produced the arrays in `../data/`. They are not needed for
the CPU reproduction check; the generated arrays ship with the repository.

They are included to document the measurement stage.

## Requirements

Beyond the base install:

```bash
pip install -e '.[generate]'      # run from the project root
```

and, separately, the five released 256x256 latent diffusion checkpoints
(LSUN-church, LSUN-bedroom, CelebA-HQ, FFHQ, class-conditional ImageNet) plus
their configs. These are not redistributed here. Each script takes the
checkpoint and config paths as arguments; run with `--help` for the exact flags.

The scripts also import `adapter_compvis.py` and utilities from the parent TAED
research repository. They are therefore provenance code in a standalone copy
of this artifact, not a self-contained GPU pipeline.

A CUDA device is assumed. Runtimes are hours per model, not minutes.

## What each produces

| script | produces | feeds |
| --- | --- | --- |
| `curvature_probe.py` | `data/profiles/curv_*.npz` | the cost and shape tables, both figures |
| `search_vs_sample.py` | per-image LPIPS search outputs in `data/runs/` | the LPIPS advantage table |
| `solver_probe.py` | `data/runs/dpm_*.npz` | the second-order solver proxy comparison |
| `direction_probe.py` | stored second-difference directions | the aggregation and rotation analysis |

## Reproducibility notes

- The measurement grid is `round(linspace(999, 0, steps + 1))`, which is not
  uniform. `steps/grid.py` documents the consequence and excises the affected
  columns. Do not "fix" the grid here without regenerating every array.
- Trajectories are deterministic DDIM with eta = 0. Stochastic sampling is out
  of scope; comparing it soundly would need common random numbers.
- Second-difference directions are stored in float16 to keep the arrays small.
  The reported rotation statistics sit orders of magnitude above that rounding,
  which the analysis checks rather than assumes.
