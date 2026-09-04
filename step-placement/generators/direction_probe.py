"""Measure directions of second differences along DDIM trajectories.

The saved norms are checked against a matching curvature archive. Analysis
reports cross-sample directional agreement, within-sample rotation profiles,
and lagged cosine similarity. Directions are stored in float16; norms are stored
in float64.
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_TAED = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _TAED, os.path.join(_TAED, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from curvature_probe import _stub_clip_if_missing  # noqa: E402
from verify_sep2 import grid_cols  # noqa: E402
from verify_sep2 import norm as _norm  # noqa: E402

# Only to fail fast before a checkpoint load; the adapter remains the source of truth.
_KNOWN_CONDITIONAL = {"cin256"}


def measure(args):
    import torch
    if args.dataset in _KNOWN_CONDITIONAL and args.classes <= 0:
        raise SystemExit(
            f"{args.dataset} is class-conditional: pass --classes 1000 before loading a "
            f"checkpoint, or every image would be generated from class 0.")
    if _stub_clip_if_missing():
        print("  note: `clip` not installed; stubbed for the import only")
    from adapter_compvis import create_compvis_adapter
    from reward_structure import resolve_paths

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    cfg_path, ckpt_path = resolve_paths(args.dataset, args.config, args.ckpt)
    adapter = create_compvis_adapter(cfg_path, ckpt_path, device=device,
                                     n_timesteps=args.steps).to(device).eval()

    is_cond = getattr(getattr(adapter, "model", None), "cond_stage_model", None) is not None
    if is_cond and args.classes <= 0:
        raise SystemExit(
            f"{args.dataset} is CLASS-CONDITIONAL. Without --classes every image would be "
            f"generated from class 0 and directional agreement would be overstated. "
            f"Pass --classes 1000 for ImageNet.")
    if not is_cond and args.classes > 0:
        raise SystemExit(f"{args.dataset} is unconditional; --classes would do nothing.")

    C = getattr(adapter, "latent_channels", 4)
    S = getattr(adapter, "latent_size", 32)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    z_T = torch.randn(args.n_images, C, S, S, generator=g).to(device)
    labels = None
    if is_cond:
        rng = np.random.default_rng(args.seed)
        labels = torch.from_numpy(rng.integers(0, args.classes, args.n_images)).long().to(device)
        print(f"  conditional: {len(torch.unique(labels))} distinct classes over "
              f"{args.n_images} images")

    sched = np.linspace(args.max_t - 1, 0, args.steps + 1).round().astype(np.int64)
    dim = C * S * S
    cols = args.steps - 1
    gib = args.n_images * cols * dim * 2 / 1024 ** 3
    print(f"  {args.dataset}  n={args.n_images}  {args.steps} steps  latent dim {dim}  "
          f"directions {gib:.2f} GiB (fp16)")

    dirs = np.zeros((args.n_images, cols, dim), dtype=np.float16)
    norms = np.zeros((args.n_images, cols), dtype=np.float64)

    start = time.time()
    done = 0
    for b0 in range(0, args.n_images, args.batch):
        z = z_T[b0:b0 + args.batch]
        lab = None if labels is None else labels[b0:b0 + args.batch]
        prev2, prev1 = None, z
        t_cur = torch.full((z.shape[0],), int(sched[0]), device=device, dtype=torch.long)
        for j in range(1, len(sched)):
            t_next = torch.full_like(t_cur, int(sched[j]))
            out = adapter._denoise_step_latents(z, t_cur, t_next, eta=0.0, class_labels=lab)
            z = out[0] if isinstance(out, (tuple, list)) else out
            t_cur = t_next
            if prev2 is not None:
                d2 = (z - 2.0 * prev1 + prev2).flatten(1)
                n2 = d2.norm(dim=1)
                norms[b0:b0 + z.shape[0], j - 2] = n2.double().cpu().numpy()
                unit = d2 / n2.clamp_min(1e-30).unsqueeze(1)
                dirs[b0:b0 + z.shape[0], j - 2] = unit.half().cpu().numpy()
            prev2, prev1 = prev1, z
        done += z.shape[0]
        elapsed = time.time() - start
        eta = elapsed / done * (args.n_images - done)
        print(f"    images {b0}-{b0 + z.shape[0] - 1} done  "
              f"[{elapsed / 60:.1f} min elapsed, ETA {eta / 60:.1f} min]", flush=True)

    if not np.isfinite(norms).all() or norms.sum(axis=1).min() <= 0:
        raise SystemExit("non-finite or all-zero curvature norms")
    np.savez(args.out, dirs=dirs, norms=norms, sched=sched, dataset=args.dataset,
             classes=(labels.cpu().numpy() if labels is not None else np.array([])))
    print(f"  saved {args.out}  dirs{dirs.shape}  norms{norms.shape}")


def _pair_stats(unit):
    """Off-diagonal cosine similarities among unit rows."""
    gram = unit @ unit.T
    off = gram[~np.eye(len(gram), dtype=bool)]
    return float(off.mean()), float(np.quantile(off, 0.01)), float(off.min())


def _unit(block):
    block = block.astype(np.float32)
    return block / (np.linalg.norm(block, axis=1, keepdims=True) + 1e-30)


def analyze(args):
    z = np.load(args.analyze)
    dirs, norms = z["dirs"], z["norms"]
    dataset = str(z["dataset"])

    # The grid comes from the file, not from the --steps default, which would silently
    # mis-trim anything not measured at 200.
    steps = len(z["sched"]) - 1
    if args.steps is not None and args.steps != steps:
        raise SystemExit(f"--steps {args.steps} contradicts the file's grid ({steps})")

    ref = np.load(args.curv)["curv"]
    n = norms.shape[0]
    if norms.shape[1] != ref.shape[1] or n > ref.shape[0]:
        raise SystemExit(f"incompatible shapes: curv{ref.shape} vs norms{norms.shape}")
    if n < ref.shape[0]:
        # z_T and the labels are drawn from one fixed generator, so the first n of an
        # n-image run are the first n of the 256-image run provided --batch is unchanged.
        print(f"  PILOT SUBSET: first {n} of {ref.shape[0]} samples "
              f"({n * (n - 1)} ordered pairs); rerun at full n before quoting numbers")
        ref = ref[:n]
    rel = np.abs(norms - ref) / np.maximum(np.abs(ref), 1e-30)
    print(f"  reproduction check vs {os.path.basename(args.curv)}: "
          f"max rel err {rel.max():.3e}")
    if rel.max() > args.tol:
        raise SystemExit(
            f"norms do not reproduce curv (max rel err {rel.max():.3e} > {args.tol:g}). "
            f"The trajectories differ, so these directions do not belong to the published "
            f"profiles. Re-run on the machine that produced {os.path.basename(args.curv)}, "
            f"or raise --tol only if you can justify the drift.")

    keep = grid_cols(steps, norms.shape[1])
    dim = dirs.shape[2]
    # The shared mesh is fitted to the mean of NORMALISED profiles, so weight by that same
    # object rather than by the raw mean, which high-mass samples would dominate.
    retained = ref[:, keep]
    cbar = (retained / retained.sum(1, keepdims=True)).mean(axis=0)
    top = set(np.argsort(cbar)[::-1][:args.top].tolist())

    rows_all, rows_top = [], []
    for idx, col in enumerate(keep):
        stats = _pair_stats(_unit(dirs[:, col, :]))
        rows_all.append(stats)
        if idx in top:
            rows_top.append(stats)
    rows_all, rows_top = np.array(rows_all), np.array(rows_top)
    weighted_mean = float((rows_all[:, 0] * cbar).sum() / cbar.sum())

    # In 4k-12k dimensions independent unit vectors are already near-orthogonal, so the
    # observed cosines mean nothing without this reference.
    rng = np.random.default_rng(0)
    null_mean, null_tail, _ = _pair_stats(_unit(rng.normal(size=(n, dim))))
    null_sd = 1.0 / np.sqrt(dim)

    # Per-sample rotation profile. Only consecutive retained points whose grid indices are
    # genuinely adjacent: one pair straddles the two-point excision gap and is really lag 3.
    adjacent = [i for i in range(len(keep) - 1) if keep[i + 1] - keep[i] == 1]
    rot = np.zeros((n, len(adjacent)), dtype=np.float64)
    for slot, i in enumerate(adjacent):
        a, b = _unit(dirs[:, keep[i], :]), _unit(dirs[:, keep[i + 1], :])
        rot[:, slot] = 1.0 - (a * b).sum(axis=1)

    rot_unit = _norm(rot)
    rot_mean, rot_tail, _ = _pair_stats(rot_unit)
    curv_mean, curv_tail, _ = _pair_stats(_norm(retained))
    per_sample = rot.mean(axis=1)
    # A near-constant r_k would leave only residual after row-normalising, and its
    # correlations would then measure noise rather than disagreement.
    cov = rot.std(axis=1) / np.maximum(rot.mean(axis=1), 1e-30)
    # fp16 stores ~11 mantissa bits per component; the d products average, so the floor on
    # a dot product is ~2^-11/sqrt(d), not 2^-11.
    fp16_floor = 7e-4 / np.sqrt(dim)

    lags = {}
    for lag in (1, 10, 50):
        clean = [i for i in range(len(keep) - lag) if keep[i + lag] - keep[i] == lag]
        vals = [float((_unit(dirs[:, keep[i], :])
                       * _unit(dirs[:, keep[i + lag], :])).sum(axis=1).mean())
                for i in clean]
        lags[lag] = (float(np.mean(vals)) if vals else float("nan"), len(clean))

    print(f"\n  {dataset}: second-difference DIRECTIONS   (dim {dim}, {len(keep)} points)")
    print("    ROTATION PROFILE r_k[j] = 1 - cos(d2[j], d2[j+1])   <- the load-bearing one")
    print(f"      pairwise corr of normalised r_k : mean {rot_mean:+.4f}  "
          f"1% tail {rot_tail:+.4f}   over {len(adjacent)} steps")
    print(f"      same statistic for the profile  : mean {curv_mean:+.4f}  "
          f"1% tail {curv_tail:+.4f}   <- paper's c_k, for comparison")
    print(f"      mean rotation per step          : {per_sample.mean():.4f}  "
          f"(across samples: min {per_sample.min():.4f}, max {per_sample.max():.4f})")
    print(f"      CoV of r_k across j (median)    : {np.median(cov):.2f}   "
          f"<- near 0 would make the correlations noise")
    print(f"      smallest r_k vs fp16 floor      : {rot.min():.1e} vs {fp16_floor:.1e}"
          f"  ({rot.min() / fp16_floor:,.0f}x headroom)")
    print("    ACROSS SAMPLES at matched timesteps   (context: direction has no slot in Eq. 1)")
    print(f"      unweighted / c-weighted / top-{args.top} : {rows_all[:, 0].mean():+.4f}  "
          f"{weighted_mean:+.4f}  {rows_top[:, 0].mean():+.4f}")
    print(f"      NULL, random unit vectors       : {null_mean:+.4f}  "
          f"(single-pair sd {null_sd:.4f})")
    print("    WITHIN A SAMPLE, across timesteps")
    print("      " + "   ".join(f"lag {k}: {v:+.4f} ({c} win)" for k, (v, c) in lags.items()))
    print("\n    If the rotation correlations match the profile's, samples agree about WHERE")
    print("    the curvature swings, so a rotation-aware objective would weight steps almost")
    print("    identically across them. If they are much weaker, that is a mechanism by which")
    print("    samples could prefer different placements and Eq. (1) cannot express it.")
    print("    Read the across-sample rows against the NULL, not zero; they bound how much")
    print("    the norm removes, not how much placement-relevant information it removes.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--out")
    p.add_argument("--analyze")
    p.add_argument("--curv", help="curv_*.npz whose norms these directions must reproduce")
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=None,
                   help="measurement grid; for --analyze it is read from the file")
    p.add_argument("--max-t", dest="max_t", type=int, default=1000)
    p.add_argument("--n-images", dest="n_images", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classes", type=int, default=0)
    p.add_argument("--top", type=int, default=20, help="highest-curvature points to isolate")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="max rel err vs curv norms; 1e-6 held exactly on a same-machine "
                        "repeat, but GPU nondeterminism can need ~1e-5")
    args = p.parse_args()

    if args.analyze:
        if not args.curv:
            raise SystemExit("--analyze requires --curv, the profiles these must reproduce")
        analyze(args)
    elif args.dataset and args.out:
        args.steps = args.steps or 200
        measure(args)
    else:
        p.error("give either --dataset/--out to measure, or --analyze/--curv to report")


if __name__ == "__main__":
    main()
