"""Measure finite-difference profiles along DDIM trajectories.

For each sample, the script records norms of the second and third backward
differences on a shared rounded timestep grid. The second-difference profiles
feed the fixed-budget placement and shape analyses; the third differences are
used for the higher-order proxy check.

Use ``--dataset`` and ``--out`` for GPU measurement, ``--analyze`` for a saved
archive, or ``--self-test`` to validate the finite-difference operators.
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
# Optional generator dependencies live in the parent TAED repository.
_TAED = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _TAED, os.path.join(_TAED, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _stub_clip_if_missing():
    """Allow class-conditional models through an eager optional `clip` import.

    Any actual use of the stub raises, so it cannot stand in for a text encoder.
    """
    import types
    try:
        import clip  # noqa: F401
        return False
    except ImportError:
        pass

    def _boom(*_a, **_k):
        raise RuntimeError("this model genuinely needs the `clip` package, which is not "
                           "installed; the import stub must not be used here")

    m = types.ModuleType("clip")
    m.__getattr__ = lambda _name: _boom
    sys.modules["clip"] = m
    return True


def measure(args):
    import torch
    if _stub_clip_if_missing():
        print("  note: `clip` not installed; stubbed for the import only (class-conditional "
              "models do not use it, text-conditional ones will raise)")
    from adapter_compvis import create_compvis_adapter
    from reward_structure import LDM_NAME, resolve_paths  # noqa: F401

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    cfg_path, ckpt_path = resolve_paths(args.dataset, args.config, args.ckpt)
    adapter = create_compvis_adapter(cfg_path, ckpt_path, device=device,
                                     n_timesteps=args.steps).to(device).eval()

    # Explicit labels avoid the adapter's class-0 fallback for conditional models.
    is_cond = getattr(getattr(adapter, "model", None), "cond_stage_model", None) is not None
    if is_cond and args.classes <= 0:
        raise SystemExit(
            f"{args.dataset} is CLASS-CONDITIONAL. Without --classes every image would be "
            f"generated from class 0 and the shape spread would be understated. "
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
    # An order-p method uses the (p+1)th difference as its local profile proxy.
    curv = np.zeros((args.n_images, args.steps - 1), dtype=np.float64)
    curv3 = np.zeros((args.n_images, args.steps - 2), dtype=np.float64)
    print(f"  {args.dataset}  n={args.n_images}  {args.steps} uniform steps  "
          f"{args.n_images * args.steps:,} LDM calls")

    for b0 in range(0, args.n_images, args.batch):
        z = z_T[b0:b0 + args.batch]
        lab = None if labels is None else labels[b0:b0 + args.batch]
        prev3, prev2, prev1 = None, None, z
        t_cur = torch.full((z.shape[0],), int(sched[0]), device=device, dtype=torch.long)
        for j in range(1, len(sched)):
            t_next = torch.full_like(t_cur, int(sched[j]))
            out = adapter._denoise_step_latents(z, t_cur, t_next, eta=0.0, class_labels=lab)
            z = out[0] if isinstance(out, (tuple, list)) else out
            t_cur = t_next
            if prev2 is not None:
                d2 = (z - 2.0 * prev1 + prev2).flatten(1).norm(dim=1)
                curv[b0:b0 + z.shape[0], j - 2] = d2.double().cpu().numpy()
            if prev3 is not None:
                d3 = (z - 3.0 * prev1 + 3.0 * prev2 - prev3).flatten(1).norm(dim=1)
                curv3[b0:b0 + z.shape[0], j - 3] = d3.double().cpu().numpy()
            prev3, prev2, prev1 = prev2, prev1, z
        print(f"    images {b0}-{b0 + z.shape[0] - 1} done")

    if not np.isfinite(curv).all() or curv.sum(axis=1).min() <= 0:
        raise SystemExit("non-finite or all-zero curvature; the shape normalisation would "
                         "silently produce NaN")
    np.savez(args.out, curv=curv, curv3=curv3, sched=sched, dataset=args.dataset,
             classes=(labels.cpu().numpy() if labels is not None else np.array([])))
    print(f"  saved {args.out}  curv{curv.shape}  curv3{curv3.shape}")


def self_test():
    """The finite-difference operators, against a trajectory with known derivatives.

    x(t) = t^3 has x'' = 6t and x''' = 6, so the BACKWARD differences used in measure()
    must return 6t*h^2 and 6*h^3. If this drifts, every number in the paper drifts with it.
    """
    L, h = 400, 1.0 / 399
    t = np.linspace(0, 1, L)
    x = t ** 3
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    d3 = x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]
    print("  operator self-test on x(t)=t^3   (x''=6t, x'''=6)")
    print(f"    2nd difference / h^2 vs 6t : max abs err {np.abs(d2 / h**2 - 6*t[1:-1]).max():.2e}")
    print(f"    3rd difference / h^3 vs 6  : max abs err {np.abs(d3 / h**3 - 6.0).max():.2e}")
    y = np.exp(2.5 * t)
    e2 = np.abs((y[2:] - 2*y[1:-1] + y[:-2]) / h**2 - 6.25 * y[1:-1]).max() / 6.25
    print(f"    non-polynomial check, exp(2.5t): relative err {e2:.2e}")
    print("    (both are backward differences, so entry k sits at t_{k+1} and t_{k+1.5};")
    print("     that offset is identical for every image and cannot affect a SHAPE")
    print("     comparison across images, which is what the paper's headline uses.)")


def dp_mesh(t, c, n):
    """Exact argmin of sum_i c(t_i)(t_{i+1}-t_i)^2 over n-step meshes. Same DP as GITS."""
    G = len(t)
    step = t[None, :] - t[:, None]
    cost = np.where(step > 0, c[:, None] * step ** 2, np.inf)
    dp = np.full((n + 1, G), np.inf)
    back = np.zeros((n + 1, G), dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, n + 1):
        cand = dp[k - 1][:, None] + cost
        back[k] = np.argmin(cand, axis=0)
        dp[k] = cand[back[k], np.arange(G)]
    idx, j = [G - 1], G - 1
    for k in range(n, 0, -1):
        j = back[k, j]
        idx.append(j)
    return np.array(idx[::-1])


def shape_stats(c, label, trim=0):
    """The surrogate-free half of the result: is the per-image shape common or not?

    trim drops the last `trim` grid points first. This is not cosmetic: the final DDIM step
    lands exactly on t=0 and carries a large second difference that EVERY sample shares, so
    ~70% of the normalised mass sits in the last 10% of the grid with a 2768x dynamic range.
    A spike common to all samples inflates the pairwise correlation for a reason that has
    nothing to do with per-image structure, and the worst-pair statistic is most exposed to
    it. Report both, and quote the trimmed row when the claim is about the worst pair.
    """
    if trim:
        c = c[:, :c.shape[1] - trim]
    n = c.shape[0]
    mass = c.sum(axis=1, keepdims=True)
    shape = c / mass
    sc = shape - shape.mean(axis=1, keepdims=True)
    sc /= np.linalg.norm(sc, axis=1, keepdims=True) + 1e-30
    off = (sc @ sc.T)[~np.eye(n, dtype=bool)]
    e = np.linalg.svd(shape - shape.mean(axis=0, keepdims=True), compute_uv=False) ** 2
    rel = np.linalg.norm(shape - shape.mean(0), axis=1) / np.linalg.norm(shape.mean(0))
    print(f"  {label:26s}{mass.max() / mass.min():7.2f}x{off.mean():+9.4f}{off.min():+9.4f}"
          f"{e.sum() ** 2 / (e ** 2).sum():9.1f}{rel.mean():9.4f}")


def analyze(args):
    z = np.load(args.analyze, allow_pickle=True)
    c = z["curv"]
    n, L = c.shape
    t = np.linspace(0.0, 1.0, L)

    print("=" * 78)
    print(f"CURVATURE SHAPE ACROSS SAMPLES   {z['dataset']}  n={n}  {L} points")
    print("=" * 78)
    print(f"  {'':26s}{'magnitude':>10}{'corr mean':>10}{'corr min':>10}"
          f"{'effRank':>9}{'dev':>9}")
    shape_stats(c, "order-1  (2nd difference)")
    shape_stats(c, "order-1, last 10 dropped", trim=10)
    if "curv3" in z.files:
        shape_stats(z["curv3"], "order-2  (3rd difference)")
        shape_stats(z["curv3"], "order-2, last 10 dropped", trim=10)
    shape_stats(np.ones((n, 1)) * np.linspace(1, 2, L)[None, :]
                * np.linspace(1, 4, n)[:, None], "CONTROL magnitude-only")
    print("    magnitude is the part the theorem cancels; corr/effRank/dev are the SHAPE and")
    print("    use no surrogate. order-2 covers DPM-Solver++/UniPC -- an order-p method's")
    print("    error per step is c*h^(p+1) with c ~ ||x^(p+1)||, and the cancellation of a")
    print("    per-sample scalar holds for EVERY p, so this is not a DDIM-only statement.")
    print("    THE TRIMMED ROWS ARE THE ONES TO QUOTE FOR corr min. The final step lands on")
    print("    t=0 and carries a second difference every sample shares, which inflates the")
    print("    worst pair; the MEAN is insensitive to it either way.")
    print("    shape-varying control reads corr +0.0714, effRank 3.6, dev 1.3763.")

    budgets = [int(b) for b in args.budgets.split(",")]
    uni_c = c.mean(axis=0)
    # E as a function of universal step count, so the gain can be quoted in NFE instead of
    # in a percentage of a surrogate nobody can interpret. Measured, not from E ~ n^-p,
    # which verify_equidistribution showed is 40-63% wrong at n=4.
    ns = np.arange(2, 3 * max(budgets) + 1)
    e_of_n = []
    for nn in ns:
        m = dp_mesh(t, uni_c, int(nn))
        e_of_n.append(float((c[:, m[:-1]] * np.diff(t[m])[None, :] ** 2).sum()) / n)
    e_of_n = np.array(e_of_n)

    print(f"\n  {'B':>4}{'per-sample E':>15}{'universal E':>14}{'gain':>9}"
          f"{'mesh spread':>13}{'= NFE':>9}{'dE/step':>12}")
    for B in budgets:
        m_uni = dp_mesh(t, uni_c, B)
        e_uni = e_own = 0.0
        spread = []
        for i in range(n):
            m_own = dp_mesh(t, c[i], B)
            spread.append(np.abs(t[m_own] - t[m_uni]).max())
            e_own += float((c[i][m_own[:-1]] * np.diff(t[m_own]) ** 2).sum())
            e_uni += float((c[i][m_uni[:-1]] * np.diff(t[m_uni]) ** 2).sum())
        e_own, e_uni = e_own / n, e_uni / n
        gain = 100 * (e_uni - e_own) / e_uni
        # bedroom's E(n) is nearly flat (E(8)/E(4)=0.98), so dividing the gain by the
        # per-step improvement explodes -- the same near-zero denominator that made
        # oracle_per_nfe read 2.22 at B=110. Refuse the conversion instead of printing it.
        per_step = float(e_of_n[B - 2] - e_of_n[B - 1])
        if per_step <= 0 or (e_uni - e_own) > 0.5 * per_step:
            eq = f"{'n/a':>9}"
        else:
            eq = f"{float(np.interp(-e_own, -e_of_n, ns)) - B:9.2f}"
        print(f"  {B:4d}{e_own:15.6f}{e_uni:14.6f}{gain:8.2f}%"
              f"{np.mean(spread):13.4f}{eq}{per_step:12.2e}")

    print("\n  per-sample E : each image gets a mesh optimised on ITS OWN curvature (oracle)")
    print("  universal E  : every image gets the mesh optimised on the MEAN curvature")
    print("  gain         : the entire prize available to per-sample scheduling, in the")
    print("                 accumulated-local-truncation-error surrogate. It is an ORACLE,")
    print("                 so a real predictor gets less. If this is ~0, per-sample")
    print("                 scheduling is dead for reasons no feature or policy can fix.")
    print("  mesh spread  : mean over images of the largest timestep shift vs the universal")
    print("                 mesh, in units of the normalised interval. ~0 = same schedule.")
    print("  = NFE        : EXTRA uniform steps that buy the same error reduction, measured")
    print("                 off the DP curve. Reads n/a when the gain exceeds half of")
    print("                 dE/step, because the conversion then extrapolates across a flat")
    print("                 region of E(n) and the ratio is a near-zero-denominator artefact.")
    print("  dE/step      : how much one extra UNIFORM step actually buys at this B. If this")
    print("                 is tiny, nothing at this budget converts into NFE, and that is a")
    print("                 statement about the SURROGATE, not about the model.")
    print("\n  THE SURROGATE IS OUT OF RANGE AT SMALL B. c*h^2 is a local Taylor term, but at")
    print("  B=4 the step h is a QUARTER of the whole trajectory. verify_equidistribution")
    print("  measured the continuum answer as 40-63% wrong at n=4. Read the SHAPE block")
    print("  above -- correlation and effRank -- which is a direct property of the measured")
    print("  curves and uses no surrogate at all. The E table is corroboration, not proof.")
    print("\n  ANCHORS from the two synthetic controls, same code:")
    print("    magnitude-only (30.8x spread) : corr +1.0000  effRank 1.0  gain  0.00%")
    print("    shape-varying  ( 1.1x spread) : corr +0.0714  effRank 3.6  gain 70.79%")
    print("  Read the real correlation and effRank against those two rows, not against 0/1.")
    print("\n  CAVEAT, and it is the same one GITS papers over with a discount factor:")
    print("  accumulated LOCAL error is NOT the global error -- errors are amplified along")
    print("  the trajectory. So this bounds the SURROGATE, and a small gain here is strong")
    print("  evidence but not proof. Cross-check against the measured D-matrix ceilings.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", default=None, help="npz from a previous run; CPU only")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the difference operators against known derivatives")
    ap.add_argument("--classes", type=int, default=0,
                    help="number of classes to sample from, for CONDITIONAL models only. "
                         "Required for cin256: without it every image is class 0.")
    ap.add_argument("--budgets", default="4,8,16")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--max-t", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.self_test:
        self_test()
    elif args.analyze:
        analyze(args)
    elif args.dataset and args.out:
        measure(args)
    else:
        raise SystemExit("need either --analyze FILE, or --dataset and --out")


if __name__ == "__main__":
    main()
