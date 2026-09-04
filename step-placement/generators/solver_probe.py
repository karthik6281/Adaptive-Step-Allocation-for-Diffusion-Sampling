"""Measure error profiles for DPM-Solver++(2M) and stochastic DDIM.

DPM-Solver++ local error is estimated by comparing one coarse step with two
half steps. The stochastic DDIM path implements the eta-dependent variance term
and checks that eta=0 matches the deterministic adapter. ``--self-test`` runs
analytic checks without loading a model checkpoint.
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_TAED = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _TAED, os.path.join(_TAED, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from curvature_probe import _stub_clip_if_missing, shape_stats  # noqa: E402


def _alphas_at(alphas, t, batch):
    import torch
    idx = torch.full((batch,), int(t), device=alphas.device, dtype=torch.long)
    return alphas.gather(0, idx.clamp(min=0)).view(-1, 1, 1, 1)


def ddim_eta_step(eps, x0, a_cur, a_next, eta, noise):
    """DDIM update with the sigma term the adapter drops. eta=0 reduces to the adapter.

    Returns (next state, drift). The second difference of the STATE necessarily contains
    sigma*(xi_j - 2 xi_{j-1} + xi_{j-2}), and those xi are independent ACROSS images, so at
    large eta the cross-image correlation of the state profile falls for a reason that has
    nothing to do with curvature: it is measuring the noise we injected. The drift is the
    part that carries the trajectory geometry and is the quantity comparable with eta=0.
    """
    sig = eta * (((1.0 - a_next) / (1.0 - a_cur)).clamp(min=0).sqrt()
                 * (1.0 - a_cur / a_next).clamp(min=0).sqrt())
    dir_z = (1.0 - a_next - sig ** 2).clamp(min=0).sqrt()
    drift = a_next.sqrt() * x0 + dir_z * eps
    return drift + sig * noise, drift


def _noise_for(gens, i0, bs, shape, device):
    import torch
    return torch.stack([torch.randn(shape, generator=gens[i0 + k]) for k in range(bs)]).to(device)


def _load(args):
    import torch
    if _stub_clip_if_missing():
        print("  note: `clip` stubbed for the import only; text-conditional models will raise")
    from adapter_compvis import create_compvis_adapter
    from reward_structure import resolve_paths

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    cfg_path, ckpt_path = resolve_paths(args.dataset, args.config, args.ckpt)
    adapter = create_compvis_adapter(cfg_path, ckpt_path, device=device,
                                     n_timesteps=args.steps).to(device).eval()

    is_cond = getattr(getattr(adapter, "model", None), "cond_stage_model", None) is not None
    if is_cond and args.classes <= 0:
        raise SystemExit(f"{args.dataset} is CLASS-CONDITIONAL; without --classes every "
                         f"image would come from class 0 and the shape spread would be "
                         f"understated. Pass --classes 1000 for ImageNet.")
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
    return adapter, z_T, labels, device, (C, S)


def measure_eta(args):
    import torch
    adapter, z_T, labels, device, (C, S) = _load(args)
    alphas = adapter.scheduler.alphas_cumprod
    sched = np.linspace(args.max_t - 1, 0, args.steps + 1).round().astype(np.int64)
    curv = np.zeros((args.n_images, args.steps - 1), dtype=np.float64)
    drift_curv = np.zeros((args.n_images, args.steps - 1), dtype=np.float64)
    gens = [torch.Generator(device="cpu").manual_seed(args.seed * 1000003 + i)
            for i in range(args.n_images)]
    print(f"  {args.dataset}  eta={args.eta}  n={args.n_images}  {args.steps} steps  "
          f"{args.n_images * args.steps:,} LDM calls")
    checked = False

    for b0 in range(0, args.n_images, args.batch):
        z = z_T[b0:b0 + args.batch]
        bs = z.shape[0]
        lab = None if labels is None else labels[b0:b0 + args.batch]
        prev2, prev1 = None, z
        prev2d, prev1d = None, z
        t_cur = torch.full((bs,), int(sched[0]), device=device, dtype=torch.long)
        for j in range(1, len(sched)):
            t_next = torch.full_like(t_cur, int(sched[j]))
            z_ddim, eps, x0 = adapter._denoise_step_latents(z, t_cur, t_next, eta=0.0,
                                                            class_labels=lab)
            a_cur = _alphas_at(alphas, sched[j - 1], bs)
            a_next = _alphas_at(alphas, sched[j], bs)
            noise = _noise_for(gens, b0, bs, (C, S, S), device)
            z_new, drift = ddim_eta_step(eps, x0, a_cur, a_next, args.eta, noise)

            if not checked:
                z_chk, _ = ddim_eta_step(eps, x0, a_cur, a_next, 0.0,
                                         torch.zeros_like(noise))
                d0 = (z_chk - z_ddim).abs().max().item()
                if d0 > 1e-4:
                    raise SystemExit(f"eta=0 reimplementation disagrees with the adapter by "
                                     f"{d0:.3e}; the sigma term is wired up wrong")
                d1 = (z_new - z_ddim).abs().max().item()
                if args.eta > 0 and d1 <= 1e-6:
                    raise SystemExit(f"eta={args.eta} changed the step by only {d1:.3e}. The "
                                     f"noise is not reaching the trajectory and this run "
                                     f"would report the DETERMINISTIC answer.")
                print(f"  checks: eta=0 matches adapter ({d0:.2e}), "
                      f"eta={args.eta} perturbs by {d1:.3e}")
                checked = True

            z = z_new
            t_cur = t_next
            if prev2 is not None:
                d2 = (z - 2.0 * prev1 + prev2).flatten(1).norm(dim=1)
                curv[b0:b0 + bs, j - 2] = d2.double().cpu().numpy()
                d2d = (drift - 2.0 * prev1d + prev2d).flatten(1).norm(dim=1)
                drift_curv[b0:b0 + bs, j - 2] = d2d.double().cpu().numpy()
            prev2, prev1 = prev1, z
            prev2d, prev1d = prev1d, drift
        print(f"    images {b0}-{b0 + bs - 1} done")

    _save(args, curv, sched, labels,
          extra={"eta": args.eta, "sampler": "ddim-eta", "curv_drift": drift_curv})


def self_test():
    """DPM-Solver++ at first order IS DDIM(eta=0). That identity pins the log-SNR
    convention and the exponential step, needs no checkpoint, and is what catches a
    factor of 2 in lambda -- which otherwise surfaces only as a wrong measured order,
    long after a GPU has been committed.
    """
    import torch
    from romberg_dpm_sampler import dpm_exponential_step, get_alpha_sigma, get_lambda

    torch.manual_seed(0)
    abar = torch.linspace(0.9999, 1e-4, 1000)      # index 0 cleanest, as in alphas_cumprod
    print("=" * 74)
    print("SELF-TEST  DPM-Solver++(1) must reproduce DDIM(eta=0) exactly")
    print("=" * 74)
    worst, worst_repo = 0.0, 0.0
    for i, j in ((900, 700), (700, 350), (350, 40), (40, 5), (500, 499)):
        al_i, sg_i = get_alpha_sigma(abar[i])
        al_j, sg_j = get_alpha_sigma(abar[j])
        x0, eps = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)
        z = al_i * x0 + sg_i * eps
        ddim = al_j * x0 + sg_j * eps
        scale = ddim.abs().max().item()

        lam_i = torch.log(al_i) - torch.log(sg_i)
        lam_j = torch.log(al_j) - torch.log(sg_j)
        got = dpm_exponential_step(z, x0, sg_i, sg_j, al_j, lam_i, lam_j)
        rel = (got - ddim).abs().max().item() / scale
        worst = max(worst, rel)

        repo = dpm_exponential_step(z, x0, sg_i, sg_j, al_j,
                                    get_lambda(abar[i]), get_lambda(abar[j]))
        worst_repo = max(worst_repo, (repo - ddim).abs().max().item() / scale)
        print(f"  t {i:4d} -> {j:4d}   log(alpha/sigma) rel err {rel:.2e}")

    print(f"\n  this file's convention      worst rel err {worst:.3e}")
    print(f"  romberg_dpm_sampler.get_lambda worst rel err {worst_repo:.3e}")
    print("  the second number is large because get_lambda returns log(abar/(1-abar))")
    print("  = 2*lambda despite its docstring. dpm_solver_pp_update is separately wrong,")
    print("  as the next test shows, so the two defects are independent.")
    if worst > 1e-4:
        raise SystemExit(f"FAILED: {worst:.3e} -- the exponential step does not reduce to DDIM")
    print("  PASS")

    # For a linear x0(lambda) the exact step is closed form, so the extrapolation
    # coefficient can be checked without a model. r=1 is deliberately included to show
    # that a uniform grid HIDES the inverted-ratio bug.
    print("\n" + "=" * 74)
    print("SELF-TEST  multistep D must give order 3 at every h_prev/h ratio")
    print("=" * 74)
    a, b, lam_c = 0.7, 1.3, 0.4
    ok = True
    for rho in (1.0, 2.0, 0.5):
        got = []
        for h in (0.2, 0.1, 0.05):
            hp, lam_t = rho * h, lam_c + h
            xc, xp = a + b * lam_c, a + b * (lam_c - hp)
            r = h / hp
            exact = (a + b * lam_t) * (1 - np.exp(-h)) - b * (1 - np.exp(-h) * (1 + h))
            mine = (1 - np.exp(-h)) * ((1 + r / 2) * xc - (r / 2) * xp)
            repo = (1 - np.exp(-h)) * ((1 + 1 / (2 * r)) * xc - (1 / (2 * r)) * xp)
            got.append((abs(mine - exact), abs(repo - exact)))
        o_mine = np.log2(got[-2][0] / max(got[-1][0], 1e-300))
        o_repo = np.log2(got[-2][1] / max(got[-1][1], 1e-300))
        ok &= o_mine > 2.4
        print(f"  h_prev={rho:4.1f}h   (1+r/2) order {o_mine:5.2f}   "
              f"(1+1/2r) order {o_repo:5.2f}")
    print("  the two columns agree only at h_prev=1.0h. Everywhere else the inverted")
    print("  ratio collapses DPM-Solver++(2M) to first order.")
    if not ok:
        raise SystemExit("FAILED: the multistep coefficient is not second order")
    print("  PASS")

    _pipeline_test()


class _AnalyticAdapter:
    """A 'model' whose data prediction is a QUADRATIC in lambda, independent of state.

    The exact ODE solution is then closed form, so the ASSEMBLED _local_error path --
    _dpm_pieces, _dpm_D, the midpoint, both exponential steps -- can be checked against a
    known answer rather than against my own reasoning about it.

    Quadratic, not linear, and that is the whole point: linear extrapolation of a LINEAR
    x0 is exact for ANY ratio, so a linear model gives byte-identical errors at h_prev=h
    and 2h and passes just as happily with the ratio inverted. The curvature term is what
    makes the coefficient observable.
    """

    def __init__(self, alphas, A=0.7, B=1.3, C=0.9):
        import types
        self.scheduler = types.SimpleNamespace(alphas_cumprod=alphas)
        self.A, self.B, self.C = A, B, C

    def _denoise_step_latents(self, z, t_cur, t_next, eta=0.0, class_labels=None):
        import torch
        a = self.scheduler.alphas_cumprod[int(t_cur[0])]
        al, sg = a.sqrt(), (1.0 - a).sqrt()
        lam = torch.log(al) - torch.log(sg)
        x0 = torch.full_like(z, float(self.A + self.B * lam + self.C * lam * lam))
        return None, (z - al * x0) / sg, x0


def _pipeline_test():
    import torch
    betas = torch.linspace(1e-4, 2e-2, 1000, dtype=torch.float64)
    alphas = torch.cumprod(1.0 - betas, 0)
    ad = _AnalyticAdapter(alphas)
    dev = torch.device("cpu")
    torch.manual_seed(0)
    z = torch.randn(2, 1, 4, 4, dtype=torch.float64)
    t0 = 500

    print("\n" + "=" * 74)
    print("SELF-TEST  the assembled _local_error path, on a model with a known answer")
    print("=" * 74)
    fine, orders_by_rho = {}, {}
    for invert in (False, True):
        print(f"  {'INVERTED (negative control)' if invert else 'correct (1 + r/2)'}")
        for rho in (1, 2):
            prev, orders = None, []
            for h in (160, 80, 40):
                t_prev = t0 + rho * h
                lam_prev = _dpm_pieces(alphas, t_prev)[3]
                _, _, x0_prev = ad._denoise_step_latents(z, torch.full((2,), t_prev),
                                                         torch.full((2,), t0))
                e, _, _, _ = _local_error(ad, z, None, alphas, t0, t0 - h,
                                          x0_prev, lam_prev, dev, invert)
                e = float(e.mean())
                line = f"    h_prev={rho}h  h={h:3d}   local error {e:.6e}"
                if prev is not None:
                    orders.append(float(np.log2(prev / max(e, 1e-300))))
                    line += f"   order {orders[-1]:5.2f}"
                print(line)
                prev = e
            if not invert:
                fine[rho], orders_by_rho[rho] = prev, orders

    # a linear x0 would make these identical and the whole test vacuous
    if abs(fine[1] - fine[2]) < 1e-12 * max(fine[1], fine[2]):
        raise SystemExit("FAILED: h_prev=h and h_prev=2h gave identical errors, so this "
                         "test does not exercise the multistep ratio at all")
    for rho in (1, 2):
        p = orders_by_rho[rho][-1]
        if not 2.7 < p < 3.3:
            raise SystemExit(f"FAILED: h_prev={rho}h converged to order {p:.2f}, not 3, where "
                             f"the answer is known analytically. The reported local error is "
                             f"not the local truncation error of DPM-Solver++(2M).")
    print("  PASS -- so a lower order on a checkpoint is about the learned x0, not this code")


def _dpm_pieces(alphas, t):
    """(alpha_bar, alpha, sigma, log-SNR) at an integer timestep.

    romberg_dpm_sampler.get_lambda is NOT used here: it returns log(abar/(1-abar)), which
    is log(alpha^2/sigma^2) = 2*lambda, while its own docstring says log(alpha/sigma).
    DPM-Solver++ at first order must reduce to DDIM, and that forces
    exp(h) = (alpha_i/sigma_i)/(alpha_prev/sigma_prev), i.e. lambda = log(alpha/sigma).
    With h doubled, dpm_exponential_step pairs a true sigma ratio with expm1(-2h) and is
    not even first-order consistent. The factor of 2 does cancel inside r = h/h_prev, but
    dpm_solver_pp_update is separately wrong for inverting that ratio -- see _dpm_D. Treat
    romberg_dpm_sampler's sampler output as untrustworthy on both counts.
    """
    import torch
    from romberg_dpm_sampler import get_alpha_sigma
    a = alphas[int(t)]
    al, sg = get_alpha_sigma(a)
    return a, al, sg, torch.log(al) - torch.log(sg)


def _dpm_D(x0_curr, x0_prev, lam_prev, lam_curr, lam_next, invert=False):
    """DPM-Solver++(2M) data prediction, linearly extrapolated to the step midpoint.

    NOT romberg_dpm_sampler.dpm_solver_pp_update, which computes (1 + 1/(2r)) with
    r = h/h_prev and so INVERTS the ratio. Extrapolating x0(lambda) through
    (lam_prev, x0_prev) and (lam_curr, x0_curr) to lam_curr + h/2 gives (1 + r/2).
    The two agree only at r = 1, and against the closed-form integral for a linear
    x0(lambda) the inverted form measures order 1.96 at h_prev = 2h instead of 3.
    A grid uniform in t is NOT uniform in lambda, so r != 1 everywhere here.

    invert=True reproduces that bug on purpose, as the order check's negative control.
    """
    if x0_prev is None or lam_prev is None:
        return x0_curr
    h_prev = lam_curr - lam_prev
    if abs(float(h_prev)) < 1e-12:
        return x0_curr
    r = (lam_next - lam_curr) / h_prev
    if invert:
        r = 1.0 / r
    return (1.0 + r / 2.0) * x0_curr - (r / 2.0) * x0_prev


def _local_error(adapter, z, lab, alphas, t_i, t_n, x0_prev, lam_prev, device, invert=False):
    """One coarse DPM-Solver++(2M) step over [t_i, t_n] vs two half steps. Returns
    (local error per image, z along the fine path, x0 at t_i, lambda at t_i)."""
    import torch
    from romberg_dpm_sampler import dpm_exponential_step

    t_m = (int(t_i) + int(t_n)) // 2
    if not (int(t_n) < t_m < int(t_i)):
        raise SystemExit(f"degenerate midpoint {t_m} in [{t_n},{t_i}]: the grid is finer "
                         f"than the integer timestep axis, so the half step is not defined")

    bs = z.shape[0]
    tb = lambda v: torch.full((bs,), int(v), device=device, dtype=torch.long)  # noqa: E731
    _, al_i, sg_i, lam_i = _dpm_pieces(alphas, t_i)
    _, al_m, sg_m, lam_m = _dpm_pieces(alphas, t_m)
    _, al_n, sg_n, lam_n = _dpm_pieces(alphas, t_n)

    _, _, x0_i = adapter._denoise_step_latents(z, tb(t_i), tb(t_n), eta=0.0, class_labels=lab)

    D_c = _dpm_D(x0_i, x0_prev, lam_prev, lam_i, lam_n, invert)
    z_coarse = dpm_exponential_step(z, D_c, sg_i, sg_n, al_n, lam_i, lam_n)

    D_1 = _dpm_D(x0_i, x0_prev, lam_prev, lam_i, lam_m, invert)
    z_mid = dpm_exponential_step(z, D_1, sg_i, sg_m, al_m, lam_i, lam_m)
    _, _, x0_m = adapter._denoise_step_latents(z_mid, tb(t_m), tb(t_n), eta=0.0, class_labels=lab)
    D_2 = _dpm_D(x0_m, x0_i, lam_i, lam_m, lam_n, invert)
    z_fine = dpm_exponential_step(z_mid, D_2, sg_m, sg_n, al_n, lam_m, lam_n)

    return (z_coarse - z_fine).flatten(1).norm(dim=1), z_fine, x0_i, lam_i


def _walk(adapter, z, lab, t_from, t_to, step, device):
    """DDIM-integrate z from t_from down to t_to with uniform `step`, returning the state.

    The step here is a FIXED fine value, deliberately not tied to the h being tested. If
    the walk used step = h, then h=80 would reach t0 in 4 coarse steps and h=20 in 16 fine
    ones, so the convergence test would be comparing local errors measured at three
    different states -- and the state-dependence of the error constant shows up as an
    h-independent floor that flattens the order.
    """
    import torch
    n = max(1, int(round((t_from - t_to) / step)))
    ts = np.linspace(t_from, t_to, n + 1).round().astype(int)
    tb = lambda v: torch.full((z.shape[0],), int(v), device=device, dtype=torch.long)  # noqa: E731
    for k in range(1, len(ts)):
        z, _, _ = adapter._denoise_step_latents(z, tb(ts[k - 1]), tb(ts[k]),
                                                eta=0.0, class_labels=lab)
    return z


def order_check(args):
    """Diagnostic on a real checkpoint: how large is the multistep term, and what order
    does the learned x0 actually support?

    This does NOT certify the solver -- a checkpoint has no known answer. --self-test's
    analytic case is what does that. Here the useful outputs are the ratio against the
    inverted coefficient and the observed order, which sets the grid for the real run.
    """
    import torch
    args.n_images = min(args.n_images, args.batch)
    adapter, z_T, labels, device, _ = _load(args)
    alphas = adapter.scheduler.alphas_cumprod
    z = z_T[:args.batch]
    lab = None if labels is None else labels[:args.batch]
    tb = lambda v: torch.full((z.shape[0],), int(v), device=device, dtype=torch.long)  # noqa: E731

    t0, H, FINE = 500, args.order_h, 5
    if t0 + 2 * H > args.max_t - 1 or t0 - H < 1:
        raise SystemExit(f"--order-h {H} does not fit: the sweep needs t0+2h <= "
                         f"{args.max_t - 1} and t0-h >= 1 with t0={t0}")

    print("=" * 74)
    print(f"DPM-Solver++(2M) ORDER CHECK   local error vs step size   (h0={H})")
    print("=" * 74)
    hs = (H, H // 2, H // 4)

    # the walk does not depend on the coefficient, so build the states once and run both
    # variants on identical inputs
    states = {}
    for rho in (1, 2):
        for h in hs:
            t_prev = t0 + rho * h
            z_prev = _walk(adapter, z, lab, args.max_t - 1, t_prev, FINE, device)
            _, _, x0_prev = adapter._denoise_step_latents(z_prev, tb(t_prev), tb(t0),
                                                          eta=0.0, class_labels=lab)
            z0 = _walk(adapter, z_prev, lab, t_prev, t0, FINE, device)
            _, _, _, lam_prev = _dpm_pieces(alphas, t_prev)
            lam_0 = _dpm_pieces(alphas, t0)[3]
            lam_n = _dpm_pieces(alphas, t0 - h)[3]
            r = float((lam_n - lam_0) / (lam_0 - lam_prev))
            states[(rho, h)] = (z0, x0_prev, lam_prev, r)

    # An absolute threshold is the wrong test. At h0=160 one step spans a sixth of the
    # trajectory, so the sweep is pre-asymptotic and the correct method reads ~2.5 rather
    # than 3. What separates a working 2M from exponential Euler is the GAP against the
    # inverted-ratio coefficient measured on identical states, which that offset cancels.
    res, fin = {}, {}
    for invert in (False, True):
        print(f"  {'INVERTED (negative control)' if invert else 'correct (1 + r/2)'}")
        for rho in (1, 2):
            prev, orders = None, []
            for h in hs:
                z0, x0_prev, lam_prev, r = states[(rho, h)]
                e, _, _, _ = _local_error(adapter, z0, lab, alphas, t0, t0 - h,
                                          x0_prev, lam_prev, device, invert)
                e = float(e.mean())
                coef = 1 + (1 / (2 * r) if invert else r / 2)
                line = (f"    h_prev={rho}h  h={h:3d}   r={r:5.3f}  coeff {coef:5.3f}   "
                        f"local error {e:.6e}")
                if prev is not None:
                    p = float(np.log2(prev / max(e, 1e-30)))
                    orders.append(p)
                    line += f"   order {p:5.2f}"
                print(line)
                prev = e
            res[(invert, rho)] = orders[-1]
            fin[(invert, rho)] = prev

    # This is a DIAGNOSTIC, not the gate. A checkpoint has no known answer, so it cannot
    # certify anything -- --self-test's analytic pipeline case does that. And the slope
    # test does not even apply here: the learned x0 is rough enough that BOTH variants cap
    # out well below 3, which compresses the difference in slope to nothing. What does
    # survive is magnitude: at a fixed h the correct extrapolation must make the smaller
    # error, and that needs no asymptotic assumption.
    print("\n  correctness signal is the ERROR RATIO; the order is reported for grid choice.")
    ok = True
    for rho in (1, 2):
        ratio = fin[(True, rho)] / max(fin[(False, rho)], 1e-30)
        ok &= ratio > 1.5
        print(f"  h_prev={rho}h   correct {fin[(False, rho)]:.4e}   "
              f"inverted {fin[(True, rho)]:.4e}   inverted/correct {ratio:5.2f}x   "
              f"{'ok' if ratio > 1.5 else 'NOT SEPARATED'}")
    if not ok:
        raise SystemExit("\n  FAILED: the inverted coefficient is not producing a clearly "
                         "larger local error, so this checkpoint cannot confirm the "
                         "multistep term is doing anything.")

    p_obs = max(res[(False, 1)], res[(False, 2)])
    print(f"\n  observed order on this checkpoint: {p_obs:.2f} (analytic case gives 3.0).")
    print("  the shortfall is the learned x0, not the solver. It matters for --steps: the")
    print(f"  measured error scales ~h^{p_obs:.1f}, so at --steps 200 (h~5) the signal is")
    print(f"  ~{(5.0 / hs[-1]) ** p_obs:.1e} of the h={hs[-1]} value. Pick a grid that keeps it")
    print("  well clear of float32 cancellation, and quote curv_raw if h^3 is not the law.")
    print("  PASS")


def measure_dpmpp(args):
    adapter, z_T, labels, device, _ = _load(args)
    alphas = adapter.scheduler.alphas_cumprod
    sched = np.linspace(args.max_t - 1, 0, args.steps + 1).round().astype(np.int64)
    err = np.zeros((args.n_images, args.steps), dtype=np.float64)

    # DPM-Solver++ integrates in log-SNR, so a grid uniform in t is NOT uniform in lambda
    # and the measured error is c(t)*h_lambda(t)^p, not c(t). h_lambda depends only on the
    # schedule, so it is a reweighting every image receives identically -- the same way the
    # t=0 boundary spike inflated the order-1 correlations. It is known exactly, so divide
    # it out. p is NOT assumed to be the theoretical 3: --order-check measures ~2.55 on
    # church256 because the learned x0 is rough, and dividing by the wrong power leaves a
    # t-dependent residual. curv_raw keeps the undivided profile either way.
    lam = np.array([float(_dpm_pieces(alphas, t)[3]) for t in sched])
    hlam = np.diff(lam)
    if hlam.min() <= 0:
        raise SystemExit("log-SNR is not increasing along the schedule")

    print(f"  {args.dataset}  DPM-Solver++(2M)  n={args.n_images}  {args.steps} steps  "
          f"~{args.n_images * args.steps * 2:,} LDM calls")
    print(f"  h_lambda varies {hlam.max() / hlam.min():.1f}x over the grid; "
          f"dividing by h_lambda^{args.dpm_exponent}")
    if args.dpm_exponent == 3.0:
        print("  NOTE: exponent 3 is the theoretical value. Set --dpm-exponent to the order")
        print("  --order-check reports for this checkpoint if it is materially below 3.")

    for b0 in range(0, args.n_images, args.batch):
        z = z_T[b0:b0 + args.batch]
        bs = z.shape[0]
        lab = None if labels is None else labels[b0:b0 + args.batch]
        x0_prev, lam_prev = None, None
        for j in range(1, len(sched)):
            e, z, x0_prev, lam_prev = _local_error(adapter, z, lab, alphas,
                                                   sched[j - 1], sched[j],
                                                   x0_prev, lam_prev, device)
            err[b0:b0 + bs, j - 1] = e.double().cpu().numpy()
        print(f"    images {b0}-{b0 + bs - 1} done")

    # the first step has no previous x0, so the multistep degenerates to exponential Euler
    # there and its error is not comparable with the rest of the row
    raw = err[:, 1:]
    _save(args, raw / hlam[None, 1:] ** args.dpm_exponent, sched, labels,
          extra={"sampler": "dpmpp", "order": 2, "curv_raw": raw, "hlam": hlam[1:],
                 "exponent": args.dpm_exponent})


def _save(args, curv, sched, labels, extra):
    if not np.isfinite(curv).all() or curv.sum(axis=1).min() <= 0:
        raise SystemExit("non-finite or all-zero profile; shape normalisation would give NaN")
    np.savez(args.out, curv=curv, sched=sched, dataset=args.dataset,
             classes=(labels.cpu().numpy() if labels is not None else np.array([])), **extra)
    print(f"  saved {args.out}  curv{curv.shape}")


def analyze(args):
    z = np.load(args.analyze, allow_pickle=True)
    c = z["curv"]
    n, L = c.shape
    sampler = str(z["sampler"]) if "sampler" in z.files else "?"
    tag = f"eta={float(z['eta'])}" if "eta" in z.files else "DPM-Solver++(2M) local error"

    print("=" * 78)
    print(f"PROFILE SHAPE ACROSS SAMPLES   {z['dataset']}  {sampler}  {tag}  n={n}  {L} pts")
    print("=" * 78)
    print(f"  {'':26s}{'magnitude':>10}{'corr mean':>10}{'corr min':>10}{'effRank':>9}{'dev':>9}")
    shape_stats(c, "measured profile")
    shape_stats(c, "last 10 dropped", trim=10)
    if "curv_raw" in z.files:
        shape_stats(z["curv_raw"], "raw error, h^3 not removed")
        shape_stats(z["curv_raw"], "raw error, last 10 dropped", trim=10)
    if "curv_drift" in z.files:
        shape_stats(z["curv_drift"], "drift only")
        shape_stats(z["curv_drift"], "drift, last 10 dropped", trim=10)
    shape_stats(np.ones((n, 1)) * np.linspace(1, 2, L)[None, :]
                * np.linspace(1, 4, n)[:, None], "CONTROL magnitude-only")
    print("    compare corr min against the eta=0 second-difference row for the same model.")
    print("    QUOTE THE TRIMMED ROW: the last steps land on t=0 and carry a feature every")
    print("    sample shares, which inflates the worst pair.")
    if "curv_raw" in z.files:
        p = float(z["exponent"]) if "exponent" in z.files else 3.0
        print(f"    THE TOP ROWS ARE c(t), with h_lambda^{p} divided out. The raw rows still")
        print("    contain it, and h_lambda is common to every image, so they are reweighted")
        print("    identically and their agreement is overstated.")
    if "curv_drift" in z.files:
        print("    THE DRIFT ROWS ARE THE ONES COMPARABLE WITH eta=0. The state profile also")
        print("    contains the injected noise, which is independent across images, so its")
        print("    correlation must fall as eta grows whatever the curvature does.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=("dpmpp", "ddim-eta"), default="dpmpp")
    ap.add_argument("--analyze", default=None, help="npz from a previous run; CPU only")
    ap.add_argument("--self-test", action="store_true",
                    help="DPM-Solver++(1) vs DDIM identity; no checkpoint needed")
    ap.add_argument("--order-check", action="store_true",
                    help="verify DPM-Solver++ really is 2nd order before measuring")
    ap.add_argument("--order-h", type=int, default=160,
                    help="largest step for the order sweep; raise it if the order flattens")
    ap.add_argument("--dpm-exponent", type=float, default=3.0,
                    help="power of h_lambda to divide out; set from --order-check, not 3")
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--classes", type=int, default=0)
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
        return self_test()
    if args.analyze:
        return analyze(args)
    if not args.dataset:
        raise SystemExit("--dataset is required unless --analyze is given")
    if args.order_check:
        return order_check(args)
    if not args.out:
        raise SystemExit("--out is required")
    if args.sampler == "ddim-eta":
        if args.eta <= 0:
            raise SystemExit("--sampler ddim-eta with --eta 0 measures nothing new; "
                             "curvature_probe.py already covers eta=0")
        # beyond eta=1 the DDIM variance exceeds 1 - a_next and the direction term is
        # clamped to zero, which silently stops being a DDIM step at all
        if args.eta > 1.0:
            raise SystemExit(f"--eta {args.eta} > 1: 1 - a_next - sigma^2 goes negative and "
                             f"the direction term would be silently clamped. eta=1 is "
                             f"already full DDPM ancestral sampling.")
        return measure_eta(args)
    return measure_dpmpp(args)


if __name__ == "__main__":
    main()
