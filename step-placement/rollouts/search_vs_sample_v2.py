"""search_vs_sample.py (frozen, md5 45d8b726fbfd76913ad53f39140bc337) plus additions only.

The frozen file produced every published search row. It cannot run two of the new cells,
so this copy adds exactly what they need and changes nothing on the default path:

  --classes K   class-conditional models (cin256). Labels are drawn the way
                curvature_probe.py drew them for Table 1, default_rng(seed).integers(0, K, m)
                with m = max(rng_images, n), keeping the first n. At n = 256 that is the
                identical draw (verified against curv_cin256.npz["classes"]). Labels reach
                every denoiser call, including the reference. No guidance: one conditional
                UNet call per step, so one step is one NFE.
  --sampler     ddim (default: the published path) or dpmpp2m, DPM-Solver++(2M) with one
                model call per step, built from solver_probe.py's verified _dpm_pieces and
                _dpm_D and romberg_dpm_sampler.dpm_exponential_step. The first step has no
                history and is exponential Euler (= DDIM); no lower-order final step. The
                200-step REFERENCE is always DDIM, as in every published run.
  outputs       D_fixed keeps its published meaning (BATCHED scores at the searched N
                schedule). Added: D_fixed_B1 (batched, at the searched N+1 schedule, needed to
                resample the denominator in the bootstrap), D_fixed_unbatched, labels, sampler
                and every setting. The D_fixed_B1 pass is NOT added to `calls`, so `calls`
                stays comparable with the frozen script.

With --classes 0 --sampler ddim this runs the frozen code path call for call;
utils/check_nov16.sh compares the two outputs array by array.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from reward_structure import (LDM_NAME, decode, lpips_to, resolve_paths,  # noqa: E402
                              run_schedule)


def uniform(T, B):
    return np.linspace(T - 1, 0, B + 1).round().astype(np.int64)


def candidates(sched, j, n_cand):
    """Legal values for interior point j: strictly between its two neighbours."""
    lo, hi = int(sched[j + 1]) + 1, int(sched[j - 1]) - 1
    if hi < lo:
        return []
    if hi - lo + 1 <= n_cand:
        return list(range(lo, hi + 1))
    return sorted(set(np.linspace(lo, hi, n_cand).round().astype(int).tolist()))


def coord_descent(score, sched, sweeps, n_cand):
    """Greedy over interior points; score(sched) -> scalar, lower is better.

    Identical to the frozen file. Returns converged=False if it ran out of sweeps.
    Also returns the STARTING score, which the caller uses as the per-image baseline.
    """
    cur = [int(v) for v in sched]
    best = score(cur)
    start = best
    n_eval = 1
    converged = False
    for _ in range(sweeps):
        improved = False
        for j in range(1, len(cur) - 1):
            for v in candidates(cur, j, n_cand):
                if v == cur[j]:
                    continue
                trial = list(cur)
                trial[j] = v
                s = score(trial)
                n_eval += 1
                if s < best - 1e-9:
                    best, cur, improved = s, trial, True
        if not improved:
            converged = True
            break
    return np.array(cur, dtype=np.int64), best, start, n_eval, converged


# ---------------------------------------------------------------------------------------
# added: conditional DDIM and DPM-Solver++(2M) rollouts
# ---------------------------------------------------------------------------------------

@torch.no_grad()
def run_schedule_cond(adapter, z_T, sched, device, labels):
    """reward_structure.run_schedule with class_labels passed to every call."""
    z = z_T
    t_cur = torch.full((z.shape[0],), int(sched[0]), device=device, dtype=torch.long)
    for j in range(1, len(sched)):
        t_next = torch.full_like(t_cur, int(sched[j]))
        out = adapter._denoise_step_latents(z, t_cur, t_next, eta=0.0, class_labels=labels)
        z = out[0] if isinstance(out, (tuple, list)) else out
        t_cur = t_next
    return z


@torch.no_grad()
def run_dpmpp2m(adapter, z_T, sched, device, labels, multistep=True):
    """DPM-Solver++(2M) over `sched`, one model call per step.

    Each step is exactly the COARSE step of solver_probe._local_error, which the
    probe's self-tests check against a closed-form answer. multistep=False drops the
    history term, giving exponential Euler, which must reproduce DDIM; the startup
    gate in main() checks that on the real checkpoint.
    """
    from romberg_dpm_sampler import dpm_exponential_step
    from solver_probe import _dpm_D, _dpm_pieces

    alphas = adapter.scheduler.alphas_cumprod
    z = z_T
    bs = z.shape[0]
    x0_prev, lam_prev = None, None
    for j in range(1, len(sched)):
        t_i, t_n = int(sched[j - 1]), int(sched[j])
        _, al_i, sg_i, lam_i = _dpm_pieces(alphas, t_i)
        _, al_n, sg_n, lam_n = _dpm_pieces(alphas, t_n)
        tb_i = torch.full((bs,), t_i, device=device, dtype=torch.long)
        tb_n = torch.full((bs,), t_n, device=device, dtype=torch.long)
        _, _, x0_i = adapter._denoise_step_latents(z, tb_i, tb_n, eta=0.0,
                                                   class_labels=labels)
        D = _dpm_D(x0_i, x0_prev, lam_prev, lam_i, lam_n) if multistep else x0_i
        z = dpm_exponential_step(z, D, sg_i, sg_n, al_n, lam_i, lam_n)
        x0_prev, lam_prev = x0_i, lam_i
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LDM_NAME))
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--rng-images", type=int, default=64,
                    help="draw this many latents then keep the first --n-images")
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--n-cand", type=int, default=12,
                    help="candidate positions tried per interior point per sweep")
    ap.add_argument("--ref-steps", type=int, default=200)
    ap.add_argument("--max-t", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--decode-chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", type=int, default=0,
                    help="number of classes for CLASS-CONDITIONAL models (1000 for cin256)")
    ap.add_argument("--sampler", choices=("ddim", "dpmpp2m"), default="ddim")
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import lpips

    from curvature_probe import _stub_clip_if_missing
    if _stub_clip_if_missing():
        print("  note: `clip` not installed; stubbed for the import only (class-conditional "
              "models do not use it, text-conditional ones will raise)")
    from adapter_compvis import create_compvis_adapter

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    T, B, n = args.max_t, args.budget, args.n_images

    print("=" * 74)
    print(f"SEARCH vs SAMPLE v2 [{args.dataset}]  B={B} NFE   n={n} images   "
          f"sampler={args.sampler}   classes={args.classes}")
    print("=" * 74)

    cfg_path, ckpt_path = resolve_paths(args.dataset, args.config, args.ckpt)
    print(f"  config {cfg_path}\n  ckpt   {ckpt_path}")
    adapter = create_compvis_adapter(cfg_path, ckpt_path, device=device,
                                     n_timesteps=args.ref_steps).to(device).eval()
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    is_cond = getattr(getattr(adapter, "model", None), "cond_stage_model", None) is not None
    if is_cond and args.classes <= 0:
        raise SystemExit(f"{args.dataset} is CLASS-CONDITIONAL; without --classes every image "
                         f"would be class 0. Pass --classes 1000 for ImageNet.")
    if not is_cond and args.classes > 0:
        raise SystemExit(f"{args.dataset} is unconditional; --classes would do nothing.")

    C = getattr(adapter, "latent_channels", 4)
    S = getattr(adapter, "latent_size", 32)
    m = max(args.rng_images, n)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    z_T = torch.randn(m, C, S, S, generator=g)[:n].to(device)
    labels = None
    if is_cond:
        rng = np.random.default_rng(args.seed)
        labels = torch.from_numpy(rng.integers(0, args.classes, m)[:n]).long().to(device)
        print(f"  conditional: {len(torch.unique(labels))} distinct classes over {n} images; "
              f"no guidance, 1 model call per step")

    def lab(i0, i1):
        return None if labels is None else labels[i0:i1]

    def ddim(z, sched, lb):
        # unconditional: the frozen file's exact call
        if lb is None:
            return run_schedule(adapter, z, sched, device)
        return run_schedule_cond(adapter, z, sched, device, lb)

    def rollout(z, sched, lb):
        if args.sampler == "ddim":
            return ddim(z, sched, lb)
        return run_dpmpp2m(adapter, z, sched, device, lb)

    if args.sampler == "dpmpp2m":
        # gate: without its history term DPM-Solver++ IS DDIM(eta=0). A wrong log-SNR
        # convention or exponential step fails this; a correct one passes to float32 noise.
        k = min(4, n)
        u = uniform(T, B)
        a = ddim(z_T[:k], u, lab(0, k))
        b1 = run_dpmpp2m(adapter, z_T[:k], u, device, lab(0, k), multistep=False)
        b2 = run_dpmpp2m(adapter, z_T[:k], u, device, lab(0, k), multistep=True)
        scale = float(a.abs().max())
        rel1 = float((b1 - a).abs().max()) / scale
        rel2 = float((b2 - a).abs().max()) / scale
        print(f"  DPM gate: first-order vs DDIM rel err {rel1:.3e} (must be < 1e-3); "
              f"2M vs DDIM rel diff {rel2:.3e} (must be > 1e-3, history term active)")
        if not rel1 < 1e-3:
            raise SystemExit("FAILED: exponential Euler does not reproduce DDIM")
        if not rel2 > 1e-3:
            raise SystemExit("FAILED: the multistep term has no effect")

    ref = np.linspace(T - 1, 0, args.ref_steps + 1).round().astype(np.int64)
    ref_z = torch.cat([ddim(z_T[i:i + args.batch], ref, lab(i, i + args.batch))
                       for i in range(0, n, args.batch)], dim=0)
    ref_img = decode(adapter, ref_z, args.decode_chunk)
    print(f"  reference generated at {args.ref_steps} NFE (DDIM)")

    calls = [0]

    def lpips_all(sched, count=True):
        if count:
            calls[0] += n * (len(sched) - 1)
        zs = torch.cat([rollout(z_T[i:i + args.batch], sched, lab(i, i + args.batch))
                        for i in range(0, n, args.batch)], dim=0)
        return lpips_to(loss_fn, decode(adapter, zs, args.decode_chunk),
                        ref_img, args.decode_chunk)

    def lpips_one(sched, i):
        calls[0] += len(sched) - 1
        z = rollout(z_T[i:i + 1], sched, lab(i, i + 1))
        return float(lpips_to(loss_fn, decode(adapter, z, 1), ref_img[i:i + 1], 1)[0])

    print(f"\n1  SEARCH THE BEST FIXED SCHEDULE AT B={B}   (objective: mean over images)")
    fB, fB_mean, _, e1, c1 = coord_descent(lambda s: float(lpips_all(s).mean()),
                                           uniform(T, B), args.sweeps, args.n_cand)
    uni_mean = float(lpips_all(uniform(T, B)).mean())
    print(f"    uniform  {uniform(T, B).tolist()}  mean LPIPS {uni_mean:.6f}")
    print(f"    searched {fB.tolist()}  mean LPIPS {fB_mean:.6f}"
          f"   ({100 * (uni_mean - fB_mean) / uni_mean:+.2f}% vs uniform,"
          f" {e1} evals, {'converged' if c1 else 'HIT SWEEP CAP'})")
    D_fB = lpips_all(fB)
    assert abs(float(D_fB.mean()) - fB_mean) < 1e-9, (
        f"NON-DETERMINISTIC: {D_fB.mean()!r} != {fB_mean!r}.")

    print(f"\n2  SEARCH THE BEST FIXED SCHEDULE AT B={B + 1}   (the yardstick, also searched)")
    fB1, fB1_mean, _, e2, c2 = coord_descent(lambda s: float(lpips_all(s).mean()),
                                             uniform(T, B + 1), args.sweeps, args.n_cand)
    print(f"    searched {fB1.tolist()}  mean LPIPS {fB1_mean:.6f}"
          f"   ({e2} evals, {'converged' if c2 else 'HIT SWEEP CAP'})")
    step_value = fB_mean - fB1_mean
    # added: per-image scores at the N+1 optimum, for the denominator bootstrap
    D_fB1 = lpips_all(fB1, count=False)
    assert abs(float(D_fB1.mean()) - fB1_mean) < 1e-9, (
        f"NON-DETERMINISTIC at B+1: {D_fB1.mean()!r} != {fB1_mean!r}.")

    print(f"\n3  SEARCH A SCHEDULE PER IMAGE AT B={B}   (the oracle, same routine)")
    per_img = np.zeros(n)
    per_base = np.zeros(n)
    per_sched = np.zeros((n, B + 1), dtype=np.int64)
    n_capped = 0
    for i in range(n):
        s_i, v_i, b_i, e_i, c_i = coord_descent(lambda s: lpips_one(s, i), fB,
                                                args.sweeps, args.n_cand)
        per_img[i], per_base[i], per_sched[i] = v_i, b_i, s_i
        n_capped += not c_i
        print(f"    image {i:3d}  {b_i:.6f} -> {v_i:.6f}"
              f"  ({100 * (b_i - v_i) / max(b_i, 1e-12):+5.2f}%, {e_i} evals"
              f"{'' if c_i else ', CAPPED'})", flush=True)

    gain = float(per_base.mean() - per_img.mean())
    n_improved = int((per_img < per_base - 1e-9).sum())
    uniq = len({tuple(r) for r in per_sched.tolist()})
    path_gap = float(np.abs(per_base - D_fB).max())
    verdict = ("negligible" if path_gap < 0.01 * max(gain, 1e-12)
               else "COMPARABLE TO THE GAIN -- the matched baseline is load-bearing here")
    print(f"\n    batched-vs-unbatched gap on the SAME schedule: max {path_gap:.3e}"
          f"  ({verdict})")

    print("\n" + "=" * 74)
    print(f"  best FIXED (searched)     : {per_base.mean():.6f}")
    print(f"  per-image  (searched)     : {per_img.mean():.6f}")
    print(f"  SEARCHED ORACLE GAIN      : {gain:.6f}")
    print(f"  one extra NFE (searched)  : {step_value:.6f}")
    if step_value > 0:
        print(f"  -> worth {gain / step_value:.3f} extra NFE")
    print(f"  images where search beat the fixed schedule : {n_improved}/{n}")
    print(f"  distinct per-image schedules found          : {uniq}/{n}")
    if n_capped:
        print(f"  WARNING: {n_capped}/{n} per-image searches hit the sweep cap; raise --sweeps")

    print(f"\n  LDM calls used: {calls[0]:,}")
    print(f"\nSUMMARY search_vs_sample dataset={args.dataset} budget={B} n={n} "
          f"searched_gain={gain:.6f} step_value={step_value:.6f} "
          f"ratio={gain / step_value if step_value > 0 else float('nan'):.4f} "
          f"n_improved={n_improved} "
          f"uniq_schedules={uniq} conv_B={int(c1)} conv_B1={int(c2)} "
          f"n_capped={n_capped} calls={calls[0]} sampler={args.sampler} "
          f"classes={args.classes}")

    if args.save_npz:
        np.savez(args.save_npz, fixed_B=fB, fixed_B1=fB1, per_sched=per_sched,
                 per_img=per_img, D_fixed=D_fB, gain=gain, step_value=step_value,
                 D_fixed_B1=D_fB1, D_fixed_unbatched=per_base,
                 n_improved=np.int64(n_improved), uniq_schedules=np.int64(uniq),
                 conv_B=np.int8(c1), conv_B1=np.int8(c2),
                 n_capped=np.int64(n_capped), calls=np.int64(calls[0]),
                 dataset=np.asarray(args.dataset), budget=np.int64(B),
                 n_images=np.int64(n), sweeps=np.int64(args.sweeps),
                 n_candidates=np.int64(args.n_cand), seed=np.int64(args.seed),
                 rng_images=np.int64(args.rng_images), ref_steps=np.int64(args.ref_steps),
                 max_t=np.int64(T), sampler=np.asarray(args.sampler),
                 classes=np.int64(args.classes),
                 labels=(labels.cpu().numpy() if labels is not None
                         else np.array([], dtype=np.int64)))
        print(f"  saved {args.save_npz}")


if __name__ == "__main__":
    main()
