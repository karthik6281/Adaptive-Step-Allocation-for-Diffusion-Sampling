"""Search shared and per-image DDIM schedules with a paired LPIPS objective.

The same coordinate-descent routine searches shared schedules at budgets B and
B+1 and a separate budget-B schedule for each image. The resulting NPZ contains
the schedules, per-image scores, summary counts, and run settings. Coordinate
descent does not certify a global optimum.
"""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.abspath(os.path.dirname(__file__))
_TAED = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _path in (_HERE, _TAED, os.path.join(_TAED, "utils")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from reward_structure import (  # noqa: E402
    LDM_NAME,
    decode,
    lpips_to,
    resolve_paths,
    run_schedule,
)


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

    converged=True means a full pass accepted no tested candidate. False means the pass
    limit was reached before that stopping test; neither value certifies a local or global
    optimum outside the sampled coordinate candidates.

    Also returns the STARTING score. The caller must use that as the baseline rather than
    a separately-computed one: a batched and an unbatched evaluation of the same schedule
    differ in the last few digits, and subtracting across the two paths books that float
    difference as a per-image gain even when the search improved nothing.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LDM_NAME))
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--rng-images", type=int, default=64,
                    help="draw this many latents then keep the first --n-images. Must match "
                         "reward_structure's --n-images or the two runs use DIFFERENT "
                         "images: torch CPU randn is not guaranteed to give the same prefix "
                         "for a different requested size.")
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--n-cand", type=int, default=12,
                    help="candidate positions tried per interior point per sweep")
    ap.add_argument("--ref-steps", type=int, default=200)
    ap.add_argument("--max-t", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--decode-chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampled-gain", type=float, default=None,
                    help="oracle_gain from the matching reward_structure run, to quantify "
                         "how much SAMPLING understated the ceiling")
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import lpips
    from adapter_compvis import create_compvis_adapter

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    T, B, n = args.max_t, args.budget, args.n_images

    print("=" * 74)
    print(f"SEARCH vs SAMPLE [{args.dataset}]  B={B} NFE   n={n} images")
    print("=" * 74)

    cfg_path, ckpt_path = resolve_paths(args.dataset, args.config, args.ckpt)
    print(f"  config {cfg_path}\n  ckpt   {ckpt_path}")
    adapter = create_compvis_adapter(cfg_path, ckpt_path, device=device,
                                     n_timesteps=args.ref_steps).to(device).eval()
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    C = getattr(adapter, "latent_channels", 4)
    S = getattr(adapter, "latent_size", 32)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    z_T = torch.randn(max(args.rng_images, n), C, S, S, generator=g)[:n].to(device)

    ref = np.linspace(T - 1, 0, args.ref_steps + 1).round().astype(np.int64)
    ref_z = torch.cat([run_schedule(adapter, z_T[i:i + args.batch], ref, device)
                       for i in range(0, n, args.batch)], dim=0)
    ref_img = decode(adapter, ref_z, args.decode_chunk)
    print(f"  reference generated at {args.ref_steps} NFE")

    calls = [0]

    def lpips_all(sched):
        calls[0] += n * (len(sched) - 1)
        zs = torch.cat([run_schedule(adapter, z_T[i:i + args.batch], sched, device)
                        for i in range(0, n, args.batch)], dim=0)
        return lpips_to(loss_fn, decode(adapter, zs, args.decode_chunk),
                        ref_img, args.decode_chunk)

    def lpips_one(sched, i):
        calls[0] += len(sched) - 1
        z = run_schedule(adapter, z_T[i:i + 1], sched, device)
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
    # re-running the same schedule must give the same LPIPS; the whole "no noise to
    # overfit" argument for searching without a held-out split depends on it
    assert abs(float(D_fB.mean()) - fB_mean) < 1e-9, (
        f"NON-DETERMINISTIC: {D_fB.mean()!r} != {fB_mean!r}. The searched optimum is then "
        f"partly fitted to sampling noise and this file's ceiling is not valid.")

    print(f"\n2  SEARCH THE BEST FIXED SCHEDULE AT B={B + 1}   (the yardstick, also searched)")
    fB1, fB1_mean, _, e2, c2 = coord_descent(lambda s: float(lpips_all(s).mean()),
                                             uniform(T, B + 1), args.sweeps, args.n_cand)
    print(f"    searched {fB1.tolist()}  mean LPIPS {fB1_mean:.6f}"
          f"   ({e2} evals, {'converged' if c2 else 'HIT SWEEP CAP'})")
    print(f"    NOTE: B+1 has {B} free interior points vs {B - 1} at B, so the yardstick is")
    print(f"    searched {B / (B - 1):.2f}x harder. That inflates step_value and therefore")
    print("    biases the ratio DOWNWARD, i.e. toward the null. Both must say 'converged'.")
    step_value = fB_mean - fB1_mean

    print(f"\n3  SEARCH A SCHEDULE PER IMAGE AT B={B}   (the oracle, same routine)")
    per_img = np.zeros(n)
    per_base = np.zeros(n)
    per_sched = np.zeros((n, B + 1), dtype=np.int64)
    n_capped = 0
    for i in range(n):
        s_i, v_i, b_i, e_i, c_i = coord_descent(
            lambda s, image_index=i: lpips_one(s, image_index),
            fB, args.sweeps, args.n_cand)
        per_img[i], per_base[i], per_sched[i] = v_i, b_i, s_i
        n_capped += not c_i
        print(f"    image {i:3d}  {b_i:.6f} -> {v_i:.6f}"
              f"  ({100 * (b_i - v_i) / max(b_i, 1e-12):+5.2f}%, {e_i} evals"
              f"{'' if c_i else ', CAPPED'})")

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
    if args.sampled_gain is not None:
        print(f"  sampled gain (M schedules): {args.sampled_gain:.6f}"
              f"   -> search found {gain / max(args.sampled_gain, 1e-12):.2f}x more")
        print(f"     (that ratio mixes n={n} here with reward_structure's n; equal only if")
        print("      --rng-images matches its --n-images, and it is still a subset mean)")

    if step_value <= 0:
        print("\n  -> one extra NFE did not help; the yardstick is broken at this budget.")
    elif gain < step_value:
        print("\n  -> THIS SEARCH FINDS LESS THAN ONE NFE. The searched per-image schedules")
        print("     buy less than one extra denoising step; this does not certify the")
        print("     global per-image optimum.")
    else:
        print("\n  -> SEARCH BEATS ONE NFE. The sampled ceiling UNDERSTATED the real one;")
        print("     the negative result does not hold and the pool was the limitation.")

    print(f"\n  LDM calls used: {calls[0]:,}")
    # max(step_value, 1e-12) would turn a NEGATIVE denominator into a huge positive ratio;
    # that clamp printed oracle_per_nfe=371,720,802 on bedroom B=110 in reward_structure.py
    print(f"\nSUMMARY search_vs_sample dataset={args.dataset} budget={B} n={n} "
          f"searched_gain={gain:.6f} step_value={step_value:.6f} "
          f"ratio={gain / step_value if step_value > 0 else float('nan'):.4f} "
          f"n_improved={n_improved} "
          f"uniq_schedules={uniq} conv_B={int(c1)} conv_B1={int(c2)} "
          f"n_capped={n_capped} calls={calls[0]}")

    if args.save_npz:
        np.savez(args.save_npz, fixed_B=fB, fixed_B1=fB1, per_sched=per_sched,
                 per_img=per_img, D_fixed=per_base, D_fixed_batched=D_fB,
                 gain=gain, step_value=step_value,
                 n_improved=np.int64(n_improved), uniq_schedules=np.int64(uniq),
                 conv_B=np.int8(c1), conv_B1=np.int8(c2),
                 n_capped=np.int64(n_capped), calls=np.int64(calls[0]),
                 dataset=np.asarray(args.dataset), budget=np.int64(B),
                 n_images=np.int64(n), sweeps=np.int64(args.sweeps),
                 n_candidates=np.int64(args.n_cand), seed=np.int64(args.seed),
                 rng_images=np.int64(args.rng_images), ref_steps=np.int64(args.ref_steps),
                 max_t=np.int64(T))
        print(f"  saved {args.save_npz}")


if __name__ == "__main__":
    main()
