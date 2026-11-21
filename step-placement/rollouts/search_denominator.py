"""Per-image LPIPS at a finished search's shared N and N+1 schedules.

The frozen search_vs_sample.py stores per-image scores at the searched N schedule
(D_fixed, batched) but only the SCALAR step_value for N+1. Resampling images for the
denominator in the bootstrap needs the per-image N+1 scores, so this re-scores both
saved schedules with the frozen file's batched path: same latents (seed, rng_images),
same 200-step DDIM reference, same batch and decode chunk.

It is also a reproduction gate. Generation and LPIPS are deterministic, so
  * the re-scored N vector must equal the stored D_fixed, and
  * the re-scored N+1 mean must equal mean(D_fixed) - step_value,
both to within --tol. If either fails, this environment does not reproduce the run and
the output is written with ok=0 and the process exits 1.

Unconditional DDIM runs only (that is all the frozen script produced). Runs from
search_vs_sample_v2.py already contain D_fixed_B1 and do not need this.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from reward_structure import (LDM_NAME, decode, lpips_to, resolve_paths,  # noqa: E402
                              run_schedule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="output of search_vs_sample.py")
    ap.add_argument("--dataset", required=True, choices=sorted(LDM_NAME))
    ap.add_argument("--n-images", type=int, required=True)
    ap.add_argument("--rng-images", type=int, default=64)
    ap.add_argument("--ref-steps", type=int, default=200)
    ap.add_argument("--max-t", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--decode-chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.dataset == "cin256":
        raise SystemExit("cin256 runs come from search_vs_sample_v2.py, which stores D_fixed_B1")

    src = np.load(args.npz)
    fB, fB1 = src["fixed_B"].astype(np.int64), src["fixed_B1"].astype(np.int64)
    D_stored = np.asarray(src["D_fixed"], float)
    step_value = float(src["step_value"])
    n = args.n_images
    if D_stored.shape != (n,):
        raise SystemExit(f"D_fixed has shape {D_stored.shape}, expected ({n},)")
    if len(fB1) != len(fB) + 1:
        raise SystemExit("fixed_B1 is not one step longer than fixed_B")

    import lpips
    from adapter_compvis import create_compvis_adapter

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    T = args.max_t
    cfg_path, ckpt_path = resolve_paths(args.dataset)
    print(f"DENOMINATOR [{args.dataset}] n={n}  B={len(fB) - 1}  from {args.npz}")
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

    def lpips_all(sched):
        zs = torch.cat([run_schedule(adapter, z_T[i:i + args.batch], sched, device)
                        for i in range(0, n, args.batch)], dim=0)
        return lpips_to(loss_fn, decode(adapter, zs, args.decode_chunk),
                        ref_img, args.decode_chunk)

    D_B = lpips_all(fB).astype(float)
    D_B1 = lpips_all(fB1).astype(float)

    dev_B = float(np.abs(D_B - D_stored).max())
    implied_B1 = float(D_stored.mean()) - step_value
    dev_B1 = abs(float(D_B1.mean()) - implied_B1)
    ok = dev_B <= args.tol and dev_B1 <= args.tol
    print(f"  N   per-image max |re-scored - stored D_fixed| = {dev_B:.3e}")
    print(f"  N+1 |mean re-scored - (mean D_fixed - step_value)| = {dev_B1:.3e}")
    print(f"  step_value stored {step_value:.6f}   re-scored {D_B.mean() - D_B1.mean():.6f}")
    print(f"  REPRODUCES: {'YES' if ok else 'NO -- do not use; environment differs'}"
          f"  (tol {args.tol:g})")

    np.savez(args.out, D_B=D_B, D_B1=D_B1, fixed_B=fB, fixed_B1=fB1,
             dev_B=dev_B, dev_B1=dev_B1, ok=np.int8(ok), tol=args.tol,
             source=np.asarray(os.path.basename(args.npz)), dataset=np.asarray(args.dataset),
             n_images=np.int64(n), rng_images=np.int64(args.rng_images),
             seed=np.int64(args.seed), ref_steps=np.int64(args.ref_steps))
    print(f"  saved {args.out}")
    print(f"\nDENOM_SUMMARY source={os.path.basename(args.npz)} ok={int(ok)} "
          f"dev_B={dev_B:.3e} dev_B1={dev_B1:.3e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
