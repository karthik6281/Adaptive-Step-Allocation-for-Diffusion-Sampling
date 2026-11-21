"""Reference-floor check for the compvis rows (church256, bedroom256, cin256).

Every row scores its rollouts against a 200-step DDIM reference. If a row's LPIPS at its
largest budget N+1 is not well above the error of that reference itself, the step value (and
so R) partly measures the reference rather than the sampler. On the first --k images of the
rows' own latent draw this measures

    floor = LPIPS(200-step DDIM reference, DDIM over every timestep: 999 steps)

and, for every --rows run on that draw, re-scores the run's stored schedules against BOTH
references on those images: shared N and N+1 on the batched path (step value), shared N and
each image's own schedule on the per-image path (gain). So the subset's gain, step value and R
can be compared under the 200-step and the 999-step reference, paired image by image.

Gates (each must hold to --tol, else exit 1 before anything is saved): on these images the
re-scored batched shared N equals the stored D_fixed (batched in both the frozen script and
v2), and each image's own schedule equals the stored per_img; for v2 files also batched
shared N+1 vs D_fixed_B1 and per-image shared N vs D_fixed_unbatched. Settings are not stored
by the frozen script, so they come from the command line and the gates check them.

  python3 -u utils/ref_floor.py --dataset church256 --draw 64 --k 32 \\
      --rows runs1/search_church256_B16_n64_sw10.npz ... --out X.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def fail(msg):
    print(f"FAILED: {msg}", flush=True)
    sys.exit(1)


def row_info(path, args):
    src = np.load(path)
    per_img = np.asarray(src["per_img"], np.float64)
    n = len(per_img)
    v2 = "sampler" in src.files
    info = dict(path=path, src=src, n=n, v2=v2, per_img=per_img,
                fB=np.asarray(src["fixed_B"]).astype(np.int64),
                fB1=np.asarray(src["fixed_B1"]).astype(np.int64),
                per_sched=np.asarray(src["per_sched"]).astype(np.int64),
                D_fixed=np.asarray(src["D_fixed"], np.float64),
                gain=float(src["gain"]), step=float(src["step_value"]),
                sampler=str(src["sampler"]) if v2 else "ddim")
    if max(args.rng_images, n) != args.draw:
        fail(f"{path}: n={n} draws {max(args.rng_images, n)} latents, not --draw {args.draw}")
    if args.k > n:
        fail(f"{path}: --k {args.k} > n={n}")
    if len(info["fB1"]) != len(info["fB"]) + 1 or info["D_fixed"].shape != (n,):
        fail(f"{path}: unexpected array shapes")
    if v2:
        for key, want in (("dataset", args.dataset), ("classes", args.classes),
                          ("seed", args.seed), ("rng_images", args.rng_images),
                          ("ref_steps", args.ref_steps), ("max_t", args.max_t)):
            got = src[key].item()
            if (str(got) if isinstance(want, str) else int(got)) != want:
                fail(f"{path}: npz {key}={got!r}, expected {want!r}")
        if info["sampler"] not in ("ddim", "dpmpp2m"):
            fail(f"{path}: unknown sampler {info['sampler']}")
    else:
        if args.classes != 0:
            fail(f"{path}: a frozen-script npz is unconditional; --classes must be 0")
        # the frozen file stores no per-image N+1 scores; search_denominator.py's output does
        base = os.path.basename(path)
        for d in (path[:-4] + "_denom.npz", os.path.join(args.denom_dir, "denom_" + base)):
            if os.path.exists(d):
                dn = np.load(d)
                if str(dn["source"]) != base or int(dn["ok"]) != 1:
                    fail(f"{d} is not a passing denominator file for {base}")
                info["D_B1"], info["denom"] = np.asarray(dn["D_B1"], np.float64), d
                break
    return info


def ratio(g, s):
    return g / s if s > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--classes", type=int, default=0)
    ap.add_argument("--draw", type=int, required=True,
                    help="latents drawn by the rows: max(rng_images, n_images)")
    ap.add_argument("--k", type=int, default=32, help="images used (first k of the draw)")
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--rng-images", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-steps", type=int, default=200)
    ap.add_argument("--max-t", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--decode-chunk", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--denom-dir", default="runs1",
                    help="where search_denominator.py outputs for runs2 rows live")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.check_only and not args.out:
        ap.error("--out is required unless --check-only")
    k, T = args.k, args.max_t
    if k % args.batch or k % args.decode_chunk:
        fail("--k must be a multiple of --batch and --decode-chunk (same chunks as the runs)")
    rows = [row_info(p, args) for p in args.rows]
    print(f"REFERENCE FLOOR [{args.dataset}] draw={args.draw} k={k}  rows: "
          + ", ".join(f"{os.path.basename(r['path'])} ({r['sampler']})" for r in rows))
    for r in rows:
        if not r["v2"]:
            print(f"  {os.path.basename(r['path'])}: N+1 gate from "
                  f"{r.get('denom', 'NONE (no denominator file; N+1 not gated)')}")
    print("  file checks passed", flush=True)
    if args.check_only:
        return

    import torch
    import lpips
    from reward_structure import decode, lpips_to, resolve_paths, run_schedule
    from search_vs_sample_v2 import run_dpmpp2m, run_schedule_cond
    if args.classes > 0:  # as search_vs_sample_v2.py does before the adapter import
        from curvature_probe import _stub_clip_if_missing
        _stub_clip_if_missing()
    from adapter_compvis import create_compvis_adapter

    # setup identical to search_vs_sample.py / search_vs_sample_v2.py
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    cfg_path, ckpt_path = resolve_paths(args.dataset)
    print(f"  config {cfg_path}\n  ckpt   {ckpt_path}")
    adapter = create_compvis_adapter(cfg_path, ckpt_path, device=device,
                                     n_timesteps=args.ref_steps).to(device).eval()
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    is_cond = getattr(getattr(adapter, "model", None), "cond_stage_model", None) is not None
    if is_cond != (args.classes > 0):
        fail(f"model conditional={is_cond} but --classes {args.classes}")
    C = getattr(adapter, "latent_channels", 4)
    S = getattr(adapter, "latent_size", 32)
    m = args.draw
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    z_T = torch.randn(m, C, S, S, generator=g)[:k].to(device)
    labels = None
    if is_cond:
        lab_np = np.random.default_rng(args.seed).integers(0, args.classes, m)[:k]
        for r in rows:
            if r["v2"] and not np.array_equal(lab_np, np.asarray(r["src"]["labels"])[:k]):
                fail(f"{r['path']}: regenerated class labels differ")
        labels = torch.from_numpy(lab_np).long().to(device)
    dc, bs = args.decode_chunk, args.batch

    def lab(i0, i1):
        return None if labels is None else labels[i0:i1]

    def ddim(z, sched, lb):
        if lb is None:
            return run_schedule(adapter, z, sched, device)
        return run_schedule_cond(adapter, z, sched, device, lb)

    def batched_lat(sched, sampler):
        f = ddim if sampler == "ddim" else (lambda z, s, lb: run_dpmpp2m(adapter, z, s,
                                                                          device, lb))
        return torch.cat([f(z_T[i:i + bs], sched, lab(i, i + bs)) for i in range(0, k, bs)], 0)

    ref = np.linspace(T - 1, 0, args.ref_steps + 1).round().astype(np.int64)
    hi = np.arange(T - 1, -1, -1, dtype=np.int64)
    ref_img = decode(adapter, batched_lat(ref, "ddim"), dc)
    hi_img = decode(adapter, batched_lat(hi, "ddim"), dc)
    floor = np.asarray(lpips_to(loss_fn, ref_img, hi_img, dc), np.float64)
    print(f"  floor  LPIPS({args.ref_steps}-step DDIM, {len(hi) - 1}-step DDIM) over {k} images:"
          f"  mean {floor.mean():.6f}  median {np.median(floor):.6f}  max {floor.max():.6f}",
          flush=True)

    def score_batched(sched, sampler):
        img = decode(adapter, batched_lat(sched, sampler), dc)
        return (np.asarray(lpips_to(loss_fn, img, ref_img, dc), np.float64),
                np.asarray(lpips_to(loss_fn, img, hi_img, dc), np.float64))

    def score_one(sched, i, sampler):
        lb = lab(i, i + 1)
        z = (ddim(z_T[i:i + 1], sched, lb) if sampler == "ddim"
             else run_dpmpp2m(adapter, z_T[i:i + 1], sched, device, lb))
        img = decode(adapter, z, 1)
        return (float(lpips_to(loss_fn, img, ref_img[i:i + 1], 1)[0]),
                float(lpips_to(loss_fn, img, hi_img[i:i + 1], 1)[0]))

    results = []
    for r in rows:
        smp = r["sampler"]
        B = len(r["fB"]) - 1
        b200, bhi = score_batched(r["fB"], smp)
        b1_200, b1hi = score_batched(r["fB1"], smp)
        base = np.array([score_one(r["fB"], i, smp) for i in range(k)])
        own = np.array([score_one(r["per_sched"][i], i, smp) for i in range(k)])
        devs = {"batched shared N vs D_fixed": float(np.abs(b200 - r["D_fixed"][:k]).max()),
                "per-image own vs per_img": float(np.abs(own[:, 0] - r["per_img"][:k]).max())}
        if r["v2"]:
            devs["batched shared N+1 vs D_fixed_B1"] = float(
                np.abs(b1_200 - np.asarray(r["src"]["D_fixed_B1"], np.float64)[:k]).max())
            devs["per-image shared N vs D_fixed_unbatched"] = float(
                np.abs(base[:, 0] - np.asarray(r["src"]["D_fixed_unbatched"],
                                               np.float64)[:k]).max())
        elif "D_B1" in r:
            devs["batched shared N+1 vs denominator D_B1"] = float(
                np.abs(b1_200 - r["D_B1"][:k]).max())
        name = os.path.basename(r["path"])
        for key, v in devs.items():
            print(f"  reproduce  {name}  {key:40s} max |dev| {v:.3e}  "
                  f"{'PASS' if v <= args.tol else 'FAIL'}", flush=True)
        if not all(v <= args.tol for v in devs.values()):
            fail(f"{name}: stored values not reproduced; nothing saved")
        g200, ghi = float((base[:, 0] - own[:, 0]).mean()), float((base[:, 1] - own[:, 1]).mean())
        s200, shi = float((b200 - b1_200).mean()), float((bhi - b1hi).mean())
        dn1_full = float(r["D_fixed"].mean()) - r["step"]
        res = dict(name=name, sampler=smp, B=B, n=r["n"], dn1_full=dn1_full,
                   dn1_200=float(b1_200.mean()), dn1_hi=float(b1hi.mean()),
                   g200=g200, ghi=ghi, s200=s200, shi=shi,
                   R200=ratio(g200, s200), Rhi=ratio(ghi, shi),
                   R_full=ratio(r["gain"], r["step"]), maxdev=max(devs.values()),
                   b200=b200, bhi=bhi, b1_200=b1_200, b1hi=b1hi, base=base, own=own)
        results.append(res)
        print(f"  {name}: D(N+1) full {dn1_full:.6f} = {dn1_full / floor.mean():.1f}x floor, "
              f"subset {res['dn1_200']:.6f} = {res['dn1_200'] / floor.mean():.1f}x;"
              f"  subset R  {res['R200']:.3f} (200-step ref) vs {res['Rhi']:.3f} (999-step ref)",
              flush=True)

    print(f"\n  {'row':44s} {'smp':7s} {'D(N+1)/floor':>12s} {'(subset)':>8s} {'R full':>7s} "
          f"{'R sub/200':>9s} {'R sub/999':>9s}  gain 200->999      step 200->999")
    for x in results:
        print(f"  {x['name']:44s} {x['sampler']:7s} {x['dn1_full'] / floor.mean():12.1f} "
              f"{x['dn1_200'] / floor.mean():8.1f} "
              f"{x['R_full']:7.3f} {x['R200']:9.3f} {x['Rhi']:9.3f}  "
              f"{x['g200']:.5f}->{x['ghi']:.5f}  {x['s200']:.5f}->{x['shi']:.5f}")
    print()
    for x in results:
        print(f"SUMMARY refloor dataset={args.dataset} row={x['name']} sampler={x['sampler']} "
              f"B={x['B']} k={k} floor_mean={floor.mean():.6f} DN1_full={x['dn1_full']:.6f} "
              f"ratio={x['dn1_full'] / floor.mean():.1f} "
              f"ratio_sub={x['dn1_200'] / floor.mean():.1f} R_full={x['R_full']:.4f} "
              f"R_sub200={x['R200']:.4f} R_sub999={x['Rhi']:.4f} repro_max_dev={x['maxdev']:.3e}")
    save = dict(dataset=np.asarray(args.dataset), draw=np.int64(m), k=np.int64(k),
                ref_steps=np.int64(args.ref_steps), hi_steps=np.int64(len(hi) - 1),
                floor=floor, rows=np.asarray([x["name"] for x in results]))
    for j, x in enumerate(results):
        for key in ("b200", "bhi", "b1_200", "b1hi", "base", "own"):
            save[f"r{j}_{key}"] = x[key]
        for key in ("g200", "ghi", "s200", "shi", "R200", "Rhi", "R_full", "dn1_full",
                    "dn1_200", "dn1_hi"):
            save[f"r{j}_{key}"] = np.float64(x[key])
    np.savez(args.out, **save)
    print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
