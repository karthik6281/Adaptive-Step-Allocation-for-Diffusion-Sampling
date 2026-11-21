"""Shared-search check for one compvis row (church256, bedroom256, cin256; DDIM or DPM-Solver++).

The step value compares two separately searched shared schedules, at N and N+1. Each search is
coordinate descent and stops at a local optimum. If the N search stopped short of its best
schedule by d_N and the N+1 search by d_N1, the measured gain grows by about d_N and the step
value by d_N - d_N1, so R is off in a direction that depends on both. The SD 1.5 N=8 row showed
d_N > 0 (a per-prompt schedule beat the shared one on the mean). This applies the procedure of
sd_more.py --warm to a compvis row:

  1. menu: the own schedules of the first --menu-cands images, each scored as a shared schedule
     on all n images. If the best beats the stored shared N, the shared N search is run from
     it to convergence (the run's candidates, --search-sweeps cap).
  2. one node inserted into the better shared N at the midpoint of each interval; the shared
     N+1 search is run from the best insertion to convergence.
  3. gain (per-image path, as published), step value (batched path, as published) and R with
     the better of stored and new shared schedules at each budget. An image's per-image value
     is the better of its own searched schedule and the shared N schedule.

Gates, before any search (exit 1, nothing saved): on all n images the batched shared N
re-scores to the stored D_fixed (batched in both the frozen script and v2), the batched shared
N+1 to the stored step value (v2: also to D_fixed_B1), and the per-image path at the shared N
to the stored gain, each to --tol. The frozen script stores no settings, so they come from the
command line and these gates check them.
"""

import argparse
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def fail(msg):
    print(f"FAILED: {msg}", flush=True)
    sys.exit(1)


def ratio(g, s):
    return g / s if s > 0 else float("nan")


def insertions(s):
    """Every schedule with one extra node at the midpoint of an interval of s (gap >= 2)."""
    out = []
    for j in range(len(s) - 1):
        a, b = int(s[j]), int(s[j + 1])
        if a - b >= 2:
            out.append(np.array(list(s[:j + 1]) + [(a + b) // 2] + list(s[j + 1:]), np.int64))
    return out


def deletions(s):
    """Every schedule with one interior node of s removed."""
    return [np.array(list(s[:j]) + list(s[j + 1:]), np.int64) for j in range(1, len(s) - 1)]


def logged_uniform(log):
    """The uniform-schedule mean LPIPS the run printed (6 decimals), or None."""
    if not os.path.exists(log):
        return None
    for raw in open(log, errors="replace"):
        m = re.match(r"^\s*uniform\s+\[[^\]]*\]\s+mean LPIPS ([0-9.]+)", raw.split("\r")[-1])
        if m:
            return float(m.group(1))
    return None


def check_row(args):
    src = np.load(args.npz)
    per_img = np.asarray(src["per_img"], np.float64)
    n = len(per_img)
    v2 = "sampler" in src.files
    fB = np.asarray(src["fixed_B"]).astype(np.int64)
    fB1 = np.asarray(src["fixed_B1"]).astype(np.int64)
    per_sched = np.asarray(src["per_sched"]).astype(np.int64)
    if len(fB1) != len(fB) + 1 or per_sched.shape != (n, len(fB)):
        fail("unexpected array shapes")
    if np.asarray(src["D_fixed"]).shape != (n,):
        fail("D_fixed has the wrong shape")
    sampler = str(src["sampler"]) if v2 else "ddim"
    if v2:
        want = dict(dataset=args.dataset, classes=args.classes, seed=args.seed,
                    rng_images=args.rng_images, ref_steps=args.ref_steps, max_t=args.max_t,
                    n_images=n, budget=len(fB) - 1)
        if not args.test:
            want.update(sweeps=args.sweeps, n_candidates=args.n_cand)
        for key, v in want.items():
            got = src[key].item()
            if (str(got) if isinstance(v, str) else int(got)) != v:
                fail(f"npz {key}={got!r}, expected {v!r}")
        if sampler not in ("ddim", "dpmpp2m"):
            fail(f"unknown sampler {sampler}")
    elif args.classes != 0:
        fail("a frozen-script npz is unconditional; --classes must be 0")
    return dict(src=src, n=n, v2=v2, fB=fB, fB1=fB1, per_sched=per_sched, per_img=per_img,
                D_fixed=np.asarray(src["D_fixed"], np.float64), gain=float(src["gain"]),
                step=float(src["step_value"]), sampler=sampler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--classes", type=int, default=0)
    ap.add_argument("--sweeps", type=int, default=10,
                    help="the run's --sweeps (checked against v2 files)")
    ap.add_argument("--n-cand", type=int, default=12, help="the run's --n-cand")
    ap.add_argument("--search-sweeps", type=int, default=30,
                    help="sweep cap of the two new shared searches (they stop at convergence; "
                         "a capped one is reported with a WARNING)")
    ap.add_argument("--menu-cands", type=int, default=64)
    ap.add_argument("--rng-images", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-steps", type=int, default=200)
    ap.add_argument("--max-t", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--decode-chunk", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--test", action="store_true",
                    help="allow --sweeps/--n-cand that differ from the run (quick tests only)")
    ap.add_argument("--rescore", default=None,
                    help="an earlier shared_check output for this row: after the gates, only "
                         "re-score its best shared N and N+1 per image on the batched path "
                         "(for the bootstrap) and check them against its stored means")
    ap.add_argument("--round2", default=None,
                    help="an earlier (round-1) shared_check output for this row: search the "
                         "shared N again from the best one-node deletion of its best N+1, and, "
                         "if that finds a better N, the shared N+1 from the best insertion into it")
    ap.add_argument("--uniform-only", action="store_true",
                    help="after the gates, only score the uniform N schedule per image on the "
                         "batched path (checked against the run's log), for the value of tuning")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.check_only and not args.out:
        ap.error("--out is required unless --check-only")
    R = check_row(args)
    n, fB, fB1, B = R["n"], R["fB"], R["fB1"], len(R["fB"]) - 1
    smp, per_sched, per_img = R["sampler"], R["per_sched"], R["per_img"]
    name = os.path.basename(args.npz)
    print(f"SHARED-SEARCH CHECK [{args.dataset}] {name}  B={B} n={n} sampler={smp} "
          f"{'v2' if R['v2'] else 'frozen'} file  sweeps={args.sweeps} n_cand={args.n_cand}"
          f"{'  (TEST)' if args.test else ''}")
    print("  file checks passed", flush=True)
    if args.check_only:
        return

    import torch
    import lpips
    from reward_structure import decode, lpips_to, resolve_paths, run_schedule
    from search_vs_sample_v2 import coord_descent, run_dpmpp2m, run_schedule_cond, uniform
    if args.classes > 0:  # as search_vs_sample_v2.py does before the adapter import
        from curvature_probe import _stub_clip_if_missing
        _stub_clip_if_missing()
    from adapter_compvis import create_compvis_adapter

    # setup identical to search_vs_sample.py / search_vs_sample_v2.py
    t_start = time.time()
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    T = args.max_t
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
    m = max(args.rng_images, n)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    z_T = torch.randn(m, C, S, S, generator=g)[:n].to(device)
    labels = None
    if is_cond:
        lab_np = np.random.default_rng(args.seed).integers(0, args.classes, m)[:n]
        if R["v2"] and not np.array_equal(lab_np, np.asarray(R["src"]["labels"])):
            fail("regenerated class labels differ from the npz")
        labels = torch.from_numpy(lab_np).long().to(device)
    dc, bs = args.decode_chunk, args.batch

    def lab(i0, i1):
        return None if labels is None else labels[i0:i1]

    def ddim(z, sched, lb):
        if lb is None:
            return run_schedule(adapter, z, sched, device)
        return run_schedule_cond(adapter, z, sched, device, lb)

    def rollout(z, sched, lb):
        if smp == "ddim":
            return ddim(z, sched, lb)
        return run_dpmpp2m(adapter, z, sched, device, lb)

    ref = np.linspace(T - 1, 0, args.ref_steps + 1).round().astype(np.int64)
    ref_z = torch.cat([ddim(z_T[i:i + bs], ref, lab(i, i + bs)) for i in range(0, n, bs)], 0)
    ref_img = decode(adapter, ref_z, dc)
    print(f"  reference generated at {args.ref_steps} NFE ({time.time() - t_start:.0f}s)",
          flush=True)

    def lpips_all(sched):   # float32 array; its .mean() is the search objective
        zs = torch.cat([rollout(z_T[i:i + bs], sched, lab(i, i + bs)) for i in range(0, n, bs)],
                       dim=0)
        return lpips_to(loss_fn, decode(adapter, zs, dc), ref_img, dc)

    def obj(s):
        return float(lpips_all(s).mean())

    def lpips_one(sched, i):
        z = rollout(z_T[i:i + 1], sched, lab(i, i + 1))
        return float(lpips_to(loss_fn, decode(adapter, z, 1), ref_img[i:i + 1], 1)[0])

    def per_image(sched):
        return np.array([lpips_one(sched, i) for i in range(n)])

    # ---- gates ----
    D_fB, D_fB1 = lpips_all(fB), lpips_all(fB1)
    fB_mean, fB1_mean = float(D_fB.mean()), float(D_fB1.mean())
    base_u = per_image(fB)
    devs = {"batched shared N vs stored D_fixed":
                float(np.abs(np.asarray(D_fB, np.float64) - R["D_fixed"]).max()),
            "batched N - N+1 vs stored step value": abs((fB_mean - fB1_mean) - R["step"]),
            "per-image path at shared N vs stored gain":
                abs(float(base_u.mean() - per_img.mean()) - R["gain"])}
    if R["v2"]:
        devs["batched shared N+1 vs stored D_fixed_B1"] = float(np.abs(
            np.asarray(D_fB1, np.float64) - np.asarray(R["src"]["D_fixed_B1"], np.float64)).max())
        devs["per-image shared N vs stored D_fixed_unbatched"] = float(np.abs(
            base_u - np.asarray(R["src"]["D_fixed_unbatched"], np.float64)).max())
    for key, v in devs.items():
        print(f"  reproduce  {key:48s} |dev| {v:.3e}  {'PASS' if v <= args.tol else 'FAIL'}")
    if not all(v <= args.tol for v in devs.values()):
        fail("stored values not reproduced; nothing saved")
    print(f"  stored shared N {fB_mean:.6f}   stored shared N+1 {fB1_mean:.6f}", flush=True)

    if args.uniform_only:
        u = uniform(T, B)
        D_u = lpips_all(u)
        logged = logged_uniform(args.npz[:-4] + ".txt")
        if logged is None:
            fail("no uniform line in the run's log to check against")
        du = abs(float(D_u.mean()) - logged)
        print(f"  reproduce  {'uniform N mean vs the log (6 decimals)':48s} |dev| {du:.3e}  "
              f"{'PASS' if du <= 5.1e-7 else 'FAIL'}")
        if du > 5.1e-7:
            fail("the uniform schedule does not re-score to the logged value")
        np.savez(args.out, source=np.asarray(args.npz), dataset=np.asarray(args.dataset),
                 sampler=np.asarray(smp), budget=np.int64(B), n_images=np.int64(n),
                 uniform_B=u, uniform_mean=np.float64(float(D_u.mean())),
                 logged_uniform_mean=np.float64(logged),
                 D_uniform_batched=np.asarray(D_u, np.float64),
                 repro_keys=np.asarray(list(devs) + ["uniform vs log"]),
                 repro_devs=np.asarray(list(devs.values()) + [du]), tol=np.float64(args.tol))
        print(f"\nSUMMARY shared_uniform dataset={args.dataset} row={name} sampler={smp} B={B} "
              f"n={n} uniform={float(D_u.mean()):.6f} stored_N={fB_mean:.6f} "
              f"repro_max_dev={max(devs.values()):.3e}")
        print(f"  saved {args.out}  (wall {time.time() - t_start:.0f}s)")
        return

    if args.round2:
        prior = np.load(args.round2)
        if os.path.basename(str(prior["source"])) != name or int(prior["budget"]) != B \
                or int(prior["n_images"]) != n or int(prior["test"]) != 0:
            fail(f"{args.round2} is not a (non-test) shared_check output for {name}")
        pN = np.asarray(prior["best_B"]).astype(np.int64)
        pN1 = np.asarray(prior["best_B1"]).astype(np.int64)
        pN_mean, pN1_mean = float(prior["best_B_mean"]), float(prior["best_B1_mean"])
        D_pN, D_pN1 = lpips_all(pN), lpips_all(pN1)
        rdev = {"round-1 best shared N mean vs stored": abs(float(D_pN.mean()) - pN_mean),
                "round-1 best shared N+1 mean vs stored": abs(float(D_pN1.mean()) - pN1_mean)}
        for key, v in rdev.items():
            print(f"  reproduce  {key:48s} |dev| {v:.3e}  {'PASS' if v <= args.tol else 'FAIL'}")
        if not all(v <= args.tol for v in rdev.values()):
            fail("the round-1 shared schedules do not re-score to their values")
        devs.update(rdev)
        # N from the best one-node deletion of the round-1 N+1
        t0 = time.time()
        dels = deletions(pN1)
        del_vals = [obj(q) for q in dels]
        k0 = int(np.argmin(del_vals))
        fD, fD_mean, _, eD, cD = coord_descent(obj, dels[k0], args.search_sweeps, args.n_cand)
        print(f"\n  round 2, shared N from the best deletion of the round-1 N+1 "
              f"({del_vals[k0]:.6f}): -> {fD_mean:.6f} {fD.tolist()}  ({eD} evals, "
              f"{'converged' if cD else 'HIT SWEEP CAP'}, {time.time() - t0:.0f}s)")
        print(f"  round-1 shared N {pN_mean:.6f}: round 2 "
              f"{'BEATS' if fD_mean < pN_mean - 1e-9 else 'does not beat'} it by "
              f"{pN_mean - fD_mean:+.6f}", flush=True)
        newN = fD_mean < pN_mean - 1e-9
        bestN, bestN_mean = (fD, fD_mean) if newN else (pN, pN_mean)
        fI, fI_mean, eI, cI = pN1, pN1_mean, 0, True
        if newN:   # otherwise the insertion start is round 1's own, and would repeat it exactly
            t0 = time.time()
            ins = insertions(bestN)
            ins_vals = [obj(q) for q in ins]
            j0 = int(np.argmin(ins_vals))
            fI, fI_mean, _, eI, cI = coord_descent(obj, ins[j0], args.search_sweeps, args.n_cand)
            print(f"  round 2, shared N+1 from the best insertion ({ins_vals[j0]:.6f}): -> "
                  f"{fI_mean:.6f} {fI.tolist()}  ({eI} evals, "
                  f"{'converged' if cI else 'HIT SWEEP CAP'}, {time.time() - t0:.0f}s)")
        else:
            print("  shared N unchanged, so the N+1 insertion start equals round 1's: not repeated")
        newN1 = fI_mean < pN1_mean - 1e-9
        bestN1, bestN1_mean = (fI, fI_mean) if newN1 else (pN1, pN1_mean)
        print(f"  round-1 shared N+1 {pN1_mean:.6f}: round 2 "
              f"{'BEATS' if newN1 else 'does not beat'} it by {pN1_mean - fI_mean:+.6f}",
              flush=True)
        D_N_b = lpips_all(bestN) if newN else D_pN
        D_N1_b = lpips_all(bestN1) if newN1 else D_pN1
        for label, D, want in (("N", D_N_b, bestN_mean), ("N+1", D_N1_b, bestN1_mean)):
            v = abs(float(D.mean()) - want)
            devs[f"re-score of round-2 best shared {label} vs its search value"] = v
            if v > args.tol:
                print(f"  WARNING: re-score of the round-2 best shared {label} differs by {v:.3e}")
        b1 = np.asarray(prior["D_best_B_unbatched"], np.float64)
        base_new = per_image(bestN) if newN else b1
        per_img2 = np.minimum(per_img, base_new)
        g1 = float(b1.mean() - np.minimum(per_img, b1).mean())
        g2 = float(base_new.mean() - per_img2.mean())
        rows = {"stored (as published)": (R["gain"], R["step"]),
                "round 1": (g1, pN_mean - pN1_mean),
                "round 2": (g2, bestN_mean - bestN1_mean)}
        print(f"\n  {'':24s} {'gain':>9s} {'step':>9s} {'R':>7s} {'R/N':>6s}")
        for key, (gg, ss) in rows.items():
            print(f"  {key:24s} {gg:9.6f} {ss:9.6f} {ratio(gg, ss):7.3f} "
                  f"{100 * ratio(gg, ss) / B:5.1f}%")
        for label, c in (("shared N", cD), ("shared N+1", cI)):
            if not c:
                print(f"  WARNING: the round-2 {label} search hit the {args.search_sweeps}-sweep "
                      f"cap; its improvement is a lower bound")
        print(f"\nSUMMARY shared_round2 dataset={args.dataset} row={name} sampler={smp} B={B} "
              f"n={n} round1_N={pN_mean:.6f} round2_N={bestN_mean:.6f} "
              f"d_N={pN_mean - bestN_mean:.6f} round1_N1={pN1_mean:.6f} "
              f"round2_N1={bestN1_mean:.6f} d_N1={pN1_mean - bestN1_mean:.6f} "
              f"conv_N={int(cD)} conv_N1={int(cI)} R_round1={ratio(*rows['round 1']):.4f} "
              f"R_round2={ratio(*rows['round 2']):.4f} repro_max_dev={max(devs.values()):.3e}"
              f"{' TEST' if args.test else ''}")
        np.savez(args.out, source=np.asarray(args.npz), round2_of=np.asarray(args.round2),
                 dataset=np.asarray(args.dataset), sampler=np.asarray(smp), budget=np.int64(B),
                 n_images=np.int64(n), test=np.int8(args.test),
                 search_sweeps=np.int64(args.search_sweeps), deletions=np.stack(dels),
                 deletion_means=np.asarray(del_vals), round2_B=fD,
                 round2_B_mean=np.float64(fD_mean), round2_B_conv=np.int8(cD),
                 round2_B1=fI, round2_B1_mean=np.float64(fI_mean), round2_B1_conv=np.int8(cI),
                 best_B=bestN, best_B_mean=np.float64(bestN_mean), best_B1=bestN1,
                 best_B1_mean=np.float64(bestN1_mean), D_best_B_unbatched=base_new,
                 per_img_best=per_img2, D_best_B_batched=np.asarray(D_N_b, np.float64),
                 D_best_B1_batched=np.asarray(D_N1_b, np.float64),
                 row_names=np.asarray(list(rows)), row_gain_step=np.asarray(list(rows.values())),
                 repro_keys=np.asarray(list(devs)), repro_devs=np.asarray(list(devs.values())),
                 tol=np.float64(args.tol))
        print(f"  saved {args.out}  (wall {time.time() - t_start:.0f}s)")
        return

    if args.rescore:
        prior = np.load(args.rescore)
        if os.path.basename(str(prior["source"])) != name or int(prior["budget"]) != B \
                or int(prior["n_images"]) != n or int(prior["test"]) != 0:
            fail(f"{args.rescore} is not a (non-test) shared_check output for {name}")
        bN = np.asarray(prior["best_B"]).astype(np.int64)
        bN1 = np.asarray(prior["best_B1"]).astype(np.int64)
        D_N_b, D_N1_b = lpips_all(bN), lpips_all(bN1)
        rdev = {"best shared N mean vs stored": abs(float(D_N_b.mean())
                                                    - float(prior["best_B_mean"])),
                "best shared N+1 mean vs stored": abs(float(D_N1_b.mean())
                                                      - float(prior["best_B1_mean"]))}
        for key, v in rdev.items():
            print(f"  reproduce  {key:48s} |dev| {v:.3e}  {'PASS' if v <= args.tol else 'FAIL'}")
        if not all(v <= args.tol for v in rdev.values()):
            fail("the earlier check's shared schedules do not re-score to its values")
        np.savez(args.out, source=np.asarray(args.npz), rescore_of=np.asarray(args.rescore),
                 dataset=np.asarray(args.dataset), sampler=np.asarray(smp), budget=np.int64(B),
                 n_images=np.int64(n), best_B=bN, best_B1=bN1,
                 best_B_mean=np.float64(float(D_N_b.mean())),
                 best_B1_mean=np.float64(float(D_N1_b.mean())),
                 D_best_B_batched=np.asarray(D_N_b, np.float64),
                 D_best_B1_batched=np.asarray(D_N1_b, np.float64),
                 D_best_B_unbatched=np.asarray(prior["D_best_B_unbatched"], np.float64),
                 per_img_best=np.asarray(prior["per_img_best"], np.float64),
                 repro_keys=np.asarray(list(devs) + list(rdev)),
                 repro_devs=np.asarray(list(devs.values()) + list(rdev.values())),
                 tol=np.float64(args.tol))
        print(f"\nSUMMARY shared_rescore dataset={args.dataset} row={name} sampler={smp} B={B} "
              f"n={n} best_N={float(D_N_b.mean()):.6f} best_N1={float(D_N1_b.mean()):.6f} "
              f"repro_max_dev={max(list(devs.values()) + list(rdev.values())):.3e}")
        print(f"  saved {args.out}  (wall {time.time() - t_start:.0f}s)")
        return

    # ---- 1. menu and the shared N search from the best entry ----
    t0 = time.time()
    nc = min(n, args.menu_cands)
    menu = np.array([obj(per_sched[j]) for j in range(nc)])
    j_best = int(np.argmin(menu))
    print(f"  menu: {nc} own schedules scored as shared; best is image {j_best}'s, "
          f"{menu[j_best]:.6f} vs stored {fB_mean:.6f} ({time.time() - t0:.0f}s)", flush=True)
    fP, fP_mean, eP, cP = fB, fB_mean, 0, True
    if menu[j_best] < fB_mean - 1e-9:
        t0 = time.time()
        fP, fP_mean, fP_start, eP, cP = coord_descent(obj, per_sched[j_best],
                                                      args.search_sweeps, args.n_cand)
        if fP_start != float(menu[j_best]):
            fail(f"NON-DETERMINISTIC: start {fP_start!r} != menu {float(menu[j_best])!r}")
        print(f"  shared N from image {j_best}'s schedule: {fP_start:.6f} -> {fP_mean:.6f}  "
              f"{fP.tolist()}  ({eP} evals, {'converged' if cP else 'HIT SWEEP CAP'}, "
              f"{time.time() - t0:.0f}s)", flush=True)
    else:
        print("  no menu entry beats the stored shared N; it is kept")
    if fP_mean < fB_mean - 1e-9:
        bestN, bestN_mean, src_N = fP, fP_mean, "new"
    else:
        bestN, bestN_mean, src_N = fB, fB_mean, "stored"

    # ---- 2. shared N+1 from the best one-node insertion ----
    t0 = time.time()
    ins = insertions(bestN)
    ins_vals = [obj(s) for s in ins]
    k0 = int(np.argmin(ins_vals))
    print(f"\n  N+1 insertions into the {src_N} shared N: {len(ins)} tried, best "
          f"{ins_vals[k0]:.6f} {ins[k0].tolist()}", flush=True)
    fW, fW_mean, _, eW, cW = coord_descent(obj, ins[k0], args.search_sweeps, args.n_cand)
    print(f"  shared N+1 warm start: {ins_vals[k0]:.6f} -> {fW_mean:.6f}  {fW.tolist()}  "
          f"({eW} evals, {'converged' if cW else 'HIT SWEEP CAP'}, {time.time() - t0:.0f}s)")
    print(f"  stored shared N+1 {fB1_mean:.6f}   warm start "
          f"{'BEATS' if fW_mean < fB1_mean - 1e-9 else 'does not beat'} it by "
          f"{fB1_mean - fW_mean:+.6f}", flush=True)
    bestN1, bestN1_mean = (fW, fW_mean) if fW_mean < fB1_mean - 1e-9 else (fB1, fB1_mean)

    # ---- 3. gain, step value and R ----
    # per-image batched scores at the best shared N and N+1, for the bootstrap
    D_N_b, D_N1_b = lpips_all(bestN), lpips_all(bestN1)
    for label, D, want in (("N", D_N_b, bestN_mean), ("N+1", D_N1_b, bestN1_mean)):
        v = abs(float(D.mean()) - want)
        devs[f"re-score of best shared {label} vs its search value"] = v
        if v > args.tol:   # recorded, not fatal: the searches above are still valid
            print(f"  WARNING: re-score of the best shared {label} differs from its search "
                  f"value by {v:.3e}")
    base_new = per_image(bestN) if src_N == "new" else base_u
    per_img2 = np.minimum(per_img, base_new)
    gain_new = float(base_new.mean() - per_img2.mean())
    rows = {"stored (as published)": (R["gain"], R["step"]),
            "better shared N, stored N+1": (gain_new, bestN_mean - fB1_mean),
            "better shared N and N+1 (primary)": (gain_new, bestN_mean - bestN1_mean)}
    print(f"\n  {'':40s} {'gain':>9s} {'step':>9s} {'R':>7s} {'R/N':>6s}")
    for key, (gg, ss) in rows.items():
        print(f"  {key:40s} {gg:9.6f} {ss:9.6f} {ratio(gg, ss):7.3f} "
              f"{100 * ratio(gg, ss) / B:5.1f}%")
    g_p, s_p = rows["better shared N and N+1 (primary)"]
    for label, c in (("shared N", cP), ("shared N+1", cW)):
        if not c:
            print(f"  WARNING: the new {label} search hit the {args.search_sweeps}-sweep cap; "
                  f"its improvement is a lower bound")
    print(f"\nSUMMARY shared_check dataset={args.dataset} row={name} sampler={smp} B={B} n={n} "
          f"stored_N={fB_mean:.6f} new_N={fP_mean:.6f} d_N={fB_mean - bestN_mean:.6f} "
          f"stored_N1={fB1_mean:.6f} new_N1={fW_mean:.6f} d_N1={fB1_mean - bestN1_mean:.6f} "
          f"conv_N={int(cP)} conv_N1={int(cW)} R_stored={ratio(R['gain'], R['step']):.4f} "
          f"R_primary={ratio(g_p, s_p):.4f} repro_max_dev={max(devs.values()):.3e}"
          f"{' TEST' if args.test else ''}")
    np.savez(args.out, source=np.asarray(args.npz), dataset=np.asarray(args.dataset),
             sampler=np.asarray(smp), budget=np.int64(B), n_images=np.int64(n),
             sweeps=np.int64(args.sweeps), n_candidates=np.int64(args.n_cand),
             search_sweeps=np.int64(args.search_sweeps),
             test=np.int8(args.test), menu_cands=np.int64(nc), menu_means=menu,
             menu_best=np.int64(j_best), polish_B=fP, polish_B_mean=np.float64(fP_mean),
             polish_B_evals=np.int64(eP), polish_B_conv=np.int8(cP), best_B=bestN,
             best_B_mean=np.float64(bestN_mean), insertions=np.stack(ins),
             insertion_means=np.asarray(ins_vals), warm_B1=fW,
             warm_B1_mean=np.float64(fW_mean), warm_B1_evals=np.int64(eW),
             warm_B1_conv=np.int8(cW), best_B1=bestN1, best_B1_mean=np.float64(bestN1_mean),
             D_best_B_unbatched=base_new, per_img_best=per_img2,
             D_best_B_batched=np.asarray(D_N_b, np.float64),
             D_best_B1_batched=np.asarray(D_N1_b, np.float64),
             row_names=np.asarray(list(rows)), row_gain_step=np.asarray(list(rows.values())),
             repro_keys=np.asarray(list(devs)), repro_devs=np.asarray(list(devs.values())),
             tol=np.float64(args.tol))
    print(f"  saved {args.out}  (wall {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
