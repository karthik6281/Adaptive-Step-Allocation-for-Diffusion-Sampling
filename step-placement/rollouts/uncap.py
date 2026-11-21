"""Continue the per-image searches that hit the sweep cap in one finished search run.

search_vs_sample.py and search_vs_sample_v2.py mark an image CAPPED when its coordinate
descent was still improving at --sweeps. coord_descent keeps no state besides the current
schedule, and generation and LPIPS are deterministic, so continuing from the stored
schedule for K more sweeps is exactly what --sweeps (old + K) would have done for that
image. This re-runs only the capped images and recomputes the gain:

    gain_new = gain_stored + sum over capped i of (per_img_old[i] - per_img_new[i]) / n

The per-image baseline (the image's score at the shared N schedule) and step_value do not
change: the shared searches are untouched (both converged in every run here).

Gates, before any search, for every capped image:
  * the stored per-image schedule must re-score to the stored per_img (to --tol), and the
    continued search must start from exactly that value;
  * the shared N schedule must re-score to the image's baseline: D_fixed_unbatched (to
    --tol) in a v2 file; the logged baseline (to its 6 printed decimals) in a frozen-script
    file, whose npz does not store the unbatched baseline.
Any failure exits 1 without saving.

Inputs: the run's .npz and its log (.txt). The log is the only record of WHICH images were
capped. The frozen script's npz stores no settings, so they come from the command line and
are checked against the log (and against the npz for v2 files). DDIM runs only.

  python3 -u utils/uncap.py --npz X.npz --log X.txt --dataset church256 --out Y.npz
  add --classes 1000 for cin256; --check-only does the file checks without a GPU.
"""

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

IMG = re.compile(r"^\s*image\s+(\d+)\s+(-?\d+\.\d+) -> (-?\d+\.\d+)\s+\((.*)\)\s*$")
SUM = re.compile(r"^SUMMARY search_vs_sample (.*)$")
LOG_ROUND = 5.1e-7  # the log prints 6 decimals


def fail(msg):
    print(f"FAILED: {msg}", flush=True)
    sys.exit(1)


def parse_log(path):
    """{image index: (baseline, final, capped, evals)} and the SUMMARY key=value dict."""
    rows, summary = {}, None
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n").split("\r")[-1]
            m = IMG.match(line)
            if m:
                i = int(m.group(1))
                if i in rows:
                    fail(f"{path}: image {i} appears twice")
                ev = re.search(r"(\d+) evals", m.group(4))
                rows[i] = (float(m.group(2)), float(m.group(3)), "CAPPED" in m.group(4),
                           int(ev.group(1)) if ev else -1)
                continue
            m = SUM.match(line)
            if m:
                if summary is not None:
                    fail(f"{path}: more than one SUMMARY line")
                summary = dict(kv.split("=", 1) for kv in m.group(1).split())
    if summary is None:
        fail(f"{path}: no SUMMARY line (unfinished run?)")
    return rows, summary


def check_files(args):
    rows, summ = parse_log(args.log)
    src = np.load(args.npz)
    per_img = np.asarray(src["per_img"], dtype=np.float64)
    per_sched = np.asarray(src["per_sched"]).astype(np.int64)
    fB = np.asarray(src["fixed_B"]).astype(np.int64)
    n, B, T = len(per_img), len(fB) - 1, args.max_t
    gain, step = float(src["gain"]), float(src["step_value"])
    v2 = "sampler" in src.files

    if per_sched.shape != (n, B + 1):
        fail(f"per_sched shape {per_sched.shape} != ({n}, {B + 1})")
    if fB[0] != T - 1 or fB[-1] != 0 or np.any(per_sched[:, 0] != T - 1) \
            or np.any(per_sched[:, -1] != 0):
        fail("schedules do not run from T-1 to 0")
    if sorted(rows) != list(range(n)):
        fail(f"log has {len(rows)} image lines, npz has {n} images")
    if summ.get("dataset") != args.dataset or int(summ["budget"]) != B \
            or int(summ["n"]) != n:
        fail(f"log SUMMARY {summ.get('dataset')} B={summ.get('budget')} n={summ.get('n')} "
             f"!= {args.dataset} B={B} n={n}")
    logged_b = np.array([rows[i][0] for i in range(n)])
    logged_v = np.array([rows[i][1] for i in range(n)])
    if float(np.abs(logged_v - per_img).max()) > LOG_ROUND:
        fail("the log's per-image results do not match the npz (not the same run)")
    if abs(float(summ["searched_gain"]) - gain) > LOG_ROUND:
        fail(f"log gain {summ['searched_gain']} != npz gain {gain:.6f}")
    if abs(float(summ["step_value"]) - step) > LOG_ROUND:
        fail(f"log step_value {summ['step_value']} != npz step_value {step:.6f}")
    capped = [i for i in range(n) if rows[i][2]]
    if int(summ["n_capped"]) != len(capped):
        fail(f"SUMMARY n_capped={summ['n_capped']} but {len(capped)} CAPPED image lines")
    if int(summ["conv_B"]) != 1 or int(summ["conv_B1"]) != 1:
        fail("a shared search did not converge; continuing per-image searches is not enough")
    if abs(float(logged_b.mean() - logged_v.mean()) - gain) > 2 * LOG_ROUND:
        fail("logged baselines do not reproduce the stored gain")

    if v2:
        want = dict(dataset=args.dataset, budget=B, n_images=n, sweeps=args.orig_sweeps,
                    n_candidates=args.n_cand, seed=args.seed, rng_images=args.rng_images,
                    ref_steps=args.ref_steps, max_t=T, sampler="ddim", classes=args.classes)
        for k, v in want.items():
            got = src[k].item() if src[k].shape == () else src[k]
            got = str(got) if isinstance(v, str) else int(got)
            if got != v:
                fail(f"npz {k}={got!r} but this run expects {v!r}")
        if int(src["n_capped"]) != len(capped):
            fail(f"npz n_capped={int(src['n_capped'])} != {len(capped)} in the log")
        base = np.asarray(src["D_fixed_unbatched"], dtype=np.float64)
        if abs(float(base.mean() - per_img.mean()) - gain) > 1e-9:
            fail("D_fixed_unbatched and per_img do not give the stored gain")
        if float(np.abs(base - logged_b).max()) > LOG_ROUND:
            fail("D_fixed_unbatched does not match the logged baselines")
        if summ.get("sampler") != "ddim" or summ.get("classes") != str(args.classes):
            fail(f"log sampler={summ.get('sampler')} classes={summ.get('classes')}")
    else:
        if args.classes != 0:
            fail("a frozen-script npz is unconditional; --classes must be 0")
        if "sampler" in summ:
            fail("the log is from search_vs_sample_v2.py but the npz is not")
        base = None
    return dict(src=src, rows=rows, per_img=per_img, per_sched=per_sched, fB=fB, n=n, B=B,
                gain=gain, step=step, v2=v2, base=base, logged_b=logged_b, capped=capped)


def updated_gain(gain, per_img_old, per_img_new):
    n = len(per_img_old)
    return gain + float((per_img_old - per_img_new).sum()) / n


def ratio(g, s):
    return g / s if s > 0 else float("nan")


def summary_line(args, F, still, dev, gain_new):
    return (f"SUMMARY uncap source={os.path.basename(args.npz)} dataset={args.dataset} "
            f"budget={F['B']} n={F['n']} capped={len(F['capped'])} still_capped={still} "
            f"sweeps={args.orig_sweeps}+{args.extra_sweeps} repro_max_dev={dev:.3e} "
            f"gain_old={F['gain']:.6f} gain_new={gain_new:.6f} step_value={F['step']:.6f} "
            f"R_old={ratio(F['gain'], F['step']):.4f} R_new={ratio(gain_new, F['step']):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--classes", type=int, default=0)
    ap.add_argument("--orig-sweeps", type=int, default=10)
    ap.add_argument("--extra-sweeps", type=int, default=30)
    ap.add_argument("--n-cand", type=int, default=12)
    ap.add_argument("--rng-images", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-steps", type=int, default=200)
    ap.add_argument("--max-t", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--decode-chunk", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.check_only and not args.out:
        ap.error("--out is required unless --check-only")

    F = check_files(args)
    n, B, capped = F["n"], F["B"], F["capped"]
    print(f"UNCAP [{args.dataset}] B={B} n={n} {'v2' if F['v2'] else 'frozen'} file  "
          f"{args.npz}")
    print(f"  file checks passed; capped images: {capped if capped else 'none'}", flush=True)
    if args.check_only:
        return
    per_img, per_sched = F["per_img"], F["per_sched"]
    new_img, new_sched = per_img.copy(), per_sched.copy()
    conv = np.ones(len(capped), dtype=np.int8)
    evals = np.zeros(len(capped), dtype=np.int64)
    dev_v = np.zeros(len(capped))
    dev_b = np.zeros(len(capped))

    if capped:
        import torch
        import lpips
        from reward_structure import decode, lpips_to, resolve_paths, run_schedule
        from search_vs_sample_v2 import coord_descent, run_schedule_cond
        if args.classes > 0:  # search_vs_sample_v2.py does this before the adapter import
            from curvature_probe import _stub_clip_if_missing
            _stub_clip_if_missing()
        from adapter_compvis import create_compvis_adapter

        # identical to the setup in search_vs_sample.py / search_vs_sample_v2.py
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
            rng = np.random.default_rng(args.seed)
            lab_np = rng.integers(0, args.classes, m)[:n]
            if F["v2"] and not np.array_equal(lab_np, np.asarray(F["src"]["labels"])):
                fail("regenerated class labels differ from the npz")
            labels = torch.from_numpy(lab_np).long().to(device)

        def lab(i0, i1):
            return None if labels is None else labels[i0:i1]

        def ddim(z, sched, lb):
            if lb is None:
                return run_schedule(adapter, z, sched, device)
            return run_schedule_cond(adapter, z, sched, device, lb)

        ref = np.linspace(T - 1, 0, args.ref_steps + 1).round().astype(np.int64)
        ref_z = torch.cat([ddim(z_T[i:i + args.batch], ref, lab(i, i + args.batch))
                           for i in range(0, n, args.batch)], dim=0)
        ref_img = decode(adapter, ref_z, args.decode_chunk)
        print(f"  reference generated at {args.ref_steps} NFE", flush=True)

        def lpips_one(sched, i):
            z = ddim(z_T[i:i + 1], sched, lab(i, i + 1))
            return float(lpips_to(loss_fn, decode(adapter, z, 1), ref_img[i:i + 1], 1)[0])

        start_v = {}
        for k, i in enumerate(capped):
            b = lpips_one(F["fB"], i)
            v = lpips_one(per_sched[i], i)
            want_b, tol_b = ((float(F["base"][i]), args.tol) if F["v2"]
                             else (float(F["logged_b"][i]), LOG_ROUND))
            dev_b[k], dev_v[k] = abs(b - want_b), abs(v - float(per_img[i]))
            ok = dev_b[k] <= tol_b and dev_v[k] <= args.tol
            print(f"  reproduce image {i:3d}  shared N |dev| {dev_b[k]:.3e}   "
                  f"own schedule |dev| {dev_v[k]:.3e}  {'PASS' if ok else 'FAIL'}", flush=True)
            if not ok:
                fail(f"image {i} does not reproduce; this environment differs from the run")
            start_v[i] = v

        for k, i in enumerate(capped):
            s_i, v_i, st, e_i, c_i = coord_descent(lambda s, i=i: lpips_one(s, i),
                                                   per_sched[i], args.extra_sweeps,
                                                   args.n_cand)
            if st != start_v[i]:
                fail(f"image {i}: NON-DETERMINISTIC start {st!r} != {start_v[i]!r}")
            if v_i > per_img[i] + args.tol:
                fail(f"image {i}: continued search got worse ({v_i} > {per_img[i]})")
            new_img[i], new_sched[i] = v_i, s_i
            conv[k], evals[k] = int(c_i), e_i
            print(f"    image {i:3d}  {per_img[i]:.6f} -> {v_i:.6f}  "
                  f"({100 * (per_img[i] - v_i) / max(per_img[i], 1e-12):+5.2f}% more, "
                  f"{e_i} evals, {'converged' if c_i else 'STILL CAPPED'})", flush=True)

    gain_new = updated_gain(F["gain"], per_img, new_img)
    if F["v2"]:
        direct = float(F["base"].mean() - new_img.mean())
        if abs(direct - gain_new) > 1e-9:
            fail(f"gain bookkeeping: {gain_new!r} vs {direct!r}")
    still = int((conv == 0).sum())
    dev = float(max(dev_v.max(initial=0.0), dev_b.max(initial=0.0) if F["v2"] else 0.0))
    print(f"\n  gain {F['gain']:.6f} -> {gain_new:.6f}   step value {F['step']:.6f}   "
          f"R {ratio(F['gain'], F['step']):.4f} -> {ratio(gain_new, F['step']):.4f}")
    if still:
        print(f"  WARNING: {still} image(s) still capped after {args.extra_sweeps} more sweeps")
    print("\n" + summary_line(args, F, still, dev, gain_new))
    np.savez(args.out, source=np.asarray(os.path.basename(args.npz)),
             dataset=np.asarray(args.dataset), budget=np.int64(B), n_images=np.int64(n),
             classes=np.int64(args.classes), orig_sweeps=np.int64(args.orig_sweeps),
             extra_sweeps=np.int64(args.extra_sweeps), n_candidates=np.int64(args.n_cand),
             capped_idx=np.asarray(capped, dtype=np.int64), conv_new=conv, evals_new=evals,
             repro_dev_own=dev_v, repro_dev_shared=dev_b,
             per_img_old=per_img, per_img_new=new_img,
             per_sched_old=per_sched, per_sched_new=new_sched,
             gain_old=F["gain"], gain_new=gain_new, step_value=F["step"],
             R_old=ratio(F["gain"], F["step"]), R_new=ratio(gain_new, F["step"]),
             still_capped=np.int64(still))
    print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
