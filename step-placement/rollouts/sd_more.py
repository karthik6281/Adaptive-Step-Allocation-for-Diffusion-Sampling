"""Last two SD 1.5 checks on a finished search_sd.py run. The source npz files are only read.

  --warm   the N+1 warm-start check (menu from --sdpost, else computed here). The schedule
           menu (sd_checks.py --render) found a
           per-prompt schedule that beats the stored shared N schedule on the mean over all
           prompts, so the shared search stopped at a local optimum. That lowers the gain, and
           whether R falls or rises depends on whether the shared N+1 search left as much on
           the table. This
             1. starts the shared N search from the best menu column and runs it to
                convergence (run's candidates, --search-sweeps cap): the improved shared N;
             2. inserts one node into that schedule at the midpoint of each interval, starts
                from the best insertion, and runs the shared N+1 search to convergence;
             3. recomputes gain, step value and R with the better of stored and new shared
                schedules at each budget, on the batched path (primary) and on the
                per-prompt path. A prompt's per-prompt value is the better of its own searched
                schedule and the shared N schedule.
  --floor  the reference floor: on the first --k prompts, LPIPS between the run's 200-step
           guided reference and guided DDIM over every timestep (999 steps), and the run's
           stored schedules re-scored against both references (gain, step value and R on
           that subset under each). Pass more runs with the same guidance via --also.

Setup (prompts, noise, model, fp16/fp32 split, classic attention, guided DDIM step, decode,
reference, LPIPS, determinism flags) is copied from sd_checks.py, which reproduced the
stored LPIPS of the SD N=8 run exactly. Gates: stored values must be reproduced to --tol
before anything is computed; otherwise exit 1 and nothing is saved.
"""

import argparse
import os
import re
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from reward_structure import lpips_to  # noqa: E402
from search_sd import Roller, coco_prompts  # noqa: E402
from search_vs_sample_v2 import coord_descent, uniform  # noqa: E402


def fail(msg):
    print(f"FAILED: {msg}", flush=True)
    sys.exit(1)


def ratio(g, s):
    return g / s if s > 0 else float("nan")


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


def insertions(s):
    """Every schedule with one extra node at the midpoint of an interval of s (gap >= 2)."""
    out = []
    for j in range(len(s) - 1):
        a, b = int(s[j]), int(s[j + 1])
        if a - b >= 2:
            out.append(np.array(list(s[:j + 1]) + [(a + b) // 2] + list(s[j + 1:]), np.int64))
    return out


def settings(src):
    return dict(B=int(src["budget"]), n=int(src["n_images"]), T=int(src["max_t"]),
                cfg=float(src["cfg"]), seed=int(src["seed"]),
                rng_prompts=int(src["rng_prompts"]), ref_steps=int(src["ref_steps"]),
                sweeps=int(src["sweeps"]), n_cand=int(src["n_candidates"]),
                model=str(src["model"]), attn=str(src["attn"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="search_sd.py output")
    ap.add_argument("--coco-captions", required=True)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--sdpost", default=None,
                    help="--warm: sd_checks.py --render output; its menu matrix is reused "
                         "(verified). Without it the menu is computed here.")
    ap.add_argument("--also", nargs="*", default=[],
                    help="--floor: more search_sd.py runs with the same prompts and guidance")
    ap.add_argument("--k", type=int, default=32, help="--floor: prompts used (first k)")
    ap.add_argument("--search-sweeps", type=int, default=30,
                    help="--warm: sweep cap of the two new shared searches (they stop at "
                         "convergence; a capped one is reported with a WARNING)")
    ap.add_argument("--round2", default=None,
                    help="--warm: an earlier --warm output for this run: search the shared N "
                         "again from the best one-node deletion of its best N+1, and, if that "
                         "finds a better N, the shared N+1 from the best insertion into it")
    ap.add_argument("--uniform-only", action="store_true",
                    help="--warm: after the gates, only score the uniform N schedule per prompt "
                         "on the batched path (checked against the run's log)")
    ap.add_argument("--rescore", default=None,
                    help="--warm: an earlier --warm output for this run; after the gates, only "
                         "re-score its best shared N and N+1 per prompt on the batched path "
                         "(for the bootstrap) and check them against its stored means")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--decode-chunk", type=int, default=4)
    ap.add_argument("--cache-cap", type=int, default=256)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.warm == args.floor:
        fail("pass exactly one of --warm or --floor")

    src = np.load(args.npz)
    st = settings(src)
    B, n, T, cfg, seed = st["B"], st["n"], st["T"], st["cfg"], st["seed"]
    others = [np.load(p) for p in args.also]
    for p, o in zip(args.also, others):
        so = settings(o)
        for key in ("n", "T", "cfg", "seed", "rng_prompts", "ref_steps", "model", "attn"):
            if so[key] != st[key]:
                fail(f"{p}: {key}={so[key]!r} differs from {args.npz} ({st[key]!r})")
    if args.floor and (args.k > n or args.k % args.batch or args.k % args.decode_chunk):
        fail("--k must be <= n and a multiple of --batch and --decode-chunk")

    import lpips
    from diffusers import StableDiffusionPipeline

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    t_start = time.time()
    dc = args.decode_chunk

    print("=" * 74)
    print(f"SD 1.5 {'N+1 WARM START' if args.warm else 'REFERENCE FLOOR'}  source {args.npz}")
    print(f"  B={B} n={n} cfg={cfg} seed={seed} ref_steps={st['ref_steps']} "
          f"sweeps={st['sweeps']} n_cand={st['n_cand']} attn={st['attn']} batch={args.batch} "
          f"decode_chunk={dc}")
    print("=" * 74)

    img_ids, ann_ids, prompts = coco_prompts(args.coco_captions, n,
                                             max(st["rng_prompts"], n), seed)
    seeds = [seed * 100000 + i for i in range(n)]
    for p, o in [(args.npz, src)] + list(zip(args.also, others)):
        same = (np.array_equal(np.asarray(img_ids, dtype=np.int64), o["coco_image_ids"])
                and np.array_equal(np.asarray(ann_ids, dtype=np.int64),
                                   o["coco_annotation_ids"])
                and [str(q) for q in o["prompts"]] == prompts
                and np.array_equal(np.asarray(seeds, dtype=np.int64), o["noise_seeds"]))
        print(f"  prompts, annotation ids and noise seeds match {os.path.basename(p)}: {same}")
        if not same:
            fail(f"prompts or seeds differ from {p}")

    # ---- rollout setup: copied from sd_checks.py (itself copied from search_sd.py) ----
    pipe = StableDiffusionPipeline.from_pretrained(
        st["model"], torch_dtype=torch.float16, variant="fp16", safety_checker=None,
        feature_extractor=None, requires_safety_checker=False, local_files_only=True)
    unet = pipe.unet.to(device).eval()
    vae = pipe.vae.to(device).eval()
    te = pipe.text_encoder.to(device).eval()
    tok = pipe.tokenizer
    if st["attn"] == "classic":
        from diffusers.models.attention_processor import AttnProcessor
        unet.set_attn_processor(AttnProcessor())
        vae.set_attn_processor(AttnProcessor())
    alphas = pipe.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
    assert alphas.shape[0] == T, alphas.shape
    sf = float(vae.config.scaling_factor)

    def embed(text):
        ids = tok([text], padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(device)
        return te(ids)[0]

    ctx = torch.cat([embed(p) for p in prompts], dim=0)
    unc = embed("")
    z_T = torch.cat([torch.randn(1, 4, 64, 64, generator=torch.Generator("cpu").manual_seed(s))
                     for s in seeds], dim=0).to(device)

    def ddim_step(z, t_s, t_e, c):
        b = z.shape[0]
        tt = torch.full((2 * b,), int(t_s), device=device, dtype=torch.long)
        e = unet(torch.cat([z, z], 0).half(), tt,
                 encoder_hidden_states=torch.cat([unc.expand(b, -1, -1), c], 0)).sample.float()
        e_u, e_c = e.chunk(2)
        eps = e_u + cfg * (e_c - e_u)
        a_s, a_e = alphas[int(t_s)], alphas[int(t_e)]
        x0 = (z - (1.0 - a_s).sqrt() * eps) / a_s.sqrt().clamp(min=1e-8)
        return a_e.sqrt() * x0 + (1.0 - a_e).clamp(min=0).sqrt() * eps

    def decode(z, chunk):
        out = torch.cat([vae.decode((z[i:i + chunk] / sf).half()).sample.float()
                         for i in range(0, z.shape[0], chunk)], dim=0)
        if not torch.isfinite(out).all():
            raise SystemExit("non-finite decoded image (fp16 overflow?)")
        return out

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    def dev(a, b):
        return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())

    m = args.k if args.floor else n      # prompts used; the first m, same batches as the run
    zs, cs = z_T[:m], ctx[:m]
    t0 = time.time()
    ref = np.linspace(T - 1, 0, st["ref_steps"] + 1).round().astype(np.int64)
    ref_img = decode(Roller(ddim_step, zs, cs, args.batch, 0).run(ref, use_cache=False), dc)
    print(f"  reference generated at {st['ref_steps']} guided steps for {m} prompts in "
          f"{time.time() - t0:.0f}s", flush=True)

    def batched_img(s):
        return decode(Roller(ddim_step, zs, cs, args.batch, 0).run(s, use_cache=False), dc)

    def one_img(s, i):
        r = Roller(ddim_step, z_T[i:i + 1], ctx[i:i + 1], 1, 0)
        return decode(r.run(s, use_cache=False), 1)

    if args.floor:
        run_floor(args, st, src, others, m, T, ddim_step, zs, cs, ref_img, batched_img, one_img,
                  loss_fn, decode, dev, t_start)
    else:
        run_warm(args, st, src, n, T, ddim_step, z_T, ctx, ref_img, batched_img, one_img,
                 loss_fn, decode, dev, t_start)


def run_floor(args, st, src, others, k, T, ddim_step, zs, cs, ref_img, batched_img, one_img,
              loss_fn, decode, dev, t_start):
    dc = args.decode_chunk
    t0 = time.time()
    hi = np.arange(T - 1, -1, -1, dtype=np.int64)
    hi_img = decode(Roller(ddim_step, zs, cs, args.batch, 0).run(hi, use_cache=False), dc)
    floor = np.asarray(lpips_to(loss_fn, ref_img, hi_img, dc), np.float64)
    print(f"  floor  LPIPS({st['ref_steps']}-step, {len(hi) - 1}-step guided DDIM) over {k} "
          f"prompts: mean {floor.mean():.6f}  median {np.median(floor):.6f}  "
          f"max {floor.max():.6f}  ({time.time() - t0:.0f}s)", flush=True)

    def both(img, i0, i1):
        return (np.asarray(lpips_to(loss_fn, img, ref_img[i0:i1], min(dc, i1 - i0)), np.float64),
                np.asarray(lpips_to(loss_fn, img, hi_img[i0:i1], min(dc, i1 - i0)), np.float64))

    names = [os.path.basename(args.npz)] + [os.path.basename(p) for p in args.also]
    results = []
    for name, o in zip(names, [src] + others):
        fB, fB1 = o["fixed_B"].astype(np.int64), o["fixed_B1"].astype(np.int64)
        per_sched = o["per_sched"].astype(np.int64)
        b200, bhi = both(batched_img(fB), 0, k)
        b1_200, b1hi = both(batched_img(fB1), 0, k)
        base = np.array([[v[0] for v in both(one_img(fB, i), i, i + 1)] for i in range(k)])
        own = np.array([[v[0] for v in both(one_img(per_sched[i], i), i, i + 1)]
                        for i in range(k)])
        devs = {"batched shared N vs D_fixed": dev(b200, o["D_fixed"][:k]),
                "batched shared N+1 vs D_fixed_B1": dev(b1_200, o["D_fixed_B1"][:k]),
                "per-prompt own vs per_img": dev(own[:, 0], o["per_img"][:k]),
                "per-prompt shared N vs D_fixed_unbatched":
                    dev(base[:, 0], o["D_fixed_unbatched"][:k])}
        for key, v in devs.items():
            print(f"  reproduce  {name}  {key:42s} max |dev| {v:.3e}  "
                  f"{'PASS' if v <= args.tol else 'FAIL'}", flush=True)
        if not all(v <= args.tol for v in devs.values()):
            fail(f"{name}: stored values not reproduced; nothing saved")
        g200, ghi = float((base[:, 0] - own[:, 0]).mean()), float((base[:, 1] - own[:, 1]).mean())
        s200, shi = float((b200 - b1_200).mean()), float((bhi - b1hi).mean())
        dn1_full = float(np.asarray(o["D_fixed_B1"], np.float64).mean())
        x = dict(name=name, B=int(o["budget"]), dn1_full=dn1_full, g200=g200, ghi=ghi,
                 s200=s200, shi=shi, R200=ratio(g200, s200), Rhi=ratio(ghi, shi),
                 R_full=ratio(float(o["gain"]), float(o["step_value"])),
                 maxdev=max(devs.values()), b200=b200, bhi=bhi, b1_200=b1_200, b1hi=b1hi,
                 base=base, own=own)
        results.append(x)
        print(f"  {name}: D(N+1) full {dn1_full:.6f} = {dn1_full / floor.mean():.1f}x floor;"
              f"  subset R {x['R200']:.3f} (200-step ref) vs {x['Rhi']:.3f} (999-step ref)",
              flush=True)

    print(f"\n  {'row':44s} {'D(N+1)/floor':>12s} {'R full':>7s} {'R sub/200':>9s} "
          f"{'R sub/999':>9s}  gain 200->999      step 200->999")
    for x in results:
        print(f"  {x['name']:44s} {x['dn1_full'] / floor.mean():12.1f} {x['R_full']:7.3f} "
              f"{x['R200']:9.3f} {x['Rhi']:9.3f}  {x['g200']:.5f}->{x['ghi']:.5f}  "
              f"{x['s200']:.5f}->{x['shi']:.5f}")
    print()
    for x in results:
        print(f"SUMMARY refloor dataset=sd15_coco cfg={st['cfg']} row={x['name']} "
              f"sampler=ddim B={x['B']} k={k} floor_mean={floor.mean():.6f} "
              f"DN1_full={x['dn1_full']:.6f} ratio={x['dn1_full'] / floor.mean():.1f} "
              f"R_full={x['R_full']:.4f} R_sub200={x['R200']:.4f} R_sub999={x['Rhi']:.4f} "
              f"repro_max_dev={x['maxdev']:.3e}")
    save = dict(source=np.asarray(args.npz), cfg=np.float64(st["cfg"]), k=np.int64(k),
                ref_steps=np.int64(st["ref_steps"]), hi_steps=np.int64(len(hi) - 1),
                floor=floor, rows=np.asarray([x["name"] for x in results]))
    for j, x in enumerate(results):
        for key in ("b200", "bhi", "b1_200", "b1hi", "base", "own"):
            save[f"r{j}_{key}"] = x[key]
        for key in ("g200", "ghi", "s200", "shi", "R200", "Rhi", "R_full", "dn1_full"):
            save[f"r{j}_{key}"] = np.float64(x[key])
    np.savez(args.out, **save)
    print(f"  saved {args.out}  (wall {time.time() - t_start:.0f}s)")


def run_rescore(args, st, src, n, ref_img, batched_img, one_img, loss_fn, dev, t_start):
    dc, B = args.decode_chunk, st["B"]
    prior = np.load(args.rescore)
    if os.path.basename(str(prior["source"])) != os.path.basename(args.npz) \
            or int(prior["budget"]) != B or int(prior["n_images"]) != n \
            or abs(float(prior["cfg"]) - st["cfg"]) > 0:
        fail(f"{args.rescore} is not a --warm output for {args.npz}")
    fB, fB1 = src["fixed_B"].astype(np.int64), src["fixed_B1"].astype(np.int64)
    D_fB = lpips_to(loss_fn, batched_img(fB), ref_img, dc)
    D_fB1 = lpips_to(loss_fn, batched_img(fB1), ref_img, dc)
    base_chk = np.array([float(lpips_to(loss_fn, one_img(fB, i), ref_img[i:i + 1], 1)[0])
                         for i in range(n)])
    bN = np.asarray(prior["best_B"]).astype(np.int64)
    bN1 = np.asarray(prior["best_B1"]).astype(np.int64)
    D_N = lpips_to(loss_fn, batched_img(bN), ref_img, dc)
    D_N1 = lpips_to(loss_fn, batched_img(bN1), ref_img, dc)
    devs = {"shared N (batched) vs stored D_fixed": dev(D_fB, src["D_fixed"]),
            "shared N+1 (batched) vs stored D_fixed_B1": dev(D_fB1, src["D_fixed_B1"]),
            "shared N (per-prompt path) vs stored D_fixed_unbatched":
                dev(base_chk, src["D_fixed_unbatched"]),
            "best shared N (batched) vs earlier --warm values":
                dev(D_N, prior["D_best_B_batched"]),
            "best shared N mean vs earlier --warm mean":
                abs(float(D_N.mean()) - float(prior["best_B_mean"])),
            "best shared N+1 mean vs earlier --warm mean":
                abs(float(D_N1.mean()) - float(prior["best_B1_mean"]))}
    for key, v in devs.items():
        print(f"  reproduce  {key:52s} max |dev| {v:.3e}  {'PASS' if v <= args.tol else 'FAIL'}")
    if not all(v <= args.tol for v in devs.values()):
        fail("stored LPIPS values not reproduced; nothing written")
    np.savez(args.out, source=np.asarray(args.npz), rescore_of=np.asarray(args.rescore),
             budget=np.int64(B), n_images=np.int64(n), cfg=np.float64(st["cfg"]),
             best_B=bN, best_B1=bN1, best_B_mean=np.float64(float(D_N.mean())),
             best_B1_mean=np.float64(float(D_N1.mean())),
             D_best_B_batched=np.asarray(D_N, np.float64),
             D_best_B1_batched=np.asarray(D_N1, np.float64),
             D_best_B_unbatched=np.asarray(prior["D_best_B_unbatched"], np.float64),
             own_batched=np.asarray(prior["own_batched"], np.float64),
             repro_keys=np.asarray(list(devs)), repro_devs=np.asarray(list(devs.values())),
             tol=np.float64(args.tol))
    print(f"\nSUMMARY sd_rescore B={B} n={n} cfg={st['cfg']} best_N={float(D_N.mean()):.6f} "
          f"best_N1={float(D_N1.mean()):.6f} repro_max_dev={max(devs.values()):.3e}")
    print(f"  saved {args.out}  (wall {time.time() - t_start:.0f}s)")


def run_extra(args, st, src, n, T, ddim_step, z_T, ctx, ref_img, batched_img, one_img,
              loss_fn, decode, dev, t_start):
    """--uniform-only and --round2 (see the argument help)."""
    dc, B = args.decode_chunk, st["B"]
    fB, fB1 = src["fixed_B"].astype(np.int64), src["fixed_B1"].astype(np.int64)
    per_img = np.asarray(src["per_img"], np.float64)
    D_fB = lpips_to(loss_fn, batched_img(fB), ref_img, dc)
    D_fB1 = lpips_to(loss_fn, batched_img(fB1), ref_img, dc)
    base_chk = np.array([float(lpips_to(loss_fn, one_img(fB, i), ref_img[i:i + 1], 1)[0])
                         for i in range(n)])
    devs = {"shared N (batched) vs stored D_fixed": dev(D_fB, src["D_fixed"]),
            "shared N+1 (batched) vs stored D_fixed_B1": dev(D_fB1, src["D_fixed_B1"]),
            "shared N (per-prompt path) vs stored D_fixed_unbatched":
                dev(base_chk, src["D_fixed_unbatched"])}
    name = os.path.basename(args.npz)

    if args.uniform_only:
        u = uniform(T, B)
        D_u = lpips_to(loss_fn, batched_img(u), ref_img, dc)
        logged = logged_uniform(args.npz[:-4] + ".txt")
        if logged is None:
            fail("no uniform line in the run's log to check against")
        devs["uniform N mean vs the log (6 decimals)"] = abs(float(D_u.mean()) - logged)
        for key, v in devs.items():
            lim = 5.1e-7 if "log" in key else args.tol
            print(f"  reproduce  {key:52s} max |dev| {v:.3e}  {'PASS' if v <= lim else 'FAIL'}")
        if not all(v <= (5.1e-7 if "log" in k else args.tol) for k, v in devs.items()):
            fail("stored LPIPS values not reproduced; nothing written")
        np.savez(args.out, source=np.asarray(args.npz), budget=np.int64(B), n_images=np.int64(n),
                 cfg=np.float64(st["cfg"]), uniform_B=u,
                 uniform_mean=np.float64(float(D_u.mean())),
                 logged_uniform_mean=np.float64(logged),
                 D_uniform_batched=np.asarray(D_u, np.float64),
                 repro_keys=np.asarray(list(devs)), repro_devs=np.asarray(list(devs.values())),
                 tol=np.float64(args.tol))
        print(f"\nSUMMARY sd_uniform B={B} n={n} cfg={st['cfg']} uniform={float(D_u.mean()):.6f} "
              f"stored_N={float(D_fB.mean()):.6f} repro_max_dev={max(devs.values()):.3e}")
        print(f"  saved {args.out}  (wall {time.time() - t_start:.0f}s)")
        return

    prior = np.load(args.round2)
    if os.path.basename(str(prior["source"])) != name or int(prior["budget"]) != B \
            or int(prior["n_images"]) != n or abs(float(prior["cfg"]) - st["cfg"]) > 0:
        fail(f"{args.round2} is not a --warm output for {args.npz}")
    pN = np.asarray(prior["best_B"]).astype(np.int64)
    pN1 = np.asarray(prior["best_B1"]).astype(np.int64)
    pN_mean, pN1_mean = float(prior["best_B_mean"]), float(prior["best_B1_mean"])
    D_pN = lpips_to(loss_fn, batched_img(pN), ref_img, dc)
    D_pN1 = lpips_to(loss_fn, batched_img(pN1), ref_img, dc)
    devs["round-1 best shared N mean vs stored"] = abs(float(D_pN.mean()) - pN_mean)
    devs["round-1 best shared N+1 mean vs stored"] = abs(float(D_pN1.mean()) - pN1_mean)
    for key, v in devs.items():
        print(f"  reproduce  {key:52s} max |dev| {v:.3e}  {'PASS' if v <= args.tol else 'FAIL'}")
    if not all(v <= args.tol for v in devs.values()):
        fail("stored LPIPS values not reproduced; nothing written")

    shared = Roller(ddim_step, z_T, ctx, args.batch, args.cache_cap)

    def obj(q):
        return float(lpips_to(loss_fn, decode(shared.run(q), dc), ref_img, dc).mean())

    t0 = time.time()
    dels = deletions(pN1)
    del_vals = [obj(q) for q in dels]
    k0 = int(np.argmin(del_vals))
    fD, fD_mean, _, eD, cD = coord_descent(obj, dels[k0], args.search_sweeps, st["n_cand"])
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
        fI, fI_mean, _, eI, cI = coord_descent(obj, ins[j0], args.search_sweeps, st["n_cand"])
        print(f"  round 2, shared N+1 from the best insertion ({ins_vals[j0]:.6f}): -> "
              f"{fI_mean:.6f} {fI.tolist()}  ({eI} evals, "
              f"{'converged' if cI else 'HIT SWEEP CAP'}, {time.time() - t0:.0f}s)")
    else:
        print("  shared N unchanged, so the N+1 insertion start equals round 1's: not repeated")
    newN1 = fI_mean < pN1_mean - 1e-9
    bestN1, bestN1_mean = (fI, fI_mean) if newN1 else (pN1, pN1_mean)
    print(f"  round-1 shared N+1 {pN1_mean:.6f}: round 2 "
          f"{'BEATS' if newN1 else 'does not beat'} it by {pN1_mean - fI_mean:+.6f}", flush=True)
    D_N = lpips_to(loss_fn, batched_img(bestN), ref_img, dc) if newN else D_pN
    D_N1 = lpips_to(loss_fn, batched_img(bestN1), ref_img, dc) if newN1 else D_pN1
    for label, D, want in (("N", D_N, bestN_mean), ("N+1", D_N1, bestN1_mean)):
        devs[f"uncached re-score of round-2 best shared {label} vs search value"] = abs(
            float(D.mean()) - want)
        if abs(float(D.mean()) - want) > args.tol:
            print(f"  WARNING: uncached re-score of the round-2 best shared {label} differs")
    b1 = np.asarray(prior["D_best_B_unbatched"], np.float64)
    base_u = (np.array([float(lpips_to(loss_fn, one_img(bestN, i), ref_img[i:i + 1], 1)[0])
                        for i in range(n)]) if newN else b1)
    own_b = np.asarray(prior["own_batched"], np.float64)
    D_N64, D_pN64 = np.asarray(D_N, np.float64), np.asarray(D_pN, np.float64)
    step1, step2 = pN_mean - pN1_mean, bestN_mean - bestN1_mean
    rows = {"round 1, batched (primary)":
                (float(D_pN64.mean() - np.minimum(own_b, D_pN64).mean()), step1),
            "round 2, batched (primary)":
                (float(D_N64.mean() - np.minimum(own_b, D_N64).mean()), step2),
            "round 1, per-prompt": (float(b1.mean() - np.minimum(per_img, b1).mean()), step1),
            "round 2, per-prompt":
                (float(base_u.mean() - np.minimum(per_img, base_u).mean()), step2)}
    print(f"\n  {'':30s} {'gain':>9s} {'step':>9s} {'R':>7s} {'R/N':>6s}")
    for key, (g, sv) in rows.items():
        print(f"  {key:30s} {g:9.6f} {sv:9.6f} {ratio(g, sv):7.3f} {100 * ratio(g, sv) / B:5.1f}%")
    for label, c in (("shared N", cD), ("shared N+1", cI)):
        if not c:
            print(f"  WARNING: the round-2 {label} search hit the {args.search_sweeps}-sweep cap; "
                  f"its improvement is a lower bound")
    print(f"\nSUMMARY sd_round2 B={B} n={n} cfg={st['cfg']} round1_N={pN_mean:.6f} "
          f"round2_N={bestN_mean:.6f} d_N={pN_mean - bestN_mean:.6f} round1_N1={pN1_mean:.6f} "
          f"round2_N1={bestN1_mean:.6f} d_N1={pN1_mean - bestN1_mean:.6f} conv_N={int(cD)} "
          f"conv_N1={int(cI)} R_round1={ratio(*rows['round 1, batched (primary)']):.4f} "
          f"R_round2={ratio(*rows['round 2, batched (primary)']):.4f} "
          f"repro_max_dev={max(devs.values()):.3e}")
    np.savez(args.out, source=np.asarray(args.npz), round2_of=np.asarray(args.round2),
             budget=np.int64(B), n_images=np.int64(n), cfg=np.float64(st["cfg"]),
             search_sweeps=np.int64(args.search_sweeps), deletions=np.stack(dels),
             deletion_means=np.asarray(del_vals), round2_B=fD, round2_B_mean=np.float64(fD_mean),
             round2_B_conv=np.int8(cD), round2_B1=fI, round2_B1_mean=np.float64(fI_mean),
             round2_B1_conv=np.int8(cI), best_B=bestN, best_B_mean=np.float64(bestN_mean),
             best_B1=bestN1, best_B1_mean=np.float64(bestN1_mean),
             D_best_B_batched=D_N64, D_best_B1_batched=np.asarray(D_N1, np.float64),
             D_best_B_unbatched=base_u, own_batched=own_b,
             row_names=np.asarray(list(rows)), row_gain_step=np.asarray(list(rows.values())),
             repro_keys=np.asarray(list(devs)), repro_devs=np.asarray(list(devs.values())),
             tol=np.float64(args.tol))
    print(f"  saved {args.out}  (wall {time.time() - t_start:.0f}s)")


def run_warm(args, st, src, n, T, ddim_step, z_T, ctx, ref_img, batched_img, one_img,
             loss_fn, decode, dev, t_start):
    if args.rescore:
        run_rescore(args, st, src, n, ref_img, batched_img, one_img, loss_fn, dev, t_start)
        return
    if args.uniform_only or args.round2:
        run_extra(args, st, src, n, T, ddim_step, z_T, ctx, ref_img, batched_img, one_img,
                  loss_fn, decode, dev, t_start)
        return
    dc, B = args.decode_chunk, st["B"]
    fB, fB1 = src["fixed_B"].astype(np.int64), src["fixed_B1"].astype(np.int64)
    per_sched = src["per_sched"].astype(np.int64)
    per_img, per_base = np.asarray(src["per_img"], np.float64), np.asarray(
        src["D_fixed_unbatched"], np.float64)
    if args.sdpost:
        post = np.load(args.sdpost)
        if int(post["budget"]) != B or int(post["n_images"]) != n \
                or abs(float(post["cfg"]) - st["cfg"]) > 0 or \
                os.path.basename(str(post["source"])) != os.path.basename(args.npz):
            fail(f"{args.sdpost} is not the post-hoc of {args.npz}")
        if int(post["batch"]) != args.batch or int(post["decode_chunk"]) != dc:
            fail(f"{args.sdpost} used batch {int(post['batch'])} / decode chunk "
                 f"{int(post['decode_chunk'])}, this run {args.batch} / {dc}")
        cols = post["menu_columns"].astype(np.int64)
        M = np.asarray(post["menu_M"], np.float64)[0]
        if not (np.array_equal(cols[0], fB) and np.array_equal(cols[1:], per_sched)):
            fail("menu columns are not the run's shared and per-prompt schedules")
    else:
        # menu matrix as sd_checks.py builds it (LPIPS only): column 0 = shared N,
        # column j+1 = prompt j's own schedule, every prompt scored on the batched path
        t0 = time.time()
        cols = np.concatenate([fB[None], per_sched], 0)
        M = np.zeros((n, n + 1))
        for j in range(n + 1):
            M[:, j] = lpips_to(loss_fn, batched_img(cols[j]), ref_img, dc)
        print(f"  menu matrix computed: {n + 1} schedules ({time.time() - t0:.0f}s)", flush=True)
    own_b = M[np.arange(n), np.arange(n) + 1]       # prompt i, own schedule, batched path

    # float32 arrays as lpips_to returns them; their .mean() is the search objective's value
    D_fB = lpips_to(loss_fn, batched_img(fB), ref_img, dc)
    D_fB1 = lpips_to(loss_fn, batched_img(fB1), ref_img, dc)
    c_best = int(np.argmin(M.mean(0)))
    D_c = lpips_to(loss_fn, batched_img(cols[c_best]), ref_img, dc)
    base_chk = np.array([float(lpips_to(loss_fn, one_img(fB, i), ref_img[i:i + 1], 1)[0])
                         for i in range(n)])
    devs = {"shared N (batched) vs stored D_fixed": dev(D_fB, src["D_fixed"]),
            "shared N (per-prompt path) vs stored D_fixed_unbatched": dev(base_chk, per_base),
            "shared N+1 (batched) vs stored D_fixed_B1": dev(D_fB1, src["D_fixed_B1"]),
            "best menu column (batched) vs menu matrix": dev(D_c, M[:, c_best]),
            "menu column 0 vs stored D_fixed": dev(M[:, 0], src["D_fixed"])}
    for key, v in devs.items():
        print(f"  reproduce  {key:48s} max |dev| {v:.3e}  {'PASS' if v <= args.tol else 'FAIL'}")
    if not all(v <= args.tol for v in devs.values()):
        fail("stored LPIPS values not reproduced; nothing written")
    fB_mean, fB1_mean = float(D_fB.mean()), float(D_fB1.mean())
    print(f"  stored shared N {fB_mean:.6f}   stored shared N+1 {fB1_mean:.6f}   best menu "
          f"column {c_best} ({'shared' if c_best == 0 else f'prompt {c_best - 1}'}) "
          f"{float(D_c.mean()):.6f}", flush=True)

    shared = Roller(ddim_step, z_T, ctx, args.batch, args.cache_cap)

    def obj(s):
        return float(lpips_to(loss_fn, decode(shared.run(s), dc), ref_img, dc).mean())

    # 1. improved shared N: search from the best menu column
    t0 = time.time()
    fP, fP_mean, fP_start, eP, cP = coord_descent(obj, cols[c_best], args.search_sweeps,
                                                  st["n_cand"])
    if abs(fP_start - float(D_c.mean())) > args.tol:
        fail(f"cached start {fP_start!r} != uncached {float(D_c.mean())!r}")
    print(f"\n  shared N from menu column {c_best}: {fP_start:.6f} -> {fP_mean:.6f}  "
          f"{fP.tolist()}  ({eP} evals, {'converged' if cP else 'HIT SWEEP CAP'}, "
          f"{time.time() - t0:.0f}s)", flush=True)
    if fP_mean < fB_mean - 1e-9:
        bestN, bestN_mean, src_N = fP, fP_mean, "new"
    else:
        bestN, bestN_mean, src_N = fB, fB_mean, "stored"

    # 2. N+1 from the best single-node insertion into the best shared N
    t0 = time.time()
    ins = insertions(bestN)
    ins_vals = [obj(s) for s in ins]
    j0 = int(np.argmin(ins_vals))
    print(f"\n  N+1 insertions into the {src_N} shared N: {len(ins)} tried, best "
          f"{ins_vals[j0]:.6f} {ins[j0].tolist()}", flush=True)
    fW, fW_mean, _, eW, cW = coord_descent(obj, ins[j0], args.search_sweeps, st["n_cand"])
    print(f"  shared N+1 warm start: {ins_vals[j0]:.6f} -> {fW_mean:.6f}  {fW.tolist()}  "
          f"({eW} evals, {'converged' if cW else 'HIT SWEEP CAP'}, {time.time() - t0:.0f}s)")
    print(f"  stored shared N+1 {fB1_mean:.6f}   warm start "
          f"{'BEATS' if fW_mean < fB1_mean - 1e-9 else 'does not beat'} it by "
          f"{fB1_mean - fW_mean:+.6f}", flush=True)
    bestN1, bestN1_mean = (fW, fW_mean) if fW_mean < fB1_mean - 1e-9 else (fB1, fB1_mean)

    # 3. gain, step value and R
    D_N = lpips_to(loss_fn, batched_img(bestN), ref_img, dc)
    if abs(float(D_N.mean()) - bestN_mean) > args.tol:
        fail("uncached re-score of the best shared N differs from the search value")
    D_N1 = lpips_to(loss_fn, batched_img(bestN1), ref_img, dc)
    devs["uncached re-score of best shared N+1 vs search value"] = abs(
        float(D_N1.mean()) - bestN1_mean)
    if devs["uncached re-score of best shared N+1 vs search value"] > args.tol:
        print("  WARNING: uncached re-score of the best shared N+1 differs from the search "
              "value (recorded in repro_devs)")
    if src_N == "new":
        base_u = np.array([float(lpips_to(loss_fn, one_img(bestN, i), ref_img[i:i + 1], 1)[0])
                           for i in range(n)])
    else:
        base_u = per_base
    own_u2 = np.minimum(per_img, base_u)
    D_N64, D_fB64 = np.asarray(D_N, np.float64), np.asarray(D_fB, np.float64)
    own_b2 = np.minimum(own_b, D_N64)
    step_new = bestN_mean - bestN1_mean
    step0 = fB_mean - fB1_mean
    rows = {
        "stored, per-prompt path (as published)":
            (float(src["gain"]), float(src["step_value"])),
        "stored, batched path":
            (float(D_fB64.mean() - np.minimum(own_b, D_fB64).mean()), step0),
        "stored, batched, own schedules only": (float(D_fB64.mean() - own_b.mean()), step0),
        "best shared N and N+1, batched path (primary)":
            (float(D_N64.mean() - own_b2.mean()), step_new),
        "best shared N and N+1, batched, own schedules only":
            (float(D_N64.mean() - own_b.mean()), step_new),
        "best shared N and N+1, per-prompt path": (float(base_u.mean() - own_u2.mean()), step_new),
    }
    print(f"\n  {'':52s} {'gain':>9s} {'step':>9s} {'R':>7s} {'R/N':>6s}")
    for key, (g, s) in rows.items():
        print(f"  {key:52s} {g:9.6f} {s:9.6f} {ratio(g, s):7.3f} {100 * ratio(g, s) / B:5.1f}%")
    g_p, s_p = rows["best shared N and N+1, batched path (primary)"]
    for label, c in (("shared N", cP), ("shared N+1", cW)):
        if not c:
            print(f"  WARNING: the new {label} search hit the {args.search_sweeps}-sweep cap; "
                  f"its improvement is a lower bound")
    print(f"\nSUMMARY sd_warm B={B} n={n} cfg={st['cfg']} menu_col={c_best} "
          f"stored_N={fB_mean:.6f} new_N={fP_mean:.6f} d_N={fB_mean - bestN_mean:.6f} "
          f"stored_N1={fB1_mean:.6f} new_N1={fW_mean:.6f} d_N1={fB1_mean - bestN1_mean:.6f} "
          f"conv_N={int(cP)} conv_N1={int(cW)} "
          f"R_stored_batched={ratio(*rows['stored, batched path']):.4f} "
          f"R_primary={ratio(g_p, s_p):.4f} "
          f"R_perprompt={ratio(*rows['best shared N and N+1, per-prompt path']):.4f} "
          f"repro_max_dev={max(devs.values()):.3e}")
    np.savez(args.out, source=np.asarray(args.npz), sdpost=np.asarray(args.sdpost or ""),
             budget=np.int64(B), n_images=np.int64(n), cfg=np.float64(st["cfg"]),
             search_sweeps=np.int64(args.search_sweeps),
             menu_col=np.int64(c_best), polish_init=cols[c_best], polish_B=fP,
             polish_B_mean=np.float64(fP_mean), polish_B_evals=np.int64(eP),
             polish_B_conv=np.int8(cP), best_B=bestN, best_B_mean=np.float64(bestN_mean),
             insertions=np.stack(ins), insertion_means=np.asarray(ins_vals),
             warm_B1=fW, warm_B1_mean=np.float64(fW_mean), warm_B1_evals=np.int64(eW),
             warm_B1_conv=np.int8(cW), best_B1=bestN1, best_B1_mean=np.float64(bestN1_mean),
             D_best_B_batched=D_N64, D_best_B_unbatched=base_u, own_batched=own_b,
             D_best_B1_batched=np.asarray(D_N1, np.float64),
             menu_columns=cols, menu_M=M,
             row_names=np.asarray(list(rows)), row_gain_step=np.asarray(list(rows.values())),
             repro_keys=np.asarray(list(devs)), repro_devs=np.asarray(list(devs.values())),
             tol=np.float64(args.tol))
    print(f"  saved {args.out}  (wall {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
