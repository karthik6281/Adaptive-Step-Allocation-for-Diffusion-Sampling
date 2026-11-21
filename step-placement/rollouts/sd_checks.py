"""Post-hoc checks on a finished search_sd.py run (SD 1.5, COCO prompts). The source npz is
only read, never written.

  --render   1. re-render the run's shared N, shared N+1 and per-prompt N schedules on its own
                prompts and noise and confirm the STORED LPIPS values are reproduced (max
                deviation <= --tol, default 1e-9). Any failure aborts before results are
                written;
             2. re-score uniform N, shared N, shared N+1 and per-prompt N with PSNR and DISTS
                and recompute gain, step value and R per metric on exactly the paths the
                search used (gain: per-prompt path; step value: batched path), plus the
                shared-vs-uniform gain;
             3. score every prompt under the shared schedule and under every per-prompt
                schedule on the batched path (the schedule-menu matrix) and report how much
                of the per-prompt gain a greedy menu of K schedules captures with the best
                menu entry chosen per prompt, in-sample and 2-fold (menu picked on even
                prompts, scored on odd, and vice versa);
             4. save PNGs: reference, uniform N, shared N, shared N+1, per-prompt N.
  --restart  re-run the shared search at N, with the run's own sweeps and candidates, from a
             different start: the quadratic spacing t_k = round(999 (1 - k/N)^2). If it beats
             the stored shared schedule, N+1 is restarted the same way and gain, step value
             and R are recomputed with the better of the two shared searches at each budget
             (each prompt's per-prompt value becomes the better of its own search and the new
             shared schedule).

The rollout (prompts, noise, model, fp16/fp32 split, attention, DDIM step, decode, reference,
LPIPS) is copied from search_sd.py, Roller and coco_prompts are imported from it, and every
setting is read from the source npz. --batch and --decode-chunk must match the search's
(search_sd.py defaults, 8 and 4); the reproduction check fails otherwise.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from reward_structure import lpips_to  # noqa: E402
from search_sd import Roller, coco_prompts  # noqa: E402
from search_vs_sample_v2 import coord_descent, uniform  # noqa: E402


def quadratic(T, B):
    v = np.round((T - 1) * (1.0 - np.arange(B + 1) / B) ** 2).astype(np.int64)
    v[0], v[-1] = T - 1, 0
    assert np.all(np.diff(v) < 0), v
    return v


def greedy_menu(M, rows, cands, K):
    """Greedily pick up to K columns of M minimising the mean over `rows` of the row minimum."""
    sub = M[rows]
    chosen, means = [], []
    cur = np.full(len(rows), np.inf)
    for _ in range(min(K, len(cands))):
        best_v, best_c = np.inf, None
        for c in cands:
            if c in chosen:
                continue
            v = float(np.minimum(cur, sub[:, c]).mean())
            if v < best_v - 1e-12:
                best_v, best_c = v, c
        chosen.append(best_c)
        cur = np.minimum(cur, sub[:, best_c])
        means.append(best_v)
    return chosen, means


def menu_analysis(L, menu_max):
    """L[i, 0]: prompt i under the shared schedule; L[i, j+1]: prompt i under prompt j's own
    schedule (lower is better). Fraction of the per-prompt gain captured by a greedy menu of
    K schedules, each prompt taking its best entry: in-sample, and 2-fold (menu chosen from
    the shared schedule and the training half's own schedules, scored on the other half)."""
    n = L.shape[0]
    rows = np.arange(n)
    base_b, own_b = float(L[:, 0].mean()), float(L[rows, rows + 1].mean())
    den_in = base_b - own_b
    K = min(menu_max, n + 1)
    chosen_in, means_in = greedy_menu(L, rows, list(range(n + 1)), K)
    frac_in = [(base_b - m) / den_in if den_in > 0 else float("nan") for m in means_in]
    folds = [np.arange(0, n, 2), np.arange(1, n, 2)]
    K_cv = min(menu_max, 1 + min(len(f) for f in folds))
    num_cv, den_cv, chosen_cv = np.zeros(K_cv), 0.0, []
    for a, b in ((0, 1), (1, 0)):
        tr, tst = folds[a], folds[b]
        ch, _ = greedy_menu(L, tr, [0] + [int(j) + 1 for j in tr], K_cv)
        chosen_cv.append(ch)
        for k in range(K_cv):
            num_cv[k] += float((L[tst, 0] - L[np.ix_(tst, ch[:k + 1])].min(1)).sum())
        den_cv += float((L[tst, 0] - L[tst, tst + 1]).sum())
    frac_cv = [x / den_cv if den_cv > 0 else float("nan") for x in num_cv]
    return base_b, own_b, K, chosen_in, frac_in, K_cv, chosen_cv, frac_cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="search_sd.py output")
    ap.add_argument("--coco-captions", required=True)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--image-dir", default=None, help="--render: PNG directory")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--decode-chunk", type=int, default=4)
    ap.add_argument("--cache-cap", type=int, default=256)
    ap.add_argument("--menu-max", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--always-b1", action="store_true",
                    help="--restart: also restart N+1 when N did not improve (for testing)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.render == args.restart:
        raise SystemExit("pass exactly one of --render or --restart")
    if args.render and not args.image_dir:
        raise SystemExit("--render needs --image-dir")

    src = np.load(args.npz)
    B, n, T = int(src["budget"]), int(src["n_images"]), int(src["max_t"])
    cfg, seed = float(src["cfg"]), int(src["seed"])
    rng_prompts, ref_steps = int(src["rng_prompts"]), int(src["ref_steps"])
    sweeps, n_cand = int(src["sweeps"]), int(src["n_candidates"])
    model, attn = str(src["model"]), str(src["attn"])
    fB, fB1 = src["fixed_B"].astype(np.int64), src["fixed_B1"].astype(np.int64)
    per_sched = src["per_sched"].astype(np.int64)
    per_img, per_base = src["per_img"], src["D_fixed_unbatched"]

    import lpips
    from diffusers import StableDiffusionPipeline

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    t_start = time.time()
    dc = args.decode_chunk

    mode = "RENDER" if args.render else "RESTART"
    print("=" * 74)
    print(f"SD 1.5 POST-HOC {mode}  source {args.npz}")
    print(f"  B={B} n={n} cfg={cfg} seed={seed} ref_steps={ref_steps} sweeps={sweeps} "
          f"n_cand={n_cand} attn={attn} batch={args.batch} decode_chunk={dc}")
    print("=" * 74)

    img_ids, ann_ids, prompts = coco_prompts(args.coco_captions, n, max(rng_prompts, n), seed)
    seeds = [seed * 100000 + i for i in range(n)]
    same = (np.array_equal(np.asarray(img_ids, dtype=np.int64), src["coco_image_ids"])
            and np.array_equal(np.asarray(ann_ids, dtype=np.int64), src["coco_annotation_ids"])
            and [str(p) for p in src["prompts"]] == prompts
            and np.array_equal(np.asarray(seeds, dtype=np.int64), src["noise_seeds"]))
    print(f"  prompts, annotation ids and noise seeds match the source: {same}")
    if not same:
        raise SystemExit("FAILED: prompts or seeds differ from the source run")

    # ---- rollout setup: copied from search_sd.py ----
    pipe = StableDiffusionPipeline.from_pretrained(
        model, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
        feature_extractor=None, requires_safety_checker=False, local_files_only=True)
    unet = pipe.unet.to(device).eval()
    vae = pipe.vae.to(device).eval()
    te = pipe.text_encoder.to(device).eval()
    tok = pipe.tokenizer
    if attn == "classic":
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

    t0 = time.time()
    ref = np.linspace(T - 1, 0, ref_steps + 1).round().astype(np.int64)
    ref_img = decode(Roller(ddim_step, z_T, ctx, args.batch, 0).run(ref, use_cache=False), dc)
    print(f"  reference generated at {ref_steps} guided steps in {time.time() - t0:.0f}s")

    def batched_img(s):
        return decode(Roller(ddim_step, z_T, ctx, args.batch, 0).run(s, use_cache=False), dc)

    def one_img(s, i):
        r = Roller(ddim_step, z_T[i:i + 1], ctx[i:i + 1], 1, 0)
        return decode(r.run(s, use_cache=False), 1)

    def dev(a, b):
        return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())

    if args.restart:
        run_restart(args, src, B, n, T, sweeps, n_cand, fB, per_img, batched_img, one_img,
                    loss_fn, ref_img, ddim_step, z_T, ctx, decode, dev, t_start)
        return

    # ---------------------------------- render ----------------------------------
    import piq
    from PIL import Image
    dists_fn = piq.DISTS(reduction="none").to(device).eval()

    def to01(x):
        return (x.clamp(-1, 1) + 1) / 2

    def metrics(img, refs, chunk):
        lp = lpips_to(loss_fn, img, refs, chunk)
        ps, di = [], []
        for i in range(0, img.shape[0], chunk):
            a, b = to01(img[i:i + chunk]), to01(refs[i:i + chunk])
            mse = (a.double() - b.double()).pow(2).flatten(1).mean(1)
            ps.append(10.0 * torch.log10(1.0 / mse.clamp(min=1e-20)))
            di.append(dists_fn(a, b).flatten().double())
        return (np.asarray(lp, np.float64), torch.cat(ps).cpu().numpy(),
                torch.cat(di).cpu().numpy())

    def to_u8(x):
        return (to01(x) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    saved = {"ref": to_u8(ref_img)}
    named = {}
    for name, s in ((f"uniform_B{B}", uniform(T, B)), (f"shared_B{B}", fB),
                    (f"shared_B{B + 1}", fB1)):
        img = batched_img(s)
        named[name] = metrics(img, ref_img, dc)
        saved[name] = to_u8(img)
    print(f"  batched rollouts done ({time.time() - t_start:.0f}s)", flush=True)

    own = np.zeros((3, n))       # per-prompt path, own schedule: lpips, psnr, dists
    base = np.zeros((3, n))      # per-prompt path, shared N schedule
    own_u8 = []
    for i in range(n):
        img = one_img(per_sched[i], i)
        own[:, i] = [m[0] for m in metrics(img, ref_img[i:i + 1], 1)]
        own_u8.append(to_u8(img)[0])
        img = one_img(fB, i)
        base[:, i] = [m[0] for m in metrics(img, ref_img[i:i + 1], 1)]
    saved[f"perprompt_B{B}"] = np.stack(own_u8)
    print(f"  per-prompt rollouts done ({time.time() - t_start:.0f}s)", flush=True)

    devs = {
        "shared N (batched) vs stored D_fixed": dev(named[f"shared_B{B}"][0], src["D_fixed"]),
        "shared N+1 (batched) vs stored D_fixed_B1":
            dev(named[f"shared_B{B + 1}"][0], src["D_fixed_B1"]),
        "per-prompt N vs stored per_img": dev(own[0], per_img),
        "shared N (per-prompt path) vs stored D_fixed_unbatched": dev(base[0], per_base),
    }
    for k, v in devs.items():
        print(f"  reproduce  {k:56s} max |dev| {v:.3e}  {'PASS' if v <= args.tol else 'FAIL'}")
    if not all(v <= args.tol for v in devs.values()):
        raise SystemExit("FAILED: stored LPIPS values not reproduced; nothing written")

    # ---- PSNR / DISTS re-score, gains on the search's own paths ----
    print("\n  metric  gain(per-prompt)  step value  R      shared-vs-uniform  (in steps)")
    res = {}
    for k, (mname, hb) in enumerate((("lpips", False), ("psnr", True), ("dists", False))):
        sgn = -1.0 if hb else 1.0
        u_b, f_b, f1_b = (named[f"uniform_B{B}"][k], named[f"shared_B{B}"][k],
                          named[f"shared_B{B + 1}"][k])
        gain = sgn * (float(base[k].mean()) - float(own[k].mean()))
        step = sgn * (float(f_b.mean()) - float(f1_b.mean()))
        sh = sgn * (float(u_b.mean()) - float(f_b.mean()))
        r = gain / step if step > 0 else float("nan")
        r_sh = sh / step if step > 0 else float("nan")
        res[mname] = (gain, step, r, sh, r_sh)
        print(f"  {mname:6s}  {gain:+.6f}        {step:+.6f}   {r:6.3f}  {sh:+.6f}        "
              f"{r_sh:6.3f}")
    print(f"  stored R (LPIPS) {float(src['gain']) / float(src['step_value']):.4f}")

    # ---- schedule menu: column 0 = shared N, column j+1 = prompt j's own schedule ----
    M = np.zeros((3, n, n + 1))
    for k in range(3):
        M[k, :, 0] = named[f"shared_B{B}"][k]
    for j in range(n):
        lp, ps, di = metrics(batched_img(per_sched[j]), ref_img, dc)
        M[0, :, j + 1], M[1, :, j + 1], M[2, :, j + 1] = lp, ps, di
        if (j + 1) % 8 == 0 or j + 1 == n:
            print(f"    menu matrix {j + 1}/{n} schedules ({time.time() - t_start:.0f}s)",
                  flush=True)
    base_b, own_b, K, chosen_in, frac_in, K_cv, chosen_cv, frac_cv = menu_analysis(
        M[0], args.menu_max)
    den_in = base_b - own_b
    print("\n  schedule menu (LPIPS, batched path; best entry chosen per prompt)")
    print(f"    shared N {base_b:.6f}   own schedules {own_b:.6f}   gain {den_in:.6f}")
    for k in range(K):
        cv = f"{100 * frac_cv[k]:6.1f}%" if k < K_cv else "   n/a"
        tag = "shared" if chosen_in[k] == 0 else f"prompt {chosen_in[k] - 1}"
        print(f"    K={k + 1}  in-sample {100 * frac_in[k]:6.1f}%   2-fold {cv}   "
              f"(added: {tag})")

    for name, arr in saved.items():
        d = os.path.join(args.image_dir, name)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            Image.fromarray(arr[i]).save(os.path.join(d, f"{i:02d}.png"))
    print(f"  saved {len(saved)} x {n} PNGs under {args.image_dir}")

    def pct(f, k):
        return 100 * f[k] if k < len(f) else float("nan")

    print(f"\nSUMMARY sd_posthoc B={B} n={n} cfg={cfg} "
          f"repro_max_dev={max(devs.values()):.3e} "
          f"R_lpips={res['lpips'][2]:.4f} R_psnr={res['psnr'][2]:.4f} "
          f"R_dists={res['dists'][2]:.4f} Rsh_lpips={res['lpips'][4]:.4f} "
          f"menu2={pct(frac_in, 1):.1f}% menu4={pct(frac_in, 3):.1f}% "
          f"menu2_cv={pct(frac_cv, 1):.1f}% menu4_cv={pct(frac_cv, 3):.1f}%")
    np.savez(args.out, source=np.asarray(args.npz), budget=np.int64(B), n_images=np.int64(n),
             cfg=np.float64(cfg), tol=np.float64(args.tol),
             repro_keys=np.asarray(list(devs)), repro_devs=np.asarray(list(devs.values())),
             metrics=np.asarray(["lpips", "psnr", "dists"]),
             uniform_B=np.stack(named[f"uniform_B{B}"]),
             shared_B=np.stack(named[f"shared_B{B}"]),
             shared_B1=np.stack(named[f"shared_B{B + 1}"]),
             perprompt_own=own, perprompt_sharedB=base,
             gain_step_R_shgain_Rsh=np.asarray([res[m] for m in ("lpips", "psnr", "dists")]),
             menu_columns=np.concatenate([fB[None], per_sched], 0), menu_M=M,
             menu_chosen_in=np.asarray(chosen_in), menu_frac_in=np.asarray(frac_in),
             menu_chosen_cv=np.asarray(chosen_cv), menu_frac_cv=np.asarray(frac_cv),
             image_dir=np.asarray(args.image_dir), batch=np.int64(args.batch),
             decode_chunk=np.int64(dc))
    print(f"  saved {args.out}  (wall {time.time() - t_start:.0f}s)")


def run_restart(args, src, B, n, T, sweeps, n_cand, fB, per_img, batched_img, one_img,
                loss_fn, ref_img, ddim_step, z_T, ctx, decode, dev, t_start):
    dc = args.decode_chunk
    D_fB = lpips_to(loss_fn, batched_img(fB), ref_img, dc)
    d0 = dev(D_fB, src["D_fixed"])
    print(f"  reproduce  shared N (batched) vs stored D_fixed  max |dev| {d0:.3e}  "
          f"{'PASS' if d0 <= args.tol else 'FAIL'}")
    if d0 > args.tol:
        raise SystemExit("FAILED: stored LPIPS values not reproduced; nothing written")
    fB_mean = float(D_fB.mean())
    fB1_mean = float(src["D_fixed_B1"].mean())
    gain0, step0 = float(src["gain"]), float(src["step_value"])

    shared = Roller(ddim_step, z_T, ctx, args.batch, args.cache_cap)

    def obj(s):
        return float(lpips_to(loss_fn, decode(shared.run(s), dc), ref_img, dc).mean())

    def search(B_, label):
        init = quadratic(T, B_)
        t0 = time.time()
        f, f_mean, f_start, e, c = coord_descent(obj, init, sweeps, n_cand)
        print(f"\n  restart at B={B_} ({label}) from quadratic {init.tolist()}  "
              f"start {f_start:.6f}")
        print(f"    found {f.tolist()}  mean LPIPS {f_mean:.6f}  ({e} evals, "
              f"{'converged' if c else 'HIT SWEEP CAP'}, {time.time() - t0:.0f}s)", flush=True)
        return init, f, f_mean, e, c

    init, fR, fR_mean, eR, cR = search(B, "N")
    print(f"    stored shared N {fB.tolist()}  mean LPIPS {fB_mean:.6f}")
    improved = fR_mean < fB_mean - 1e-9
    print(f"    restart {'BEATS' if improved else 'does not beat'} the stored shared schedule "
          f"by {fB_mean - fR_mean:+.6f}")

    gain2, bestN_mean = gain0, fB_mean
    base_R = np.full(n, np.nan)
    if improved or args.always_b1:
        for i in range(n):
            base_R[i] = float(lpips_to(loss_fn, one_img(fR, i), ref_img[i:i + 1], 1)[0])
        if improved:
            per_img2 = np.minimum(per_img, base_R)
            gain2, bestN_mean = float(base_R.mean() - per_img2.mean()), fR_mean
    init1 = fR1 = np.zeros(0, np.int64)
    fR1_mean, eR1, cR1, bestN1_mean = float("nan"), 0, True, fB1_mean
    if improved or args.always_b1:
        init1, fR1, fR1_mean, eR1, cR1 = search(B + 1, "N+1")
        print(f"    stored shared N+1 mean LPIPS {fB1_mean:.6f}")
        bestN1_mean = min(fB1_mean, fR1_mean)
    step2 = bestN_mean - bestN1_mean
    R0 = gain0 / step0 if step0 > 0 else float("nan")
    R2 = gain2 / step2 if step2 > 0 else float("nan")
    print(f"\n  stored        gain {gain0:.6f}  step value {step0:.6f}  R {R0:.4f}")
    print(f"  best of two   gain {gain2:.6f}  step value {step2:.6f}  R {R2:.4f}")
    print(f"\nSUMMARY sd_restart B={B} n={n} init=quadratic stored_shared={fB_mean:.6f} "
          f"restart_shared={fR_mean:.6f} improved={int(improved)} conv={int(cR)} "
          f"restart_B1={'run' if fR1.size else 'skipped'} R_stored={R0:.4f} "
          f"R_best_of_two={R2:.4f}")
    np.savez(args.out, source=np.asarray(args.npz), budget=np.int64(B), n_images=np.int64(n),
             init_B=init, restart_B=fR, restart_B_mean=np.float64(fR_mean),
             restart_B_evals=np.int64(eR), restart_B_conv=np.int8(cR),
             stored_B_mean=np.float64(fB_mean), improved=np.int8(improved),
             D_restart_B_unbatched=base_R, init_B1=init1, restart_B1=fR1,
             restart_B1_mean=np.float64(fR1_mean), restart_B1_evals=np.int64(eR1),
             restart_B1_conv=np.int8(cR1), stored_B1_mean=np.float64(fB1_mean),
             gain_stored=np.float64(gain0), step_stored=np.float64(step0),
             gain_best2=np.float64(gain2), step_best2=np.float64(step2),
             R_stored=np.float64(R0), R_best2=np.float64(R2),
             repro_dev=np.float64(d0), tol=np.float64(args.tol))
    print(f"  saved {args.out}  (wall {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
