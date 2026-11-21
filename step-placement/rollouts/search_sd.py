"""Shared vs per-prompt DDIM schedule search for Stable Diffusion 1.5 on COCO-2017 captions.

The algorithm and search settings are those of search_vs_sample.py: coordinate_descent,
candidates and uniform are IMPORTED from search_vs_sample_v2.py, whose default path was
verified array-for-array against the frozen file. Searched shared N, searched shared N+1,
searched per-sample N, paired LPIPS-AlexNet against a 200-step DDIM reference.

Pinned for SD:
  prompts     64 distinct COCO-2017 val images: default_rng(seed).choice over the image ids
              sorted ascending, without replacement, drawing --rng-prompts (64) and keeping
              the first --n-prompts, so a 32-prompt run is the first half of the 64. Each
              image contributes its FIRST caption (lowest annotation id). Image ids,
              annotation ids, captions and noise seeds are saved, plus a TSV.
  noise       one initial latent per prompt, torch CPU generator seeded seed*100000 + index,
              shared by every schedule and by the reference.
  guidance    classifier-free, scale 7.5, empty negative prompt. One guided step (one
              batched unconditional+conditional UNet pass) counts as ONE evaluation.
  sampler     DDIM eta=0 on SD's own alphas_cumprod, the integer grid the LDM runs use
              (linspace(999, 0, N+1).round()), and adapter_compvis's update: the last step
              lands at alphas_cumprod[0] and is decoded without clamping, as the LDM path
              does. UNet, VAE and text encoder in fp16; latents, the DDIM update and the
              CFG combination in fp32; LPIPS in fp32 at 512x512.
  determinism cudnn deterministic, deterministic algorithms requested, classic (matmul +
              softmax) attention in UNet and VAE because torch 2.0.1's SDPA memory-efficient
              kernel is non-deterministic, fixed batch composition (batched path: chunks of
              --batch prompts; per-prompt path: one).
              Before any search the run checks, on its own prompts, that re-scoring is
              BITWISE identical and that a prefix-cached rollout equals a full one. It
              aborts otherwise, so the 1e-9 acceptance threshold keeps its meaning.
  prefix cache moving node j leaves nodes 0..j-1 unchanged, so the latent there is reused.
              Exact (same ops on the same inputs); it only skips UNet calls. `calls` counts
              evaluations as the uncached protocol does; `calls_run` counts those executed.

  python3 utils/search_sd.py --check   ...   gates only, 2 prompts, minutes
  python3 utils/search_sd.py --probe   ...   references + one timed sweep -> projection
  python3 utils/search_sd.py           ...   the search
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from reward_structure import lpips_to  # noqa: E402
from search_vs_sample_v2 import coord_descent, uniform  # noqa: E402

# evaluation counts of the converged church N=8, n=64, sweeps=10 search (runs2), used
# only to extrapolate the probe: shared N, shared N+1, mean per image. "high" takes the
# largest of the church n=64, church n=256 and bedroom n=64 runs.
COUNTS = {"church": (499, 476, 316.5), "high": (499, 662, 364.5)}


def coco_prompts(path, n_keep, n_draw, seed):
    with open(path) as f:
        ann = json.load(f)
    img_ids = sorted({int(im["id"]) for im in ann["images"]})
    first = {}
    for a in sorted(ann["annotations"], key=lambda a: int(a["id"])):
        first.setdefault(int(a["image_id"]), a)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(img_ids), size=n_draw, replace=False)[:n_keep]
    ids = [img_ids[int(k)] for k in pick]
    anns = [first[i] for i in ids]
    print(f"  COCO: {len(img_ids)} val images, {len(ann['annotations'])} captions; "
          f"kept {len(ids)} distinct images (drew {n_draw}, seed {seed})")
    return ids, [int(a["id"]) for a in anns], [a["caption"].strip() for a in anns]


class Roller:
    """Batched DDIM rollouts over a fixed set of prompts, with an exact prefix cache."""

    def __init__(self, step, z0, ctx, chunk, cap):
        self.step, self.z0, self.ctx, self.chunk, self.cap = step, z0, ctx, chunk, cap
        self.cache = OrderedDict()
        self.steps_run = 0          # guided steps executed, summed over prompts

    def run(self, sched, use_cache=True):
        sched = [int(v) for v in sched]
        z, start = self.z0, 0
        if use_cache:
            # latent at node k depends only on sched[0..k]; the final node is never cached
            for k in range(len(sched) - 2, 0, -1):
                key = tuple(sched[:k + 1])
                if key in self.cache:
                    self.cache.move_to_end(key)
                    z, start = self.cache[key], k
                    break
        for k in range(start, len(sched) - 1):
            z = torch.cat([self.step(z[i:i + self.chunk], sched[k], sched[k + 1],
                                     self.ctx[i:i + self.chunk])
                           for i in range(0, z.shape[0], self.chunk)], dim=0)
            self.steps_run += z.shape[0]
            if use_cache and k + 1 <= len(sched) - 2:
                key = tuple(sched[:k + 2])
                self.cache[key] = z
                self.cache.move_to_end(key)
                while len(self.cache) > self.cap:
                    self.cache.popitem(last=False)
        return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-captions", required=True, help="captions_val2017.json")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--rng-prompts", type=int, default=64)
    ap.add_argument("--sweeps", type=int, default=10)
    ap.add_argument("--n-cand", type=int, default=12)
    ap.add_argument("--ref-steps", type=int, default=200)
    ap.add_argument("--max-t", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8, help="prompts per UNet call (x2 for CFG)")
    ap.add_argument("--decode-chunk", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)
    # classic: torch 2.0.1's SDPA memory-efficient kernel is non-deterministic (it warns so
    # and failed the bitwise gates on V100), so the default is the plain matmul-softmax path
    ap.add_argument("--attn", choices=("sdpa", "classic"), default="classic")
    ap.add_argument("--cache-cap", type=int, default=256)
    ap.add_argument("--check", action="store_true", help="run the gates only, then exit")
    ap.add_argument("--probe", action="store_true", help="time one sweep and project")
    ap.add_argument("--probe-json", default=None)
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import lpips
    from diffusers import StableDiffusionPipeline

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    T, B, n = args.max_t, args.budget, args.n_prompts
    t_start = time.time()

    mode = "CHECK" if args.check else "PROBE" if args.probe else "SEARCH"
    print("=" * 74)
    print(f"SD 1.5 {mode}  B={B} guided steps   n={n} COCO prompts   cfg={args.cfg}")
    print("=" * 74)

    img_ids, ann_ids, prompts = coco_prompts(args.coco_captions, n, max(args.rng_prompts, n),
                                             args.seed)
    for i in range(min(3, n)):
        print(f"    prompt {i}: [{img_ids[i]}] {prompts[i]}")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
        feature_extractor=None, requires_safety_checker=False, local_files_only=True)
    unet = pipe.unet.to(device).eval()
    vae = pipe.vae.to(device).eval()
    te = pipe.text_encoder.to(device).eval()
    tok = pipe.tokenizer
    if args.attn == "classic":
        from diffusers.models.attention_processor import AttnProcessor
        unet.set_attn_processor(AttnProcessor())
        vae.set_attn_processor(AttnProcessor())
    alphas = pipe.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
    assert alphas.shape[0] == T, alphas.shape
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, T, dtype=torch.float32) ** 2
    expect = torch.cumprod(1.0 - betas, 0).to(device)
    assert torch.allclose(alphas, expect, rtol=1e-5, atol=1e-7), "unexpected noise schedule"
    sf = float(vae.config.scaling_factor)
    print(f"  model {args.model}  fp16  attn={args.attn}  scaling_factor {sf}")
    print(f"  alphas_cumprod[0]={float(alphas[0]):.6f}  [999]={float(alphas[T - 1]):.6f}")

    def embed(text):
        ids = tok([text], padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(device)
        return te(ids)[0]

    ctx = torch.cat([embed(p) for p in prompts], dim=0)          # (n, 77, 768) fp16
    unc = embed("")                                                # (1, 77, 768) fp16

    seeds = [args.seed * 100000 + i for i in range(n)]
    z_T = torch.cat([torch.randn(1, 4, 64, 64, generator=torch.Generator("cpu").manual_seed(s))
                     for s in seeds], dim=0).to(device)            # fp32

    def ddim_step(z, t_s, t_e, c):
        b = z.shape[0]
        tt = torch.full((2 * b,), int(t_s), device=device, dtype=torch.long)
        e = unet(torch.cat([z, z], 0).half(), tt,
                 encoder_hidden_states=torch.cat([unc.expand(b, -1, -1), c], 0)).sample.float()
        e_u, e_c = e.chunk(2)
        eps = e_u + args.cfg * (e_c - e_u)
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
    ref = np.linspace(T - 1, 0, args.ref_steps + 1).round().astype(np.int64)
    ref_roll = Roller(ddim_step, z_T, ctx, args.batch, 0)
    ref_img = decode(ref_roll.run(ref, use_cache=False), args.decode_chunk)
    t_ref = time.time() - t0
    print(f"  reference generated at {args.ref_steps} guided steps (DDIM) in {t_ref:.0f}s")

    def score_batched(roller, sched, use_cache=True):
        return lpips_to(loss_fn, decode(roller.run(sched, use_cache), args.decode_chunk),
                        ref_img, args.decode_chunk)

    def score_one(roller, sched, i, use_cache=True):
        z = roller.run(sched, use_cache)
        return float(lpips_to(loss_fn, decode(z, 1), ref_img[i:i + 1], 1)[0])

    # ---- gates: bitwise repeatability and exact prefix caching, on these prompts ----
    u = uniform(T, B)
    trial = list(u)
    j = len(u) // 2
    trial[j] = int(u[j]) + 1 if int(u[j]) + 1 < int(u[j - 1]) else int(u[j]) - 1
    fresh = lambda: Roller(ddim_step, z_T, ctx, args.batch, args.cache_cap)  # noqa: E731
    a1 = score_batched(fresh(), u, use_cache=False)
    a2 = score_batched(fresh(), u, use_cache=False)
    warm = fresh()
    score_batched(warm, u)
    c_cached = score_batched(warm, trial)
    c_full = score_batched(fresh(), trial, use_cache=False)
    r1 = Roller(ddim_step, z_T[:1], ctx[:1], 1, args.cache_cap)
    s1 = score_one(r1, u, 0, use_cache=False)
    s2 = score_one(Roller(ddim_step, z_T[:1], ctx[:1], 1, 0), u, 0, use_cache=False)
    score_one(r1, u, 0)
    s3 = score_one(r1, trial, 0)
    s4 = score_one(Roller(ddim_step, z_T[:1], ctx[:1], 1, 0), trial, 0, use_cache=False)
    gates = {
        "batched repeat bitwise": bool(np.array_equal(a1, a2)),
        "batched cached == full": bool(np.array_equal(c_cached, c_full)),
        "per-prompt repeat bitwise": s1 == s2,
        "per-prompt cached == full": s3 == s4,
    }
    for k, v in gates.items():
        print(f"  gate  {k:28s} {'PASS' if v else 'FAIL'}")
    print(f"  batched-vs-per-prompt gap on prompt 0, uniform: {abs(float(a1[0]) - s1):.3e}")
    if not all(gates.values()):
        raise SystemExit("FAILED: not bitwise deterministic. Do not loosen the acceptance "
                         "threshold silently; report this output.")
    if args.check:
        print(f"\nCHECK PASSED  ({time.time() - t_start:.0f}s)")
        return

    calls = [0]
    shared = fresh()

    def lpips_all(sched, count=True):
        if count:
            calls[0] += n * (len(sched) - 1)
        return score_batched(shared, sched)

    if args.probe:
        t0 = time.time()
        _, _, _, e_sw, _ = coord_descent(lambda s: float(lpips_all(s).mean()), u, 1,
                                         args.n_cand)
        sec_shared = (time.time() - t0) / e_sw
        r = Roller(ddim_step, z_T[:1], ctx[:1], 1, args.cache_cap)
        t0 = time.time()
        _, _, _, e_one, _ = coord_descent(lambda s: score_one(r, s, 0), u, 1, args.n_cand)
        sec_one = (time.time() - t0) / e_one
        print(f"\n  one shared sweep: {e_sw} evals, {sec_shared:.2f}s per eval over {n} prompts")
        print(f"  one per-prompt sweep: {e_one} evals, {sec_one:.3f}s per eval")
        proj = {}
        for name, (E1, E2, EP) in COUNTS.items():
            for m in (n, n // 2):
                sh = (E1 + 2) * sec_shared * m / n + (E2 + 1) * sec_shared * m / n * (B + 1) / B
                pp = m * EP * sec_one
                proj[f"{name}_n{m}"] = (t_ref * m / n + sh + pp) / 3600
        for k, v in proj.items():
            print(f"  PROJECTION {k:12s} {v:6.1f} h")
        if args.probe_json:
            with open(args.probe_json, "w") as f:
                json.dump({"sec_shared": sec_shared, "sec_one": sec_one, "t_ref": t_ref,
                           "n": n, "budget": B, "hours": proj}, f, indent=1)
        print(f"\nPROBE DONE  ({time.time() - t_start:.0f}s)")
        return

    print(f"\n1  SEARCH THE BEST FIXED SCHEDULE AT B={B}   (objective: mean over prompts)")
    t0 = time.time()
    fB, fB_mean, _, e1, c1 = coord_descent(lambda s: float(lpips_all(s).mean()),
                                           uniform(T, B), args.sweeps, args.n_cand)
    uni_mean = float(lpips_all(uniform(T, B)).mean())
    print(f"    uniform  {uniform(T, B).tolist()}  mean LPIPS {uni_mean:.6f}")
    print(f"    searched {fB.tolist()}  mean LPIPS {fB_mean:.6f}"
          f"   ({100 * (uni_mean - fB_mean) / uni_mean:+.2f}% vs uniform,"
          f" {e1} evals, {'converged' if c1 else 'HIT SWEEP CAP'}, {time.time() - t0:.0f}s)",
          flush=True)
    D_fB = lpips_all(fB)
    assert abs(float(D_fB.mean()) - fB_mean) < 1e-9, (
        f"NON-DETERMINISTIC: {D_fB.mean()!r} != {fB_mean!r}.")

    print(f"\n2  SEARCH THE BEST FIXED SCHEDULE AT B={B + 1}   (the yardstick, also searched)")
    t0 = time.time()
    fB1, fB1_mean, _, e2, c2 = coord_descent(lambda s: float(lpips_all(s).mean()),
                                             uniform(T, B + 1), args.sweeps, args.n_cand)
    print(f"    searched {fB1.tolist()}  mean LPIPS {fB1_mean:.6f}"
          f"   ({e2} evals, {'converged' if c2 else 'HIT SWEEP CAP'}, {time.time() - t0:.0f}s)",
          flush=True)
    step_value = fB_mean - fB1_mean
    D_fB1 = lpips_all(fB1, count=False)
    assert abs(float(D_fB1.mean()) - fB1_mean) < 1e-9, (
        f"NON-DETERMINISTIC at B+1: {D_fB1.mean()!r} != {fB1_mean!r}.")

    print(f"\n3  SEARCH A SCHEDULE PER PROMPT AT B={B}   (the oracle, same routine)")
    t0 = time.time()
    per_img = np.zeros(n)
    per_base = np.zeros(n)
    per_sched = np.zeros((n, B + 1), dtype=np.int64)
    n_capped = 0
    steps_per = 0
    for i in range(n):
        r = Roller(ddim_step, z_T[i:i + 1], ctx[i:i + 1], 1, args.cache_cap)

        def one(s, r=r, i=i):
            calls[0] += len(s) - 1
            return score_one(r, s, i)

        s_i, v_i, b_i, e_i, c_i = coord_descent(one, fB, args.sweeps, args.n_cand)
        steps_per += r.steps_run
        per_img[i], per_base[i], per_sched[i] = v_i, b_i, s_i
        n_capped += not c_i
        print(f"    prompt {i:3d}  {b_i:.6f} -> {v_i:.6f}"
              f"  ({100 * (b_i - v_i) / max(b_i, 1e-12):+5.2f}%, {e_i} evals"
              f"{'' if c_i else ', CAPPED'}, {time.time() - t0:.0f}s)", flush=True)

    gain = float(per_base.mean() - per_img.mean())
    n_improved = int((per_img < per_base - 1e-9).sum())
    uniq = len({tuple(r) for r in per_sched.tolist()})
    path_gap = float(np.abs(per_base - D_fB).max())
    verdict = ("negligible" if path_gap < 0.01 * max(gain, 1e-12)
               else "COMPARABLE TO THE GAIN -- the matched baseline is load-bearing here")
    print(f"\n    batched-vs-unbatched gap on the SAME schedule: max {path_gap:.3e}"
          f"  ({verdict})")
    calls_run = shared.steps_run + steps_per

    print("\n" + "=" * 74)
    print(f"  best FIXED (searched)     : {per_base.mean():.6f}")
    print(f"  per-prompt (searched)     : {per_img.mean():.6f}")
    print(f"  SEARCHED ORACLE GAIN      : {gain:.6f}")
    print(f"  one extra step (searched) : {step_value:.6f}")
    if step_value > 0:
        print(f"  -> worth {gain / step_value:.3f} extra guided steps")
    print(f"  prompts where search beat the fixed schedule : {n_improved}/{n}")
    print(f"  distinct per-prompt schedules found          : {uniq}/{n}")
    if n_capped:
        print(f"  WARNING: {n_capped}/{n} per-prompt searches hit the sweep cap")
    print(f"\n  guided steps: {calls[0]:,} by the uncached count, {calls_run:,} executed"
          f"  (wall {time.time() - t_start:.0f}s)")
    print(f"\nSUMMARY search_sd dataset=sd15_coco budget={B} n={n} "
          f"searched_gain={gain:.6f} step_value={step_value:.6f} "
          f"ratio={gain / step_value if step_value > 0 else float('nan'):.4f} "
          f"n_improved={n_improved} uniq_schedules={uniq} conv_B={int(c1)} conv_B1={int(c2)} "
          f"n_capped={n_capped} calls={calls[0]} cfg={args.cfg}")

    if args.save_npz:
        np.savez(args.save_npz, fixed_B=fB, fixed_B1=fB1, per_sched=per_sched,
                 per_img=per_img, D_fixed=D_fB, gain=gain, step_value=step_value,
                 D_fixed_B1=D_fB1, D_fixed_unbatched=per_base,
                 n_improved=np.int64(n_improved), uniq_schedules=np.int64(uniq),
                 conv_B=np.int8(c1), conv_B1=np.int8(c2),
                 n_capped=np.int64(n_capped), calls=np.int64(calls[0]),
                 calls_run=np.int64(calls_run), dataset=np.asarray("sd15_coco"),
                 budget=np.int64(B), n_images=np.int64(n), sweeps=np.int64(args.sweeps),
                 n_candidates=np.int64(args.n_cand), seed=np.int64(args.seed),
                 rng_prompts=np.int64(args.rng_prompts), ref_steps=np.int64(args.ref_steps),
                 max_t=np.int64(T), sampler=np.asarray("ddim"), cfg=np.float64(args.cfg),
                 attn=np.asarray(args.attn), model=np.asarray(args.model),
                 coco_image_ids=np.asarray(img_ids, dtype=np.int64),
                 coco_annotation_ids=np.asarray(ann_ids, dtype=np.int64),
                 prompts=np.asarray(prompts), noise_seeds=np.asarray(seeds, dtype=np.int64))
        tsv = os.path.splitext(args.save_npz)[0] + "_prompts.tsv"
        with open(tsv, "w") as f:
            f.write("index\tcoco_image_id\tcoco_annotation_id\tnoise_seed\tcaption\n")
            for i in range(n):
                f.write(f"{i}\t{img_ids[i]}\t{ann_ids[i]}\t{seeds[i]}\t{prompts[i]}\n")
        print(f"  saved {args.save_npz} and {tsv}")


if __name__ == "__main__":
    main()
