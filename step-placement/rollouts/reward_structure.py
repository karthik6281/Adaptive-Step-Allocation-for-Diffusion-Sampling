"""Is there anything for an adaptive policy to exploit? Oracle upper bound, no policy needed.

Everything in phase0_init_check.py asks whether the policy CAN respond to the image. This
asks whether responding is WORTH anything, which no amount of architecture work can settle.

Decompose the quality of image i under schedule s at fixed NFE:

    D[i,s] = mu + alpha_i + beta_s + gamma_is

    alpha_i  some images are harder                -> NOT exploitable, the policy does not
                                                      choose which image it gets
    beta_s   some schedules are better on average  -> exploitable by a FIXED schedule,
                                                      no adaptivity required
    gamma_is interaction                           -> THE ONLY THING ADAPTIVITY BUYS

So "does the reward vary across images" is the wrong question: that is alpha, and it is
worthless. The decisive quantity is

    oracle gain = min_s E_i[ D[i,s] ]  -  E_i[ min_s D[i,s] ]
                  (best FIXED schedule)   (oracle PER-IMAGE schedule)

an upper bound on what ANY adaptive policy can achieve at that budget -- RL, oracle,
anything. If it is ~0 the premise is dead regardless of every architecture question.

Two further numbers decide what a positive result is worth:

  BUDGET YARDSTICK  the oracle gain compared against simply spending one more NFE. If
                    adaptivity at budget B buys less than the step from B to B+1, it is
                    not worth having.
  LEARNABILITY      the oracle knows each image's best schedule. A policy must PREDICT it
                    from what it observes. Nearest-neighbour transfer on held-out images
                    gives a lower bound.

WHICH OF THESE IS THE ACTUAL TEST. The oracle gain is NOT, and a confidence interval on it
is worthless: min_s E_i[D] >= E_i[min_s D] holds by Jensen for ANY matrix, so the oracle
gain is >= 0 always. On pure iid noise with M=24 schedules it averages 1.7 sd. It is a
CEILING. The decisive test is the LEARNABLE gain against a PERMUTATION NULL that shuffles
which image each feature vector belongs to: that null destroys the image-schedule
association while preserving every marginal, so beating it means the best schedule is
genuinely predictable from the observation.

Generation is deterministic (DDIM eta=0), so D[i,s] carries no measurement noise and the
interaction is reproducible by construction; the uncertainty is finite-image sampling.

WHICH BUDGET TO TEST. The schedule-optimisation literature reports its gains in the
FEW-STEP regime, so a sweep should be anchored there rather than at a converged budget:
  AutoDiffusion (ICCV'23)  headline is FID 17.86 vs DDIM 138.66 at FOUR steps
  Align Your Steps (2024)  gains "especially in the few-step synthesis regime"
  DPM-Solver++ (2022)      15-20 steps for what DDIM needs 100-250 for
The project runs DDIM at NFE ~110-220 (inference200/400 at ~45% skips), which is both
saturated for schedule choice AND ~5-10x more NFE than a modern solver needs for the same
quality. Adaptivity is competing for tenths of an NFE against a solver swap worth ~100.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Same mapping and lookup order run_rl.py uses, so this cannot silently load a different
# checkpoint than training did.
LDM_NAME = {
    "church256": "lsun_churches256",
    "bedroom256": "lsun_beds256",
    "celeb256": "celeba256",
    "ffhq256": "ffhq256",
    "cin256": "cin256",
}


def resolve_paths(dataset, config_override=None, ckpt_override=None):
    name = LDM_NAME[dataset]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # taed/
    cfg = config_override or os.path.join(
        here, "..", "latent-diffusion", "models", "ldm", name, "config.yaml")
    if ckpt_override:
        return cfg, ckpt_override
    root = os.environ.get("TAED_LDM_ROOT", "/home/toolkit/ldm_models")
    for cand in (os.path.join(root, name, "model.ckpt"),
                 os.path.join("/tmp/.cache/latent-diffusion", name, "model.ckpt")):
        if os.path.exists(cand):
            return cfg, cand
    raise FileNotFoundError(
        f"No model.ckpt for '{name}'. Set TAED_LDM_ROOT or pass --ckpt.")


def _fix(interior, T):
    """Repair a candidate interior into strictly decreasing points inside (0, T-1).

    Clipping before the strict-decrease pass is not enough: the cascade v[j]=v[j-1]-1 can
    drive values back below the floor when jitter exceeds the spacing. Reserving j slots
    above and n-1-j below each point makes the cascade unable to leave the range.
    """
    n = len(interior)
    if n > T - 2:
        raise ValueError(f"cannot place {n} interior points in [1, {T - 2}]")
    v = np.sort(np.round(interior).astype(np.int64))[::-1]
    idx = np.arange(n)
    v = np.minimum(v, T - 2 - idx)          # room for the j points above
    v = np.maximum(v, n - idx)              # room for the n-1-j points below
    for j in range(1, n):
        v[j] = min(v[j], v[j - 1] - 1)      # preserves v[j] >= n-j, so v[-1] >= 1
    return v


def make_schedules(T, B, M, seed, family="random", jitter=0.5):
    """Each schedule is T-1 = t_0 > ... > t_B = 0, i.e. exactly B denoise calls.

    Schedule 0 is always uniform (the natural baseline).

    family='random' draws uniform interior subsets. That samples the whole schedule space
    but almost every draw is far worse than uniform, so the oracle ends up choosing among
    mostly-bad options and the near-optimal region is barely probed.
    family='jitter' perturbs each interior point by +-jitter of the local spacing, keeping
    every schedule competitive. That is where per-image structure would actually live.
    """
    rng = np.random.default_rng(seed)
    uni = np.linspace(T - 1, 0, B + 1)
    scheds = [uni.round().astype(np.int64)]
    step = (T - 1) / B
    for _ in range(M - 1):
        if family == "jitter":
            cand = uni[1:-1] + rng.uniform(-jitter, jitter, B - 1) * step
        else:
            cand = rng.choice(np.arange(1, T - 1), size=B - 1, replace=False)
        scheds.append(np.concatenate([[T - 1], _fix(cand, T), [0]]).astype(np.int64))
    for s in scheds:
        assert len(s) == B + 1 and s[0] == T - 1 and s[-1] == 0 and np.all(np.diff(s) < 0), s
    return scheds


@torch.no_grad()
def run_schedule(adapter, z_T, sched, device, capture_at=None):
    """len(sched)-1 calls to the LDM. A skip is not a special case: it is simply a larger
    t_start -> t_end jump, which is exactly what the trainer's executor does.

    capture_at returns the latent at the grid point nearest that timestep, which is what a
    policy would actually be conditioning on mid-rollout.
    """
    z = z_T
    cap = None
    t_cur = torch.full((z.shape[0],), int(sched[0]), device=device, dtype=torch.long)
    for j in range(1, len(sched)):
        t_next = torch.full_like(t_cur, int(sched[j]))
        out = adapter._denoise_step_latents(z, t_cur, t_next, eta=0.0)
        z = out[0] if isinstance(out, (tuple, list)) else out
        t_cur = t_next
        if capture_at is not None and cap is None and int(sched[j]) <= capture_at:
            cap = z.clone()
    return (z, cap) if capture_at is not None else z


@torch.no_grad()
def decode(adapter, z, chunk):
    return torch.cat([adapter._decode_latents_to_image(z[i:i + chunk])
                      for i in range(0, z.shape[0], chunk)], dim=0)


@torch.no_grad()
def lpips_to(loss_fn, imgs, refs, chunk):
    out = []
    for i in range(0, imgs.shape[0], chunk):
        out.append(loss_fn(imgs[i:i + chunk], refs[i:i + chunk]).flatten())
    return torch.cat(out).float().cpu().numpy()


class DinoDistance:
    """1 - cosine of DINOv2 embeddings. Drop-in for lpips.LPIPS(a, b) -> [n] distances.

    WHY THIS AND NOT THE MANIFOLD REWARD, which used the same embeddings and failed. That
    one scored an image against a dataset BANK of embeddings, so the image's own identity
    sat in the score and its per-sample SD came out 0.1005 = 7x the entire budget-induced
    signal, 0/32 bands resolvable at batch 8 (REWARD_SPEC.md, REWARD_CHOICE.md). The defect
    was that it was UNPAIRED, not that DINOv2 is insensitive. Comparing an image to ITSELF
    at a different budget cancels the per-image offset, which is exactly why LPIPS-to-
    reference has usable signal at batch 16 while the manifold term needed batch 350-1100.

    So this swaps the METRIC (low-level AlexNet -> semantic ViT) while holding the
    REFERENCE fixed. It therefore tests "is LPIPS too low-level to see per-image
    structure?" and NOT "is the same-seed reference the problem?" -- this is still pinned
    to 0 at B=200 by construction, exactly as REWARD_SPEC.md R8 concedes.
    """

    def __init__(self, device, img_size=224):
        import timm
        self.model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True,
                                       num_classes=0, dynamic_img_size=True)
        self.model = self.model.to(device).eval()
        self.size = img_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def _embed(self, x):
        x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0          # the decoder emits [-1,1]
        x = torch.nn.functional.interpolate(x, size=(self.size, self.size),
                                            mode="bilinear", align_corners=False)
        return torch.nn.functional.normalize(self.model((x - self.mean) / self.std), dim=1)

    @torch.no_grad()
    def __call__(self, a, b):
        return 1.0 - (self._embed(a) * self._embed(b)).sum(dim=1)


def decompose(D):
    """D[i, s], lower is better. Two-way ANOVA without replication."""
    mu = D.mean()
    alpha = D.mean(axis=1) - mu          # image effect
    beta = D.mean(axis=0) - mu           # schedule effect
    gamma = D - mu - alpha[:, None] - beta[None, :]
    return mu, alpha, beta, gamma


def bootstrap_gain(D, n_boot, seed):
    """Resample IMAGES. The schedules are fixed by design, images are the sample."""
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Db = D[idx]
        out[b] = Db.mean(axis=0).min() - Db.min(axis=1).mean()
    return out


def nn_predicted_loss(D, feats):
    """Mean D achieved by predicting each image's best schedule from its nearest neighbour.

    A LEVEL, not a gain. Leave-one-out, so it is honest cross-validation. The feature must
    be something the policy actually sees: z_T is pure Gaussian noise in thousands of
    dimensions where every pair is near-equidistant, so NN on it is a random predictor and
    would understate learnability rather than bound it.
    """
    n = D.shape[0]
    f = feats - feats.mean(axis=0, keepdims=True)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
    sim = f @ f.T
    np.fill_diagonal(sim, -np.inf)
    picked = D.argmin(axis=1)[sim.argmax(axis=1)]      # neighbour's best schedule
    return float(D[np.arange(n), picked].mean())


def learnable_gain(D, feats):
    """Improvement over the best FIXED schedule. Same units as the oracle gain; can be
    negative when the features carry nothing."""
    return float(D.mean(axis=0).min()) - nn_predicted_loss(D, feats)


def null_learnable(D, feats, n_perm, seed):
    """Permutation null: shuffle which image each feature vector belongs to.

    Destroys the image-schedule association while preserving every marginal -- the same D,
    the same features, the same leave-one-out estimator. Returns GAINS, so it is directly
    comparable to learnable_gain(); returning levels here silently made the first version
    compare a gain against a level.
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    return np.array([learnable_gain(D, feats[rng.permutation(n)]) for _ in range(n_perm)])


def evaluate_budget(adapter, loss_fn, z_T, ref_img, B, args, device):
    """D[i,s] at one budget, plus the one-extra-NFE yardstick at that budget."""
    n = z_T.shape[0]
    scheds = make_schedules(args.max_t, B, args.n_schedules, args.seed,
                            family=args.family, jitter=args.jitter)
    D = np.zeros((n, args.n_schedules), dtype=np.float64)
    for s, sc in enumerate(scheds):
        zs = torch.cat([run_schedule(adapter, z_T[i:i + args.batch], sc, device)
                        for i in range(0, n, args.batch)], dim=0)
        D[:, s] = lpips_to(loss_fn, decode(adapter, zs, args.decode_chunk),
                           ref_img, args.decode_chunk)

    up1 = np.linspace(args.max_t - 1, 0, B + 2).round().astype(np.int64)
    z_up1 = torch.cat([run_schedule(adapter, z_T[i:i + args.batch], up1, device)
                       for i in range(0, n, args.batch)], dim=0)
    d_up1 = lpips_to(loss_fn, decode(adapter, z_up1, args.decode_chunk),
                     ref_img, args.decode_chunk)
    return D, d_up1, scheds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LDM_NAME))
    ap.add_argument("--config", default=None, help="override the LDM .yaml")
    ap.add_argument("--ckpt", default=None, help="override the LDM .ckpt")
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--n-schedules", type=int, default=24)
    ap.add_argument("--budget", type=int, default=10, help="NFE per schedule")
    ap.add_argument("--budgets", default=None,
                    help="comma-separated sweep, e.g. 4,8,16,25,50. The reference is "
                         "generated once and shared. Anchor low: AutoDiffusion demos at 4 "
                         "steps and AYS reports gains 'especially in the few-step regime'. "
                         "Add 110 to speak to the project's own operating point.")
    ap.add_argument("--family", default="random", choices=["random", "jitter"],
                    help="'jitter' keeps every schedule near-uniform and therefore "
                         "competitive; 'random' samples the whole space but mostly badly")
    ap.add_argument("--jitter", type=float, default=0.5,
                    help="perturbation as a fraction of the uniform spacing")
    ap.add_argument("--ref-steps", type=int, default=200, help="NFE for the reference")
    ap.add_argument("--metric", default="lpips", choices=["lpips", "dino"],
                    help="'dino' swaps AlexNet LPIPS for DINOv2 cosine against the SAME "
                         "reference, i.e. tests whether the metric is too low-level to see "
                         "per-image structure. It does NOT unpin the curve from 0 at B=200; "
                         "only a reference-free score would, and the one reference-free "
                         "score already built is 7x noisier than its own signal.")
    ap.add_argument("--max-t", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--decode-chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature-t", type=int, default=500,
                    help="timestep whose latent is used as the learnability feature; this "
                         "is what the policy conditions on mid-rollout")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="permutations for the learnability null; this is the actual test")
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import lpips

    from adapter_compvis import create_compvis_adapter

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    print("=" * 74)
    print(f"REWARD STRUCTURE [{args.dataset}] — is adaptive scheduling worth anything?")
    print("=" * 74)

    cfg_path, ckpt_path = resolve_paths(args.dataset, args.config, args.ckpt)
    print(f"  config {cfg_path}")
    print(f"  ckpt   {ckpt_path}")
    adapter = create_compvis_adapter(cfg_path, ckpt_path, device=device,
                                     n_timesteps=args.ref_steps).to(device).eval()
    loss_fn = (lpips.LPIPS(net="alex").to(device).eval() if args.metric == "lpips"
               else DinoDistance(device))
    print(f"  metric {args.metric}")

    C = getattr(adapter, "latent_channels", 4)
    S = getattr(adapter, "latent_size", 32)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    z_T = torch.randn(args.n_images, C, S, S, generator=g).to(device)

    ref_sched = np.linspace(args.max_t - 1, 0, args.ref_steps + 1).round().astype(np.int64)

    budgets = ([int(b) for b in args.budgets.replace(" ", "").split(",") if b]
               if args.budgets else [args.budget])
    calls = args.n_images * (args.n_schedules * sum(budgets) + sum(b + 1 for b in budgets)
                             + args.ref_steps)
    print(f"\n  images={args.n_images}  schedules={args.n_schedules} ({args.family})"
          f"  budgets={budgets} NFE  reference={args.ref_steps} NFE")
    print(f"  LDM calls: {calls:,}")

    ref_pairs = [run_schedule(adapter, z_T[i:i + args.batch], ref_sched, device,
                              capture_at=args.feature_t)
                 for i in range(0, args.n_images, args.batch)]
    ref_z = torch.cat([p[0] for p in ref_pairs], dim=0)
    mid_z = torch.cat([p[1] for p in ref_pairs], dim=0)
    ref_img = decode(adapter, ref_z, args.decode_chunk)
    feats = mid_z.flatten(1).float().cpu().numpy()
    print(f"  reference generated; features captured at t<={args.feature_t}")

    sweep = []
    for B in budgets:
        print(f"\n{'=' * 74}\nBUDGET {B} NFE\n{'=' * 74}")
        D, d_up1, scheds = evaluate_budget(adapter, loss_fn, z_T, ref_img, B, args, device)
        for s in range(args.n_schedules):
            print(f"    schedule {s + 1:3d}/{args.n_schedules}  mean LPIPS {D[:, s].mean():.5f}")

        mu, alpha, beta, gamma = decompose(D)
        best_fixed_s = int(D.mean(axis=0).argmin())
        best_fixed = float(D.mean(axis=0).min())
        oracle = float(D.min(axis=1).mean())
        gain = best_fixed - oracle
        boot = bootstrap_gain(D, args.n_boot, args.seed)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        nn_loss = nn_predicted_loss(D, feats)
        lgain = learnable_gain(D, feats)
        null = null_learnable(D, feats, args.n_perm, args.seed)
        p_val = float((null >= lgain).mean())
        step_value = best_fixed - float(d_up1.mean())
        # the best-fixed column specifically, not the modal winner: the printed sentence
        # is about that schedule, and the two differ whenever the mode is some other column
        win_share = float(np.bincount(D.argmin(axis=1),
                                      minlength=args.n_schedules)[best_fixed_s]) / args.n_images

        col_mean = D.mean(axis=0)
        n_contend = int((col_mean - best_fixed <= max(step_value, 0.0)).sum())
        n_dominated = int((D[:, [best_fixed_s]] <= D).all(axis=0).sum()) - 1
        n_winners = int((np.bincount(D.argmin(axis=1),
                                     minlength=args.n_schedules) > 0).sum())
        # Only element-wise dominance forces the gain to zero. A schedule can be far worse
        # on the mean and still win on individual images -- that IS the interaction.
        pool_ok = n_dominated < args.n_schedules - 2
        thin = n_contend <= 1
        # bedroom B=110 has step_value < 0 -- one extra NFE makes the MEAN worse. The old
        # max(step_value, 1e-12) clamp turned that into gap_nfe = 1.1e9 and so forced
        # isolated=True: a qualitative verdict manufactured by a guard, not measured.
        step_ok = step_value > 0
        gap_nfe = (float((np.delete(col_mean, best_fixed_s).min() - best_fixed) / step_value)
                   if step_ok and args.n_schedules > 1 else float("nan"))
        # A thin pool has two opposite causes and they need opposite fixes, so they must be
        # told apart: perturbations that overshot, or an optimum nothing can get near.
        isolated = thin and step_ok and gap_nfe > 3.0
        print("\n  POOL VALIDITY")
        print(f"    beaten by the best on EVERY image    : {n_dominated} / {args.n_schedules - 1}"
              f"   <- the veto")
        print(f"    distinct per-image winners           : {n_winners} / {args.n_schedules}")
        print(f"    within one NFE of the best on average: {n_contend} / {args.n_schedules}"
              f"   <- coverage, not a veto")
        print("    runner-up is worse by                : "
              + (f"{gap_nfe:.1f} NFE" if step_ok else "n/a -- an extra NFE HURTS the mean"))
        if not pool_ok:
            print("    -> DEGENERATE. The best schedule beats every other on every image, so")
            print("       min_s D[i,s] = D[i,best] and the oracle gain is 0 BY CONSTRUCTION.")
            if args.family == "random":
                print("       Re-run with --family jitter, which keeps schedules near-uniform.")
            else:
                print("       Raise --jitter; the perturbations are too small to matter.")
        elif not step_ok:
            print("    -> THE NFE YARDSTICK IS UNDEFINED at this budget: one extra NFE makes")
            print("       the mean WORSE, so neither 'thin' nor 'isolated' can be stated in")
            print("       NFE, and no gain here converts into a compute saving.")
        elif isolated:
            print("    -> pool is VALID but the optimum is ISOLATED: the runner-up is a whole")
            print("       multiple of an NFE worse, so NO perturbation size produces a")
            print("       contender -- shrink --jitter and schedules collapse onto the")
            print("       optimum, grow it and they fall off the cliff. The thinness is the")
            print("       LANDSCAPE, not the sampling, and the ceiling is not understated.")
        elif thin:
            print("    -> pool is VALID but THIN: no alternative is within one NFE on the")
            print("       mean, so the near-optimal region is barely covered and the measured")
            print("       ceiling UNDERSTATES the true one. LOWER --jitter to pull the pool")
            print("       in, or read the ceiling as a lower bound.")

        # The ceiling is a mean over images. A mean can be one image, and one image is an
        # outlier rather than a structure a policy could ever learn.
        g_i = D[:, best_fixed_s] - D.min(axis=1)
        n_helped = int((g_i > 0).sum())
        g_sum = float(g_i.sum())
        top1_share = float(g_i.max() / max(g_sum, 1e-12))
        eff_images = float(g_sum ** 2 / max(float((g_i ** 2).sum()), 1e-12))
        print("\n  GAIN CONCENTRATION   (the ceiling is a mean -- who actually pays into it?)")
        print(f"    images the oracle helps at all       : {n_helped} / {args.n_images}")
        print(f"    largest single image's share         : {100 * top1_share:.1f}%")
        print(f"    effective images carrying the gain   : {eff_images:.1f} / {args.n_images}")
        if eff_images < 2.0:
            print("    -> the ENTIRE ceiling is one or two OUTLIER images, not a structure.")
            print("       A policy cannot learn a rule from one example, and the bootstrap CI")
            print(f"       must span zero: ~37% of resamples drop the image, and the upper")
            print(f"       bound sits at 3x the point estimate ({3 * gain:.6f}). Check it.")

        print("\n  VARIANCE DECOMPOSITION of D[image, schedule]   (lower LPIPS = better)")
        tot = alpha.var() + beta.var() + gamma.var()   # exactly var(D); verified identity
        print(f"    image effect     var(alpha) = {alpha.var():.6e}  {100 * alpha.var() / tot:5.1f}%  NOT exploitable")
        print(f"    schedule effect  var(beta)  = {beta.var():.6e}  {100 * beta.var() / tot:5.1f}%  a fixed schedule gets this")
        print(f"    INTERACTION      var(gamma) = {gamma.var():.6e}  {100 * gamma.var() / tot:5.1f}%  <- all adaptivity can buy")

        print("\n  ORACLE CEILING   (>= 0 for ANY matrix by Jensen -- a bound, NOT a test)")
        print(f"    best FIXED schedule (#{best_fixed_s})      : {best_fixed:.6f}")
        print(f"    oracle PER-IMAGE schedule          : {oracle:.6f}")
        print(f"    oracle gain                        : {gain:.6f}  95% CI [{lo:.6f}, {hi:.6f}]")
        print(f"    best schedule is ALSO per-image best for {100 * win_share:.1f}% of images")
        print(f"\n  YARDSTICK: one extra NFE ({B}->{B + 1}) buys {step_value:.6f}")
        if step_ok:
            print(f"    oracle ceiling is worth {gain / step_value:.3f} extra NFE")
        else:
            print("    NEGATIVE: the extra step makes the mean WORSE, so there is no NFE")
            print("    yardstick at this budget and every per-NFE ratio below is undefined.")

        print("\n  LEARNABLE GAIN vs PERMUTATION NULL   <- THE ACTUAL TEST")
        print(f"    leave-one-out NN on the t<={args.feature_t} latent : {nn_loss:.6f}")
        print(f"    learnable gain                     : {lgain:.6f}  "
              f"({100 * lgain / max(gain, 1e-12):.1f}% of the ceiling)")
        print(f"    null ({args.n_perm} perms)               : mean {null.mean():.6f}  "
              f"p95 {np.percentile(null, 95):.6f}")
        print(f"    p-value                            : {p_val:.4f}")

        if not pool_ok:
            print("\n    -> INCONCLUSIVE at this budget: the best schedule dominates, so a")
            print("       null here is about the schedules drawn, not about the images.")
        elif p_val >= 0.05:
            if eff_images < 2.0:
                note = "  (ceiling is one image: conclusive, not weak)"
            elif isolated:
                note = "  (optimum isolated: the ceiling is not understated)"
            elif thin:
                note = "  (thin pool: weak evidence)"
            else:
                note = ""
            print("\n    -> NOT PREDICTABLE at this budget." + note)
        elif step_value > 0 and lgain < step_value:
            print("\n    -> REAL BUT NOT WORTH IT at this budget (< one extra NFE).")
        else:
            print("\n    -> EXPLOITABLE at this budget.")

        if args.save_npz:
            path = (args.save_npz.replace(".npz", f"_B{B}.npz")
                    if args.budgets else args.save_npz)
            np.savez(path, D=D, schedules=np.array(scheds), d_up1=d_up1,
                     alpha=alpha, beta=beta, gamma=gamma, null=null, feats=feats,
                     gain_per_image=g_i)
            print(f"    saved {path}")

        print(f"\nSUMMARY reward_structure dataset={args.dataset} budget={B} "
              f"metric={args.metric} "
              f"family={args.family} n={args.n_images} m={args.n_schedules} "
              f"pool_ok={int(pool_ok)} n_dominated={n_dominated} n_winners={n_winners} "
              f"n_contend={n_contend} thin={int(thin)} isolated={int(isolated)} "
              f"gap_nfe={gap_nfe:.2f} n_helped={n_helped} eff_images={eff_images:.2f} "
              f"top1_share={top1_share:.4f} "
              f"oracle_gain={gain:.6f} learnable={lgain:.6f} p={p_val:.4f} "
              f"step_value={step_value:.6f} "
              f"oracle_per_nfe={gain / step_value if step_ok else float('nan'):.4f} "
              f"win_share={win_share:.4f} gamma_share={gamma.var() / tot:.4f}")
        sweep.append((B, gain, step_value, lgain, p_val, win_share,
                      gamma.var() / tot, best_fixed, pool_ok, thin, isolated, eff_images))

    if len(sweep) > 1:
        print("\n" + "=" * 74)
        print("BUDGET SWEEP   (oracle/NFE < 1 means an extra step beats perfect adaptivity)")
        print("=" * 74)
        print(f"  {'B':>5}{'pool':>8}{'bestFixed':>11}{'oracle':>10}{'1 NFE':>10}"
              f"{'oracle/NFE':>12}{'effN':>7}{'gamma%':>8}{'win%':>7}{'p':>7}")
        for B, gn, sv, lg, pv, ws, gs, bf, ok, th, iso, en in sweep:
            tag = "DEGEN" if not ok else ("ISOL" if iso else ("thin" if th else "ok"))
            print(f"  {B:5d}{tag:>8}{bf:11.5f}{gn:10.5f}{sv:10.5f}"
                  f"{gn / sv if sv > 0 else float('nan'):12.3f}"
                  f"{en:7.1f}{100 * gs:7.1f}%{100 * ws:6.0f}%{pv:7.2f}")
        print("\n  DEGEN: the best schedule wins on every image, so the gain is 0 by")
        print("  construction. thin: valid, but no alternative is within one NFE on the")
        print("  mean, so the ceiling is a LOWER bound -- a small value there is weak.")
        print("  ISOL: thin because the runner-up is MANY NFE worse; no perturbation size")
        print("  can produce a contender, so the ceiling is NOT understated.")
        print("  effN: effective number of images carrying the gain. effN ~ 1 means the")
        print("  whole oracle/NFE column entry is a single outlier and means nothing.")
        print("\n  Schedule-optimisation gains are reported in the FEW-STEP regime")
        print("  (AutoDiffusion at 4 steps; AYS 'especially in the few-step regime'), so")
        print("  the low-B rows are the ones that can still overturn a null. The project")
        print("  itself runs DDIM at NFE ~110-220, where DPM-Solver++ needs 15-20.")


if __name__ == "__main__":
    main()
