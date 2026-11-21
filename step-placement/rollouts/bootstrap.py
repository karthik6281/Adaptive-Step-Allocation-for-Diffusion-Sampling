"""Bootstrap intervals for every row, images resampled for gain AND step value together.

CPU only (numpy). Reads the finished outputs under runs1/ and runs2/; changes nothing.

For each row it assembles four per-image vectors, then resamples image indices with replacement:

    base[i]  image i at the shared N schedule, per-image path
    own[i]   image i's per-image value: min(its own searched schedule, base[i])
    dN[i]    image i at the shared N schedule, batched path
    dN1[i]   image i at the shared N+1 schedule, batched path

    gain = mean(base - own)    step = mean(dN - dN1)    R = gain / step    share = R / N

The same resampled indices enter numerator and denominator, so each interval includes the
denominator's sampling error. Intervals are BCa (bias-corrected and accelerated; acceleration from
the image jackknife), with the plain percentile interval printed beside them for comparison. The
schedules are held fixed: the intervals do not include search error (see stored vs corrected) or
the in-sample optimism of refitting the shared schedule.

Versions per row, where available:
  stored          the run as searched (capped images continued by uncap.py where it ran)
  corrected       the better shared N and N+1: the latest round of the shared-search check
                  (round 2 = shared_check.py / sd_more.py --round2, else round 1). For SD the
                  batched path is primary; the per-prompt path is reported too.
  corrected r1    round 1, shown when a round 2 exists (did the correction converge?)

Also: the value of tuning one shared schedule, in steps, R_sh = mean(D_uniform - dN) / step
(needs the --uniform-only outputs); paired ratios for the two single-factor contrasts on the same
images, split into gain ratio x step-value ratio; share ratios across budgets on the same images.
Every row's stored point estimate is checked against the value its run printed.

  python3 utils/bootstrap.py [--n-boot 10000] [--seed 0] [--out runs1/bootstrap.npz]
"""

import argparse
import os
import sys
from statistics import NormalDist

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from uncap import parse_log  # noqa: E402  (numpy only)

N16, S2 = "runs1", "runs2"
ND = NormalDist()


def frozen(ds, B, n, d):
    stem = f"{d}/search_{ds}_B{B}_n{n}_sw10"
    denom = (f"{N16}/denom_search_{ds}_B{B}_n{n}_sw10.npz" if d == S2 else f"{stem}_denom.npz")
    return dict(kind="frozen", ds=ds, B=B, n=n, stem=stem, denom=denom)


def v2(ds, B, n, suffix=""):
    return dict(kind="v2", ds=ds, B=B, n=n, stem=f"{N16}/search_{ds}_B{B}_n{n}_sw10{suffix}")


def sd(B, suffix=""):
    return dict(kind="sd", ds="sd15_coco", B=B, n=64,
                stem=f"{N16}/search_sd15_coco_B{B}_n64_sw10{suffix}", suffix=suffix)


ROWS = [
    ("church DDIM", frozen("church256", 4, 64, S2)), ("church DDIM", frozen("church256", 4, 256, S2)),
    ("church DDIM", frozen("church256", 8, 64, S2)), ("church DDIM", frozen("church256", 8, 256, S2)),
    ("church DDIM", frozen("church256", 12, 64, N16)),
    ("church DDIM", frozen("church256", 12, 256, N16)),
    ("church DDIM", frozen("church256", 16, 64, N16)),
    ("church DDIM", frozen("church256", 20, 64, N16)),
    ("bedroom DDIM", frozen("bedroom256", 4, 64, S2)),
    ("bedroom DDIM", frozen("bedroom256", 4, 256, S2)),
    ("bedroom DDIM", frozen("bedroom256", 8, 64, S2)),
    ("bedroom DDIM", frozen("bedroom256", 8, 256, S2)),
    ("bedroom DDIM", frozen("bedroom256", 12, 64, N16)),
    ("bedroom DDIM", frozen("bedroom256", 16, 64, N16)),
    ("church DPM++2M", v2("church256", 8, 64, "_dpmpp2m")),
    ("ImageNet DDIM", v2("cin256", 4, 256)), ("ImageNet DDIM", v2("cin256", 8, 256)),
    ("SD1.5 cfg7.5", sd(4)), ("SD1.5 cfg7.5", sd(8)), ("SD1.5 cfg3", sd(8, "_cfg3")),
]

PAIRS = [  # (label, row A, row B): same images, so resampled jointly
    ("DPM++2M / DDIM, church N=8, 64 img", ("church DPM++2M", 8, 64), ("church DDIM", 8, 64)),
    ("guidance 7.5 / 3, SD N=8", ("SD1.5 cfg7.5", 8, 64), ("SD1.5 cfg3", 8, 64)),
]
TRENDS = [("church DDIM", 64), ("church DDIM", 256), ("bedroom DDIM", 64), ("bedroom DDIM", 256)]


def tag(stem):
    return f"{os.path.basename(os.path.dirname(stem))}_{os.path.basename(stem)}"


def load(path):
    return np.load(path) if os.path.exists(path) else None


def f64(a):
    return np.asarray(a, np.float64)


def uncapped(r, per_img):
    u = load(f"{N16}/uncap_{tag(r['stem'])}.npz")
    if u is None:
        return per_img, 0
    new = f64(u["per_img_new"])
    if new.shape != per_img.shape or np.any(new > per_img + 1e-9):
        raise SystemExit(f"{r['stem']}: uncap output inconsistent with the run")
    return new, int(len(u["capped_idx"]))


def corrections(r):
    """[(label, file)] of the shared-search corrections that exist, oldest first."""
    if r["kind"] == "sd":
        b = f"sd15_coco_B{r['B']}_n64{r['suffix']}"
        cands = [("r1", f"{N16}/sdrescore_{b}.npz"), ("r2", f"{N16}/sdround2_{b}.npz")]
        return [(k, load(p)) for k, p in cands if os.path.exists(p)]
    t = tag(r["stem"])
    out = []
    c = load(f"{N16}/rescore_{t}.npz")
    if c is None:
        s = load(f"{N16}/shared_{t}.npz")
        c = s if s is not None and "D_best_B1_batched" in s.files else None
    if c is not None:
        out.append(("r1", c))
    c2 = load(f"{N16}/round2_{t}.npz")
    if c2 is not None:
        out.append(("r2", c2))
    return out


def uniform_vec(r):
    if r["kind"] == "sd":
        u = load(f"{N16}/sduniform_sd15_coco_B{r['B']}_n64{r['suffix']}.npz")
        if u is not None:
            return f64(u["D_uniform_batched"])
        p = load(f"{N16}/sdpost_sd15_coco_B{r['B']}_n64.npz") if not r["suffix"] else None
        return f64(p["uniform_B"])[0] if p is not None else None
    u = load(f"{N16}/uniform_{tag(r['stem'])}.npz")
    return f64(u["D_uniform_batched"]) if u is not None else None


def assemble(r):
    """{version: (base, own, dN, dN1)}, notes, uniform vector; None if the run is missing."""
    src = load(r["stem"] + ".npz")
    if src is None:
        return None, ["run not finished"], None
    notes, out = [], {}
    per_img, n_unc = uncapped(r, f64(src["per_img"]))
    if n_unc:
        notes.append(f"{n_unc} capped images continued")
    gain_s, step_s = float(src["gain"]), float(src["step_value"])
    if r["kind"] == "frozen":
        rows, _ = parse_log(r["stem"] + ".txt")
        base = np.array([rows[i][0] for i in range(len(per_img))])   # logged, 6 decimals
        dn = load(r["denom"])
        if dn is None or int(dn["ok"]) != 1:
            return None, ["no passing denominator file"], None
        dN, dN1 = f64(src["D_fixed"]), f64(dn["D_B1"])
        tol_g = 1e-6                                                  # log rounding
    else:
        base, dN, dN1 = f64(src["D_fixed_unbatched"]), f64(src["D_fixed"]), f64(src["D_fixed_B1"])
        tol_g = 1e-9
    g0 = float((base - f64(src["per_img"])).mean())
    if abs(g0 - gain_s) > tol_g or abs(float((dN - dN1).mean()) - step_s) > 1e-6:
        raise SystemExit(f"{r['stem']}: per-image vectors do not give the stored gain/step "
                         f"({g0:.6f} vs {gain_s:.6f})")
    out["stored"] = (base, np.minimum(per_img, base), dN, dN1)

    cors = corrections(r)
    if not cors:
        notes.append("not corrected yet")
    for k, (lab, c) in enumerate(cors):
        latest = k == len(cors) - 1
        name = "corrected" if latest else f"corrected {lab}"
        bN, bN1, bu = f64(c["D_best_B_batched"]), f64(c["D_best_B1_batched"]), \
            f64(c["D_best_B_unbatched"])
        if r["kind"] == "sd":
            own_b = f64(c["own_batched"])
            if k == 0:
                out["stored, batched"] = (dN, np.minimum(own_b, dN), dN, dN1)
            out[f"{name}, batched"] = (bN, np.minimum(own_b, bN), bN, bN1)
            if latest:
                out[f"{name}, per-prompt"] = (bu, np.minimum(per_img, bu), bN, bN1)
        else:
            out[name] = (bu, np.minimum(per_img, bu), bN, bN1)
        if latest and lab == "r2":
            notes.append("round-2 correction")
    return out, notes, uniform_vec(r)


def num_den(v):
    base, own, dN, dN1 = v
    return base - own, dN - dN1


def ratio_boot(a, b, idx):
    """Bootstrap and jackknife of mean(a)/mean(b); returns (point, boot, jack)."""
    sa, sb = a.sum(), b.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        point = sa / sb if sb > 0 else np.nan
        ma, mb = a[idx].mean(-1), b[idx].mean(-1)
        boot = np.where(mb > 0, ma / mb, np.nan)
        jack = np.where(sb - b > 0, (sa - a) / (sb - b), np.nan)
    return float(point), boot, jack


def pct(boot):
    x = boot[np.isfinite(boot)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan,) * 2


def bca(point, boot, jack, alpha=0.05):
    x = boot[np.isfinite(boot)]
    j = jack[np.isfinite(jack)]
    if len(x) < 100 or not np.isfinite(point) or len(j) < 3:
        return (np.nan, np.nan)
    p = (np.sum(x < point) + 0.5 * np.sum(x == point)) / len(x)
    if not 0 < p < 1:
        return pct(boot)
    z0 = ND.inv_cdf(p)
    d = j.mean() - j
    den = 6.0 * float((d ** 2).sum()) ** 1.5
    acc = float((d ** 3).sum()) / den if den > 0 else 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = ND.inv_cdf(q)
        w = z0 + z
        q_adj = ND.cdf(z0 + w / (1 - acc * w))
        out.append(float(np.percentile(x, 100 * q_adj)))
    return tuple(out)


# ---- SD menu against the corrected shared schedule (greedy_menu / menu_analysis: sd_checks.py)
def greedy_menu(M, rows, cands, K):
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
    n = L.shape[0]
    rows = np.arange(n)
    base_b, own_b = float(L[:, 0].mean()), float(L[rows, rows + 1].mean())
    den_in = base_b - own_b
    K = min(menu_max, n + 1)
    chosen_in, means_in = greedy_menu(L, rows, list(range(n + 1)), K)
    frac_in = [(base_b - m) / den_in if den_in > 0 else float("nan") for m in means_in]
    folds = [np.arange(0, n, 2), np.arange(1, n, 2)]
    K_cv = min(menu_max, 1 + min(len(f) for f in folds))
    num_cv, den_cv = np.zeros(K_cv), 0.0
    for a, b in ((0, 1), (1, 0)):
        tr, tst = folds[a], folds[b]
        ch, _ = greedy_menu(L, tr, [0] + [int(j) + 1 for j in tr], K_cv)
        for k in range(K_cv):
            num_cv[k] += float((L[tst, 0] - L[np.ix_(tst, ch[:k + 1])].min(1)).sum())
        den_cv += float((L[tst, 0] - L[tst, tst + 1]).sum())
    frac_cv = [x / den_cv if den_cv > 0 else float("nan") for x in num_cv]
    return base_b, own_b, frac_in, frac_cv


def sd_menu():
    post = load(f"{N16}/sdpost_sd15_coco_B8_n64.npz")
    cors = corrections(sd(8))
    if post is None or not cors:
        print("\n  SD menu vs corrected shared schedule: files not there yet")
        return
    M = f64(post["menu_M"])[0].copy()
    res = [("stored shared N", menu_analysis(M, 8))]
    for lab, c in cors:
        M[:, 0] = f64(c["D_best_B_batched"])
        res.append((f"corrected {lab} shared N", menu_analysis(M, 8)))
    print("\n  SD N=8 schedule menu (LPIPS, batched; share of the per-prompt gain captured, "
          "in-sample / 2-fold held-out)")
    for label, (bb, ob, fi, fc) in res:
        print(f"    {label:24s} {bb:.6f}  own {ob:.6f}  gain {bb - ob:.6f}   " + "  ".join(
            f"K={k + 1} {100 * fi[k]:5.1f}%/{100 * fc[k]:5.1f}%" for k in (1, 3, 7)))


def fmt(c):
    return f"[{c[0]:6.3f},{c[1]:6.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{N16}/bootstrap.npz")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    idx_cache, res, save, unif = {}, {}, {}, {}

    def idx_for(n):   # one index matrix per image count: rows on the same images pair up
        if n not in idx_cache:
            idx_cache[n] = rng.integers(0, n, size=(args.n_boot, n))
        return idx_cache[n]

    print(f"BOOTSTRAP  {args.n_boot} resamples of the images (seed {args.seed}); BCa 95% "
          f"intervals (percentile interval in the last column)\n")
    print(f"  {'row':16s} {'N':>3s} {'img':>4s} {'version':24s} {'gain':>8s} {'step':>8s} "
          f"{'R':>6s} {'R BCa 95%':>15s} {'share':>6s} {'share BCa 95%':>15s} {'P(R>1)':>6s}  "
          f"{'R percentile':>15s}")
    for label, r in ROWS:
        vers, notes, u = assemble(r)
        if vers is None:
            print(f"  {label:16s} {r['B']:3d} {r['n']:4d} -- missing: {'; '.join(notes)}")
            continue
        idx = idx_for(r["n"])
        for vname, v in vers.items():
            a, b = num_den(v)
            R0, Rb, Rj = ratio_boot(a, b, idx)
            res[(label, r["B"], r["n"], vname)] = (a, b, R0, Rb, Rj)
            save[f"{label}|B{r['B']}|n{r['n']}|{vname}"] = Rb
            cb, cp = bca(R0, Rb, Rj), pct(Rb)
            bad = int((~np.isfinite(Rb)).sum())
            print(f"  {label:16s} {r['B']:3d} {r['n']:4d} {vname:24s} {a.mean():8.5f} "
                  f"{b.mean():8.5f} {R0:6.3f} {fmt(cb)} {100 * R0 / r['B']:5.1f}% "
                  f"[{100 * cb[0] / r['B']:5.1f},{100 * cb[1] / r['B']:5.1f}]% "
                  f"{float(np.mean(Rb[np.isfinite(Rb)] > 1)):6.3f}  {fmt(cp)}"
                  + (f"  ({bad} resamples with step <= 0)" if bad else ""))
        if u is not None:
            unif[(label, r["B"], r["n"])] = (u, vers)
        if notes:
            print(f"  {'':16s} {'':3s} {'':4s} note: {'; '.join(notes)}")

    print("\n  value of tuning one shared schedule, in steps: R_sh = mean(D_uniform - D_shared N) "
          "/ step  (BCa 95%)")
    for (label, B, n), (u, vers) in unif.items():
        for vname in [k for k in vers if k in ("stored", "corrected", "corrected, batched")]:
            _, _, dN, dN1 = vers[vname]
            p0, pb, pj = ratio_boot(u - dN, dN - dN1, idx_for(n))
            R0 = res[(label, B, n, vname)][2]
            print(f"    {label:16s} N={B:2d} {n:4d} img {vname:20s} R_sh {p0:6.3f} {fmt(bca(p0, pb, pj))}"
                  f"   per-sample R {R0:6.3f}   uniform -> per-sample {p0 + R0:6.3f} steps")
    if not unif:
        print("    (no --uniform-only outputs yet)")

    print("\n  paired contrasts (same images, joint resampling): R ratio = gain ratio x "
          "step-value ratio (B over A)  (BCa 95%)")
    for plabel, A, Bq in PAIRS:
        for va in ("corrected", "corrected, batched", "stored"):
            ka, kb = A + (va,), Bq + (va,)
            if ka not in res or kb not in res:
                continue
            aA, bA, RA, RbA, RjA = res[ka]
            aB, bB, RB, RbB, RjB = res[kb]
            with np.errstate(divide="ignore", invalid="ignore"):
                q0, qb, qj = RA / RB, RbA / RbB, RjA / RjB
            g0, gb, gj = ratio_boot(aA, aB, idx_for(len(aA)))
            s0, sb, sj = ratio_boot(bB, bA, idx_for(len(aA)))
            print(f"    {plabel:38s} {va:20s} ratio {q0:5.2f} {fmt(bca(q0, qb, qj))}  "
                  f"(percentile {fmt(pct(qb))})   gain x{g0:4.2f} {fmt(bca(g0, gb, gj))}"
                  f"   step x{s0:4.2f} {fmt(bca(s0, sb, sj))}")

    print("\n  share across budgets on the same images: share(N_b) / share(N_a)  (BCa 95%)")
    for label, n in TRENDS:
        Bs = sorted(B for (lab, B, nn, v) in res if lab == label and nn == n and v == "corrected")
        for i, Ba in enumerate(Bs):
            for Bb in Bs[i + 1:]:
                _, _, Ra, Rba, Rja = res[(label, Ba, n, "corrected")]
                _, _, Rb_, Rbb, Rjb = res[(label, Bb, n, "corrected")]
                with np.errstate(divide="ignore", invalid="ignore"):
                    q0, qb, qj = (Rb_ / Bb) / (Ra / Ba), (Rbb / Bb) / (Rba / Ba), \
                        (Rjb / Bb) / (Rja / Ba)
                fin = qb[np.isfinite(qb)]
                print(f"    {label:13s} {n:4d} img  N={Bb:2d} / N={Ba:2d}: {q0:5.2f} "
                      f"{fmt(bca(q0, qb, qj))}   P(>1) {float(np.mean(fin > 1)):.3f}")
    sd_menu()
    np.savez(args.out, keys=np.asarray(list(save)), **{f"R_{j}": v for j, v in
                                                      enumerate(save.values())},
             n_boot=np.int64(args.n_boot), seed=np.int64(args.seed))
    print(f"\n  saved {args.out}")


if __name__ == "__main__":
    main()
