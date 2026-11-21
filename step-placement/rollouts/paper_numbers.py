#!/usr/bin/env python3
"""Every number in the paper that comes from the step-placement runs, written out for LaTeX.

Run from the directory that holds runs1/ and runs2/ (the taed directory on the
cluster), or pass it with --root:

    python3 utils/paper_numbers.py --out report/tmlr_v3

Writes <out>/numbers.tex (one \\pndef{key}{value} per number) and <out>/tables/<name>.tex (the
data rows of each table), prints every key with its value, and checks the qualitative claims
the text makes (non-overlapping intervals, ratios excluding one, ...); it exits with status 1
if any claim fails. It uses bootstrap.py's own functions and the same resamples (seed 0,
10,000 resamples, index matrices drawn in the same order), so every interval is the one
bootstrap.py prints. The local-error analysis (Section 5, Appendix H) is not covered here.
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import bootstrap as bs  # noqa: E402

N16 = bs.N16
NUM, TABLES, CLAIMS = {}, {}, []
RAW = {}   # full-precision values for utils/paper_figures.py: key -> [value, lo, hi] or a list


# ---------------------------------------------------------------------------- formatting
def put(key, value):
    if key in NUM:
        raise SystemExit(f"duplicate key {key}")
    NUM[key] = value


def f(x, d):
    if not np.isfinite(x):
        raise SystemExit(f"non-finite value {x}")
    s = f"{x:.{d}f}"
    return "0" if float(s) == 0 and d > 0 and abs(x) < 0.5 * 10 ** -d else s


def pc(x, d=1):            # fraction -> percent string, no % sign
    return f(100 * x, d)


def iv_tab(c, d, scale=1.0):
    return rf"[{f(scale * c[0], d)},\,{f(scale * c[1], d)}]"


def put_val(key, x, ci, d, scale=1.0):
    """key, key.lo, key.hi, key.ci (table form [a, b])."""
    put(key, f(scale * x, d))
    put(key + ".lo", f(scale * ci[0], d))
    put(key + ".hi", f(scale * ci[1], d))
    put(key + ".ci", iv_tab(ci, d, scale))
    RAW[key] = [float(scale * x), float(scale * ci[0]), float(scale * ci[1])]


def claim(text, ok):
    CLAIMS.append((text, bool(ok)))


def excl1(c):
    return c[0] > 1 or c[1] < 1


# ---------------------------------------------------------------------------- the rows
ID = {"church DDIM": "church", "bedroom DDIM": "bedroom", "church DPM++2M": "dpm",
      "ImageNet DDIM": "cin", "SD1.5 cfg7.5": "sd", "SD1.5 cfg3": "sdgthree"}


def rid(label, B, n):
    return f"{ID[label]}{B}" + ("" if label.startswith("SD") else f"n{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.chdir(args.root)

    rng = np.random.default_rng(0)
    idx = {}

    def idx_for(n):   # bootstrap.py draws one matrix per image count, in order of first use
        if n not in idx:
            idx[n] = rng.integers(0, n, size=(10000, n))
        return idx[n]

    rows = {}
    for label, r in bs.ROWS:
        vers, notes, u = bs.assemble(r)
        if vers is None or u is None:
            raise SystemExit(f"{r['stem']}: {notes}")
        I = idx_for(r["n"])
        sd = r["kind"] == "sd"
        res = {}
        for vn, v in vers.items():
            a, b = bs.num_den(v)
            R0, Rb, Rj = bs.ratio_boot(a, b, I)
            res[vn] = dict(a=a, b=b, R=R0, Rb=Rb, Rj=Rj, ci=bs.bca(R0, Rb, Rj), v=v)
        k = rid(label, r["B"], r["n"])
        rows[k] = dict(label=label, r=r, B=r["B"], n=r["n"], u=u, res=res, I=I,
                       st="stored, batched" if sd else "stored",
                       co="corrected, batched" if sd else "corrected")

    # ---- per row: corrected (primary) values, stored values, refinement, Omega
    for k, w in rows.items():
        B, co, st = w["B"], w["res"][w["co"]], w["res"][w["st"]]
        put(f"{k}.D", f(co["a"].mean(), 4))
        put(f"{k}.V", f(co["b"].mean(), 4))
        put_val(f"{k}.R", co["R"], co["ci"], 2)
        put_val(f"{k}.s", co["R"], co["ci"], 1, scale=100.0 / B)
        put(f"{k}.Rorig", f(st["R"], 2))
        dN = st["v"][2].mean() - co["v"][2].mean()
        dN1 = st["v"][3].mean() - co["v"][3].mean()
        if dN < -1e-7 or dN1 < -1e-7:
            raise SystemExit(f"{k}: corrected shared schedule worse than stored")
        put(f"{k}.dN", f(dN, 4))
        put(f"{k}.dNone", f(dN1, 4))
        w["dN"], w["dN1"] = dN, dN1
        put(f"{k}.Dsh", f(co["v"][2].mean(), 3))
        # share of the shared schedule's remaining LPIPS that per-image placement removes
        put(f"{k}.Dfrac", f(100 * co["a"].mean() / co["v"][0].mean(), 0))
        put(f"{k}.improved", str(int((co["a"] > 1e-9).sum())))
        # Omega = 1 + (Dbar_unif - Dbar_sh) / Delta, exactly 1 + R_sh / R
        base, own, dsh, _ = co["v"]
        g = base - own
        p0, pb, pj = bs.ratio_boot((w["u"] - dsh) + g, g, w["I"])
        c = bs.bca(p0, pb, pj)
        w["om"] = (p0, c)
        put_val(f"{k}.om", p0, c, 2)
        put_val(f"{k}.cap", 1 - 1 / p0, (1 - 1 / c[0], 1 - 1 / c[1]), 0, scale=100.0)
        if "corrected, per-prompt" in w["res"]:
            put(f"{k}.Rpp", f(w["res"]["corrected, per-prompt"]["R"], 2))

    # ---- Table 1 and the setting comparison
    T1_LSUN = ["church4n256", "church8n256", "bedroom4n256", "bedroom8n256"]
    T1_OTHER = ["cin4n256", "cin8n256", "sd4", "sd8", "dpm8n64"]
    for B in (4, 8):
        lsun = [rows[k] for k in T1_LSUN if rows[k]["B"] == B]
        oth = [rows[k] for k in T1_OTHER if rows[k]["B"] == B]
        hi = max(x["res"][x["co"]]["ci"][1] / B for x in lsun)
        lo = min(x["res"][x["co"]]["ci"][0] / B for x in oth)
        claim(f"N={B}: LSUN share intervals below the other settings' ({100*hi:.1f}% < "
              f"{100*lo:.1f}%)", hi < lo)
    sh = lambda ks: [rows[k]["res"][rows[k]["co"]]["R"] / rows[k]["B"] for k in ks]  # noqa: E731
    put("lsun.s.min", f(100 * min(sh(T1_LSUN)), 0))
    put("lsun.s.max", f(100 * max(sh(T1_LSUN)), 0))
    put("other.s.min", f(100 * min(sh(T1_OTHER)), 0))
    put("other.s.max", f(100 * max(sh(T1_OTHER)), 0))
    g3 = rows["sdgthree8"]
    claim("SD guidance 3 share interval overlaps the LSUN N=8 shares",
          g3["res"][g3["co"]]["ci"][0] / 8 < max(sh(["church8n256", "bedroom8n256"])))
    D = lambda ks: [rows[k]["res"][rows[k]["co"]]["a"].mean() for k in ks]  # noqa: E731
    V = lambda ks: [rows[k]["res"][rows[k]["co"]]["b"].mean() for k in ks]  # noqa: E731
    put("lsun.D.min", f(min(D(T1_LSUN)), 3))
    put("lsun.D.max", f(max(D(T1_LSUN)), 3))
    put("cinsd.D.min", f(min(D(["cin4n256", "cin8n256", "sd4", "sd8"])), 3))
    put("cinsd.D.max", f(max(D(["cin4n256", "cin8n256", "sd4", "sd8"])), 3))
    four = ["church4n256", "bedroom4n256", "cin4n256", "sd4"]
    put("Vfour.min", f(min(V(four)), 3))
    put("Vfour.max", f(max(V(four)), 3))
    put("cin8.V3", f(V(["cin8n256"])[0], 3))
    put("lsun.Veight.min", f(min(V(["church8n256", "bedroom8n256"])), 3))
    put("lsun.Veight.max", f(max(V(["church8n256", "bedroom8n256"])), 3))
    imp = [int(rows[k]["res"][rows[k]["co"]]["a"].__gt__(1e-9).sum()) for k in T1_LSUN]
    put("lsun.improved.min", str(min(imp)))
    put("lsun.improved.max", str(max(imp)))
    impp = [int(rows[k]["res"][rows[k]["co"]]["a"].__gt__(1e-9).sum())
            for k in ("sd4", "sd8", "sdgthree8")]
    put("sd.improved.min", str(min(impp)))
    put("sd.improved.max", str(max(impp)))
    distinct = {}
    for k in ("church8n256", "bedroom8n256", "cin8n256", "sd8", "sdgthree8"):
        src = np.load(rows[k]["r"]["stem"] + ".npz")
        distinct[k] = len({tuple(s) for s in src["per_sched"]})
    put("distinct.img.min", str(min(distinct[k] for k in ("church8n256", "bedroom8n256", "cin8n256"))))
    put("distinct.img.max", str(max(distinct[k] for k in ("church8n256", "bedroom8n256", "cin8n256"))))
    put("distinct.sd.min", str(min(distinct["sd8"], distinct["sdgthree8"])))
    put("distinct.sd.max", str(max(distinct["sd8"], distinct["sdgthree8"])))
    put("distinct.sdeight", str(distinct["sd8"]))
    # for the appendix figures: per-image schedules (searched from the main-search shared
    # schedule) and each image's gain over the refined shared schedule in units of V_N
    for k in ("church8n256", "bedroom8n256", "cin8n256", "sd8", "dpm8n64", "sdgthree8"):
        w = rows[k]
        src = np.load(w["r"]["stem"] + ".npz")
        co = w["res"][w["co"]]
        RAW[f"sched.{k}"] = {"shared": src["fixed_B"].astype(int).tolist(),
                             "per": src["per_sched"].astype(int).tolist()}
        RAW[f"pergain.{k}"] = (co["a"] / co["b"].mean()).astype(float).tolist()
        claim(f"{k}: mean per-image gain in units of V_N equals R_N",
              abs(np.mean(RAW[f"pergain.{k}"]) - co["R"]) < 1e-9)
        claim(f"{k}: median per-image gain lies below R_N",
              np.median(RAW[f"pergain.{k}"]) < co["R"])

    # ---- paired comparisons (as in bootstrap.py)
    I64 = idx_for(64)
    for name, ka, kb in (("dpm", "dpm8n64", "church8n64"), ("guid", "sd8", "sdgthree8")):
        A, Bq = rows[ka]["res"][rows[ka]["co"]], rows[kb]["res"][rows[kb]["co"]]
        q0, qb, qj = A["R"] / Bq["R"], A["Rb"] / Bq["Rb"], A["Rj"] / Bq["Rj"]
        cq = bs.bca(q0, qb, qj)
        g0, gb, gj = bs.ratio_boot(A["a"], Bq["a"], I64)
        s0, sb, sj = bs.ratio_boot(Bq["b"], A["b"], I64)
        cg, cs = bs.bca(g0, gb, gj), bs.bca(s0, sb, sj)
        put_val(f"pair.{name}.q", q0, cq, 2)
        put(f"pair.{name}.q.loone", f(cq[0], 1))
        put(f"pair.{name}.q.hione", f(cq[1], 1))
        put_val(f"pair.{name}.g", g0, cg, 2)
        put_val(f"pair.{name}.st", s0, cs, 2)
        claim(f"pair {name}: R ratio interval excludes 1", excl1(cq))
        claim(f"pair {name}: gain-ratio interval excludes 1", excl1(cg))
        claim(f"pair {name}: step-ratio interval {'excludes' if name == 'dpm' else 'includes'} 1",
              excl1(cs) if name == "dpm" else not excl1(cs))

    # ---- budget trends (share ratios on the same images)
    trend = {}
    for ds, n in (("church", 64), ("church", 256), ("bedroom", 64), ("bedroom", 256)):
        Bs = sorted(w["B"] for k, w in rows.items() if k.startswith(ds) and w["n"] == n
                    and w["label"].endswith("DDIM"))
        for i, Ba in enumerate(Bs):
            for Bb in Bs[i + 1:]:
                ra = rows[f"{ds}{Ba}n{n}"]["res"]["corrected"]
                rb = rows[f"{ds}{Bb}n{n}"]["res"]["corrected"]
                q0 = (rb["R"] / Bb) / (ra["R"] / Ba)
                qb = (rb["Rb"] / Bb) / (ra["Rb"] / Ba)
                qj = (rb["Rj"] / Bb) / (ra["Rj"] / Ba)
                c = bs.bca(q0, qb, qj)
                trend[(ds, n, Ba, Bb)] = c
                put_val(f"trend.{ds}{n}.{Bb}.{Ba}", q0, c, 2)
    claim("church, 64 images: every share-ratio interval includes 1",
          all(not excl1(c) for (ds, n, a, b), c in trend.items() if ds == "church" and n == 64))
    claim("church 256: share ratio N=8/N=4 excludes 1", excl1(trend[("church", 256, 4, 8)]))
    claim("bedroom 256: share ratio N=8/N=4 excludes 1", excl1(trend[("bedroom", 256, 4, 8)]))
    claim("church 256: share ratio N=12/N=8 includes 1", not excl1(trend[("church", 256, 8, 12)]))
    claim("bedroom 64: share ratio N=12/N=8 includes 1", not excl1(trend[("bedroom", 64, 8, 12)]))
    T2 = ["church4n64", "church8n64", "church12n64", "church12n256",
          "bedroom4n64", "bedroom8n64", "bedroom12n64"]
    put("budget.D.min", f(min(D(T1_LSUN + T2)), 3))
    put("budget.D.max", f(max(D(T1_LSUN + T2)), 3))
    put("church4n256.V3", f(V(["church4n256"])[0], 3))
    put("church12n256.V3", f(V(["church12n256"])[0], 3))
    put("church256.Rgrowth",
        f(rows["church12n256"]["res"]["corrected"]["R"] / rows["church4n256"]["res"]["corrected"]["R"], 1))
    put("church256.sgrowth", f((rows["church12n256"]["res"]["corrected"]["R"] / 12)
                               / (rows["church4n256"]["res"]["corrected"]["R"] / 4), 1))
    for k, want in (("church8n256", False), ("bedroom8n256", False)):
        claim(f"{k}: R interval below 1", rows[k]["res"]["corrected"]["ci"][1] < 1)
    for k, ex in (("church12n256", True), ("bedroom12n64", False), ("church16n64", False),
                  ("bedroom16n64", True), ("church20n64", True)):
        claim(f"{k}: R interval {'excludes' if ex else 'includes'} 1",
              excl1(rows[k]["res"]["corrected"]["ci"]) == ex)

    # ---- value of a tuned shared schedule (Omega)
    OM = ["church4n256", "church8n256", "church12n256", "bedroom4n256", "bedroom8n256",
          "cin4n256", "cin8n256", "sd4", "sd8", "dpm8n64", "sdgthree8"]
    om = [rows[k]["om"][0] for k in OM]
    put("om.min", f(min(om), 1))
    put("om.max", f(max(om), 1))
    put("cap.min", f(100 * (1 - 1 / min(om)), 0))
    put("cap.max", f(100 * (1 - 1 / max(om)), 0))
    claim("every Omega interval lies above 1", all(rows[k]["om"][1][0] > 1 for k in OM))
    claim("smallest Omega is church N=4, largest ImageNet N=8",
          min(OM, key=lambda k: rows[k]["om"][0]) == "church4n256"
          and max(OM, key=lambda k: rows[k]["om"][0]) == "cin8n256")

    # ---- refinement
    lsun_small = [k for k in rows if rows[k]["label"].endswith("DDIM") and k[:2] in ("ch", "be")
                  and rows[k]["B"] <= 8]
    chg = {k: 1 - rows[k]["res"][rows[k]["co"]]["R"] / rows[k]["res"][rows[k]["st"]]["R"]
           for k in rows}
    kmax = max(lsun_small, key=lambda k: chg[k])
    put("refine.lsunsmall.max", f(100 * chg[kmax], 0))
    claim(f"largest LSUN N<=8 refinement change is church N=4, 256 images ({kmax})",
          kmax == "church4n256")
    claim("refinement never raised R on LSUN N<=8", all(chg[k] >= -1e-12 for k in lsun_small))
    put("refine.sdeight", f(100 * chg["sd8"], 0))
    put("refine.sdeight.dNone", f(rows["sd8"]["dN1"], 3))
    large = [k for k in rows if rows[k]["label"].endswith("DDIM") and k[:2] in ("ch", "be")
             and rows[k]["B"] >= 12]
    klarge = max(large, key=lambda k: chg[k])
    put("refine.large.max", f(100 * chg[klarge], 0))
    LG = ["church16n64", "church20n64", "bedroom16n64"]
    put("larger.V.min", f(min(V(LG)), 3))
    put("larger.V.max", f(max(V(LG)), 3))
    put("larger.s.min", f(100 * min(sh(LG)), 1))
    put("larger.s.max", f(100 * max(sh(LG)), 1))
    claim(f"largest refinement change at N>=12 is bedroom N=16 ({klarge})", klarge == "bedroom16n64")
    # second round: only church N=20 moved
    r2 = {k: rows[k]["res"].get("corrected r1") for k in rows}
    moved = []
    for k, w in rows.items():
        if "corrected r1" in w["res"] or "corrected r1, batched" in w["res"]:
            r1 = w["res"]["corrected r1" if "corrected r1" in w["res"] else "corrected r1, batched"]
            d2N = r1["v"][2].mean() - w["res"][w["co"]]["v"][2].mean()
            d2N1 = r1["v"][3].mean() - w["res"][w["co"]]["v"][3].mean()
            if d2N > 1e-9 or d2N1 > 1e-9:
                moved.append(k)
                put(f"{k}.rtwo.dN", f(d2N, 4))
                put(f"{k}.rtwo.dNone", f(d2N1, 4))
                put(f"{k}.rtwo.Rone", f(r1["R"], 2))
    claim(f"second round moved only church N=20 ({moved})", moved == ["church20n64"])
    del r2

    # ---- capped searches
    total, capped, worst, wk = 0, 0, 0.0, None
    for k, w in rows.items():
        total += w["n"]
        src = bs.load(w["r"]["stem"] + ".npz")
        old = bs.f64(src["per_img"])
        new, nc = bs.uncapped(w["r"], old)
        capped += nc
        if nc:
            base, own, _, _ = w["res"][w["st"]]["v"]
            rel = (base - own).mean() / (base - np.minimum(old, base)).mean() - 1
            if abs(rel) > worst:
                worst, wk = abs(rel), k
    put("searches.total", f"{total:,}".replace(",", "{,}"))
    put("searches.capped", str(capped))
    put("uncap.max", f(100 * worst, 2))
    claim(f"largest change from continuing capped searches is church N=20 ({wk})", wk == "church20n64")

    # ---- SD details (Appendix A.2)
    sd8 = rows["sd8"]["res"]
    lower = 1 - sd8["stored, batched"]["a"].mean() / sd8["stored"]["a"].mean()
    put("sd.batchlower", f(100 * lower, 0))

    # ---- convexity and brackets (Appendix B)
    gaps = {}
    for ds in ("church", "bedroom"):
        pts = {}
        for k, w in rows.items():
            if k.startswith(ds) and w["n"] == 64 and w["label"].endswith("DDIM"):
                v = w["res"]["corrected"]["v"]
                pts[w["B"]], pts[w["B"] + 1] = float(v[2].mean()), float(v[3].mean())
        Bs = sorted(pts)
        RAW[f"pts.{ds}"] = [[b, pts[b]] for b in Bs]
        slopes = [(pts[a] - pts[b]) / (b - a) for a, b in zip(Bs, Bs[1:])]
        claim(f"{ds}: shared-schedule slopes fall at every measured budget",
              all(x > y for x, y in zip(slopes, slopes[1:])))
        put(f"{ds}.slope.first", f(slopes[0], 3))
        put(f"{ds}.slope.last", f(slopes[-1], 3))
        put(f"{ds}.slope.firstN", f"{Bs[0]}")
        put(f"{ds}.slope.lastN", f"{Bs[-2]}")
        for B in sorted({w["B"] for k, w in rows.items() if k.startswith(ds) and w["n"] == 64}):
            co = rows[f"{ds}{B}n64"]["res"]["corrected"]
            R, Dl, Vl = co["R"], co["a"].mean(), co["b"].mean()
            if R < 1:
                continue
            later = [b for b in Bs if b > B + 1]
            if not later:
                continue
            b2 = later[0]
            slope = (pts[B + 1] - pts[b2]) / (b2 - B - 1)
            up = 1 + (Dl - Vl) / slope
            claim(f"{ds} N={B}: chord bound lies inside the measured interval", up <= b2 - B)
            put(f"{ds}{B}.br.lo", f(R, 3))
            put(f"{ds}{B}.br.hi", f(up, 3))
            put(f"{ds}{B}.br.gap", f(100 * (1 - R / up), 1))
            put(f"{ds}{B}.br.next", str(b2))
            gaps[f"{ds}{B}"] = 1 - R / up

    claim(f"brackets exist for church N=12, church N=16, bedroom N=12 only ({sorted(gaps)})",
          sorted(gaps) == ["bedroom12", "church12", "church16"])
    claim("widest bracket is bedroom N=12, narrowest church N=12",
          max(gaps, key=gaps.get) == "bedroom12" and min(gaps, key=gaps.get) == "church12")

    # ---- fixed schedule sets (as in bootstrap.sd_menu)
    post = bs.load(f"{N16}/sdpost_sd15_coco_B8_n64.npz")
    M = bs.f64(post["menu_M"])[0].copy()
    cors = bs.corrections(bs.sd(8))
    for tag, col in (("orig", None), ("ref", bs.f64(cors[-1][1]["D_best_B_batched"]))):
        if col is not None:
            M[:, 0] = col
        _, _, fi, fc = bs.menu_analysis(M, 8)
        RAW[f"menu.{tag}.in"] = [100 * float(x) for x in fi]
        RAW[f"menu.{tag}.cv"] = [100 * float(x) for x in fc]
        for K in (2, 4, 8):
            put(f"menu.{tag}.{K}.in", f(100 * fi[K - 1], 1))
            put(f"menu.{tag}.{K}.cv", f(100 * fc[K - 1], 1))
        if tag == "ref":
            put("menu.ref.4.cv.int", f(100 * fc[3], 0))

    # ---- other metrics, original schedules, per-prompt rollouts (Appendix E)
    names = [str(x) for x in post["metrics"]]
    for m, name in enumerate(names):
        sgn = -1.0 if name == "psnr" else 1.0      # orient so that improvement is positive
        u, s, s1, own, pps = (bs.f64(post[k])[m] for k in
                              ("uniform_B", "shared_B", "shared_B1", "perprompt_own",
                               "perprompt_sharedB"))
        gain, step, shg = sgn * (pps - own), sgn * (s - s1), sgn * (u - s)
        R0, Rb, Rj = bs.ratio_boot(gain, step, I64)
        o0, ob, oj = bs.ratio_boot(shg + gain, gain, I64)
        cR = bs.bca(R0, Rb, Rj)
        put_val(f"met.{name}.R", R0, cR, 2)
        put_val(f"met.{name}.om", o0, bs.bca(o0, ob, oj), 2)
        claim(f"{name}: per-prompt gain interval lies above one evaluation", cR[0] > 1)
    claim("Appendix E LPIPS value matches the stored per-prompt R of SD N=8",
          abs(float(NUM["met.lpips.R"]) - float(f(sd8["stored"]["R"], 2))) < 1e-12)

    # ---- reference check (Appendix D), from the ref_floor SUMMARY lines
    summ = {}
    for fn in sorted(os.listdir(N16)):
        if fn.startswith("refloor_") and fn.endswith(".txt"):
            for line in open(os.path.join(N16, fn)):
                if line.startswith("SUMMARY refloor"):
                    kv = dict(re.findall(r"(\S+?)=(\S+)", line))
                    summ[kv["row"]] = kv
    REF = [("LSUN church", "search_church256_B8_n64_sw10.npz"),
           ("LSUN church", "search_church256_B16_n64_sw10.npz"),
           ("LSUN church, DPM-Solver++", "search_church256_B8_n64_sw10_dpmpp2m.npz"),
           ("LSUN bedroom", "search_bedroom256_B8_n64_sw10.npz"),
           ("LSUN bedroom", "search_bedroom256_B12_n64_sw10.npz"),
           ("ImageNet", "search_cin256_B4_n256_sw10.npz"),
           ("ImageNet", "search_cin256_B8_n256_sw10.npz"),
           ("SD 1.5, guidance $7.5$", "search_sd15_coco_B4_n64_sw10.npz"),
           ("SD 1.5, guidance $7.5$", "search_sd15_coco_B8_n64_sw10.npz"),
           ("SD 1.5, guidance $3$", "search_sd15_coco_B8_n64_sw10_cfg3.npz")]

    def r3(s):           # 4-decimal summary value -> 3 decimals, refusing a double rounding
        if s.split(".")[1][3:4] == "5":
            raise SystemExit(f"summary value {s} cannot be rounded to 3 decimals")
        return f(float(s), 3)

    lines, worst, wrow, dpm = [], 0.0, None, None
    for model, row in REF:
        kv = summ[row]
        a, b = float(kv["R_sub200"]), float(kv["R_sub999"])
        rel = abs(b / a - 1)
        if rel > worst:
            worst, wrow = rel, row
        if "dpmpp2m" in row:
            dpm = rel
            put("ref.dpm.ratio", kv["ratio"])
        lines.append(rf"{model} & ${kv['B']}$ & ${f(float(kv['floor_mean']), 4)}$ & "
                     rf"${kv['ratio']}$ & ${r3(kv['R_sub200'])}$ & ${r3(kv['R_sub999'])}$ \\")
    TABLES["reference"] = lines
    put("ref.max", f(100 * worst, 1))
    put("ref.maxint", f(100 * worst, 0))
    put("ref.dpm", f(100 * dpm, 0))
    ratios = [float(kv["ratio"]) for kv in summ.values()]
    put("ref.minratio", f"{min(ratios):.1f}")
    claim(f"largest reference change is ImageNet N=8 ({wrow})", wrow == "search_cin256_B8_n256_sw10.npz")

    # ---- tables
    def main_row(k, model, sampler):
        w = rows[k]
        return (rf"{model} & {sampler} & ${w['B']}$ & ${w['n']}$ & ${NUM[k + '.D']}$ & "
                rf"${NUM[k + '.V']}$ & ${NUM[k + '.R']}$ & ${NUM[k + '.R.ci']}$ & "
                rf"${NUM[k + '.s']}\%$ \ ${NUM[k + '.s.ci']}$ \\")
    TABLES["main"] = [
        main_row("church4n256", "LSUN church  ", "DDIM"),
        main_row("church8n256", "LSUN church  ", "DDIM"),
        main_row("bedroom4n256", "LSUN bedroom ", "DDIM"),
        main_row("bedroom8n256", "LSUN bedroom ", "DDIM"), r"\midrule",
        main_row("cin4n256", "ImageNet     ", "DDIM"),
        main_row("cin8n256", "ImageNet     ", "DDIM"),
        main_row("sd4", "SD 1.5, guidance $7.5$", "DDIM"),
        main_row("sd8", "SD 1.5, guidance $7.5$", "DDIM"),
        main_row("dpm8n64", "LSUN church  ", "DPM-Solver++"), r"\midrule",
        main_row("sdgthree8", "SD 1.5, guidance $3$", "DDIM")]

    def budget_row(k, model, orig=False):
        w = rows[k]
        s = (rf"{model} & ${w['B']}$ & " + ("" if orig else rf"${w['n']}$ & ") +
             rf"${NUM[k + '.D']}$ & ${NUM[k + '.V']}$ & ${NUM[k + '.R']}$ & ${NUM[k + '.R.ci']}$ & "
             rf"${NUM[k + '.s']}\%$ \ ${NUM[k + '.s.ci']}$")
        return s + (rf" & ${NUM[k + '.Rorig']}$ \\" if orig else r" \\")
    TABLES["budget"] = [budget_row(k, m) for k, m in (
        ("church4n64", "LSUN church "), ("church8n64", "LSUN church "),
        ("church12n64", "LSUN church "), ("church12n256", "LSUN church "))] + [r"\midrule"] + [
        budget_row(k, "LSUN bedroom") for k in ("bedroom4n64", "bedroom8n64", "bedroom12n64")]
    TABLES["larger"] = [budget_row(k, m, orig=True) for k, m in (
        ("church16n64", "LSUN church "), ("church20n64", "LSUN church "),
        ("bedroom16n64", "LSUN bedroom"))]

    def refine_row(k, model):
        w = rows[k]
        return (rf"{model} & ${w['B']}$ & ${w['n']}$ & ${NUM[k + '.dN']}$ & ${NUM[k + '.dNone']}$ & "
                rf"${NUM[k + '.Rorig']}$ & ${NUM[k + '.R']}$ \\")
    ch = ["church4n64", "church4n256", "church8n64", "church8n256", "church12n64",
          "church12n256", "church16n64", "church20n64"]
    be = ["bedroom4n64", "bedroom4n256", "bedroom8n64", "bedroom8n256", "bedroom12n64",
          "bedroom16n64"]
    TABLES["refine"] = ([refine_row(k, "LSUN church ") for k in ch] + [r"\midrule"] +
                        [refine_row(k, "LSUN bedroom") for k in be] + [r"\midrule"] +
                        [refine_row("cin4n256", "ImageNet"), refine_row("cin8n256", "ImageNet"),
                         refine_row("sd4", "SD 1.5, guidance $7.5$"),
                         refine_row("sd8", "SD 1.5, guidance $7.5$"),
                         refine_row("sdgthree8", "SD 1.5, guidance $3$"),
                         refine_row("dpm8n64", "LSUN church, DPM-Solver++")])

    def om_row(k, model, sampler):
        w = rows[k]
        return (rf"{model} & {sampler} & ${w['B']}$ & ${NUM[k + '.om']}$ & ${NUM[k + '.om.ci']}$ & "
                rf"${NUM[k + '.cap']}\%$ \\")
    TABLES["omega"] = [
        om_row("church4n256", "LSUN church ", "DDIM"), om_row("church8n256", "LSUN church ", "DDIM"),
        om_row("church12n256", "LSUN church ", "DDIM"),
        om_row("bedroom4n256", "LSUN bedroom", "DDIM"), om_row("bedroom8n256", "LSUN bedroom", "DDIM"),
        r"\midrule",
        om_row("cin4n256", "ImageNet", "DDIM"), om_row("cin8n256", "ImageNet", "DDIM"),
        om_row("sd4", "SD 1.5, guidance $7.5$", "DDIM"), om_row("sd8", "SD 1.5, guidance $7.5$", "DDIM"),
        om_row("dpm8n64", "LSUN church ", "DPM-Solver++"), r"\midrule",
        om_row("sdgthree8", "SD 1.5, guidance $3$", "DDIM")]
    TABLES["menu"] = [rf"{lab} & ${K}$ & ${NUM[f'menu.{t}.{K}.in']}\%$ & ${NUM[f'menu.{t}.{K}.cv']}\%$ \\"
                      for t, lab in (("orig", "original"),) for K in (2, 4, 8)] + [r"\midrule"] + [
        rf"{lab} & ${K}$ & ${NUM[f'menu.{t}.{K}.in']}\%$ & ${NUM[f'menu.{t}.{K}.cv']}\%$ \\"
        for t, lab in (("ref", "refined "),) for K in (2, 4, 8)]

    # ---- write
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    with open(os.path.join(out, "numbers.tex"), "w") as fh:
        fh.write("% Generated by utils/paper_numbers.py from runs1/ and runs2/. "
                 "Do not edit by hand.\n")
        for k in sorted(NUM):
            fh.write(f"\\pndef{{{k}}}{{{NUM[k]}}}\n")
    for name, lines in TABLES.items():
        with open(os.path.join(out, "tables", f"{name}.tex"), "w") as fh:
            fh.write("% Generated by utils/paper_numbers.py. Do not edit by hand.\n")
            fh.write("\n".join(lines) + "\n")
    import json
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    with open(os.path.join(out, "figures", "figdata.json"), "w") as fh:
        json.dump(RAW, fh, indent=1, sort_keys=True)
    w = max(len(k) for k in NUM)
    for k in sorted(NUM):
        print(f"  {k:{w}s}  {NUM[k]}")
    print()
    bad = 0
    for text, ok in CLAIMS:
        print(f"  {'PASS' if ok else 'FAIL'}  {text}")
        bad += not ok
    print(f"\n{len(NUM)} numbers, {len(TABLES)} tables, {len(CLAIMS)} claims, {bad} failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
