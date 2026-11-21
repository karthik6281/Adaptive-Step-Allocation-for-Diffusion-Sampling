#!/usr/bin/env python3
"""Figures for the step-placement paper.

Reads figures/figdata.json, which utils/paper_numbers.py writes from the run files, so every
plotted value is the one the tables print. Writes a vector PDF (for LaTeX) and a 600-dpi PNG
of each figure next to the data file.

    python3 utils/paper_figures.py --dir report/tmlr_v3/figures
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

# Chart ink and the first two categorical slots of the validated reference palette
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE = "#2a78d6", "#eb6834"
TEXTW = 6.5            # \textwidth of the TMLR template, inches
MS = 4.6               # marker size, points
LW = 1.1               # data line width, points
CI_LW = 1.0            # interval line width, points


def setup():
    sup = "/System/Library/Fonts/Supplemental"
    for f in ("Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf"):
        if os.path.exists(os.path.join(sup, f)):
            font_manager.fontManager.addfont(os.path.join(sup, f))
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "mathtext.default": "it",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 0,
        "ytick.minor.size": 0, "grid.color": GRID, "grid.linewidth": 0.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.4, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 6,
    })


def save(fig, d, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


def point(ax, x, y, color, filled=True, z=4):
    if filled:
        ax.plot(x, y, "o", ms=MS, mfc=color, mec=SURFACE, mew=0.9, zorder=z)
    else:
        ax.plot(x, y, "o", ms=MS - 0.4, mfc=SURFACE, mec=color, mew=1.1, zorder=z)


def hgrid(ax, axis="x"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- figures
def fig_shares(D, d, color=BLUE, width=TEXTW - 0.22, name="fig_shares", height=2.25):
    """Share of the budget by setting, N=4 (open) and N=8 (filled)."""
    rows = [("LSUN bedroom", "bedroom4n256", "bedroom8n256"),
            ("LSUN church", "church4n256", "church8n256"),
            ("SD 1.5, guidance 3", None, "sdgthree8"),
            ("SD 1.5, guidance 7.5", "sd4", "sd8"),
            ("ImageNet", "cin4n256", "cin8n256"),
            ("LSUN church, DPM-Solver++", None, "dpm8n64")]
    fig, ax = plt.subplots(figsize=(width, height))
    n = len(rows)
    off = 0.15
    for i, (label, k4, k8) in enumerate(rows):
        y = n - 1 - i
        for k, dy, filled in ((k4, off, False), (k8, -off, True)):
            if k is None:
                continue
            v, lo, hi = D[k + ".s"]
            ax.plot([lo, hi], [y + dy] * 2, color=color, lw=CI_LW, solid_capstyle="round", zorder=3)
            point(ax, v, y + dy, color, filled)
    ax.axhline(n - 3.5, color=AXIS, lw=0.8, zorder=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.tick_params(axis="y", length=0, labelcolor=INK, pad=6)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks(range(0, 51, 10))
    ax.set_xticklabels([f"{t}%" for t in range(0, 51, 10)])
    ax.set_xlabel(r"Share of the budget, $R_N/N$")
    hgrid(ax, "x")
    handles = [Line2D([], [], ls="", marker="o", ms=MS - 0.4, mfc=SURFACE, mec=color, mew=1.1),
               Line2D([], [], ls="", marker="o", ms=MS, mfc=color, mec=SURFACE, mew=0.9)]
    ax.legend(handles, ["$N$ = 4", "$N$ = 8"], loc="lower right", ncol=2,
              bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    save(fig, d, name)


def fig_paired(D, d):
    """Ratio of R between two settings on the same images, split into its two factors."""
    rows = [("q", r"$R_N$ ratio, $R^A/R^B$"), ("g", r"Gain ratio, $\Delta^A/\Delta^B$"),
            ("st", r"Step-value ratio, $V^B/V^A$")]
    panels = [("dpm", "DPM-Solver++ over DDIM", "LSUN church, $N$ = 8, 64 images"),
              ("guid", "Guidance 7.5 over guidance 3", "SD 1.5, $N$ = 8, 64 prompts")]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW - 0.05, 1.6), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    for ax, (key, title, sub) in zip(axes, panels):
        for i, (f, _) in enumerate(rows):
            y = len(rows) - 1 - i
            v, lo, hi = D[f"pair.{key}.{f}"]
            ax.plot([lo, hi], [y, y], color=BLUE, lw=CI_LW, solid_capstyle="round", zorder=3)
            point(ax, v, y, BLUE)
            ax.text(hi * 1.07, y, f"{v:.2f}", va="center", ha="left", fontsize=7.5, color=INK2)
        ax.axvline(1, color=MUTED, lw=0.8, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.45, 11)
        ticks = [0.5, 1, 2, 4, 8]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(f"{title}\n", loc="left")
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
        ax.tick_params(axis="y", length=0, labelcolor=INK, pad=6)
        ax.spines["left"].set_visible(False)
        hgrid(ax, "x")
        ax.set_xlabel("Ratio (log scale)")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[1] for r in rows][::-1])
    save(fig, d, "fig_paired")


def fig_budget(D, d):
    """R_N (top) and the share of the budget (bottom) against N, one column per LSUN model."""
    cols = [("church", "LSUN church", BLUE, {64: [4, 8, 12, 16, 20], 256: [4, 8, 12]}),
            ("bedroom", "LSUN bedroom", ORANGE, {64: [4, 8, 12, 16], 256: [4, 8]})]
    fig, axes = plt.subplots(2, 2, figsize=(TEXTW + 0.9, 3.75), sharex=True, sharey="row")
    fig.subplots_adjust(wspace=0.08, hspace=0.16)
    for j, (ds, name, color, budgets) in enumerate(cols):
        for i, field in enumerate(("R", "s")):
            ax = axes[i, j]
            for n, dx in ((64, -0.22), (256, 0.22)):
                xs, vs = [], []
                for B in budgets[n]:
                    v, lo, hi = D[f"{ds}{B}n{n}.{field}"]
                    ax.plot([B + dx] * 2, [lo, hi], color=color, lw=CI_LW,
                            solid_capstyle="round", zorder=3)
                    xs.append(B + dx)
                    vs.append(v)
                ax.plot(xs, vs, color=color, lw=LW if n == 64 else 0.9,
                        alpha=1.0 if n == 64 else 0.6, zorder=2, solid_joinstyle="round")
                for x, v in zip(xs, vs):
                    point(ax, x, v, color, filled=(n == 256))
            hgrid(ax, "y")
            ax.set_xlim(2.6, 21.4)
            ax.set_xticks([4, 8, 12, 16, 20])
            if i == 0:
                ax.axhline(1, color=MUTED, lw=0.8, zorder=1)
                ax.set_title(name, loc="left")
            else:
                ax.set_xlabel("Evaluations $N$")
            if j == 1:
                ax.tick_params(axis="y", length=0)
                ax.spines["left"].set_visible(False)
    axes[0, 0].set_ylim(0, 4)
    axes[0, 0].set_yticks([0, 1, 2, 3, 4])
    axes[0, 0].set_ylabel("$R_N$, in extra evaluations")
    axes[1, 0].set_ylim(0, 20)
    axes[1, 0].set_yticks([0, 5, 10, 15, 20])
    axes[1, 0].set_yticklabels([f"{t}%" for t in (0, 5, 10, 15, 20)])
    axes[1, 0].set_ylabel("$R_N/N$, share of the budget")
    fig.align_ylabels(axes[:, 0])
    handles = [Line2D([], [], color=MUTED, lw=LW, marker="o", ms=MS - 0.4, mfc=SURFACE,
                      mec=MUTED, mew=1.1),
               Line2D([], [], color=MUTED, lw=0.9, alpha=0.8, marker="o", ms=MS, mfc=MUTED,
                      mec=MUTED)]
    fig.legend(handles, ["64 images", "256 images"], loc="lower right", ncol=2,
               bbox_to_anchor=(0.905, 0.955))
    save(fig, d, "fig_budget")


def fig_omega(D, d):
    """Part of the improvement from uniform spacing that the tuned shared schedule provides."""
    rows = [("cin8n256", "ImageNet, $N$ = 8"), ("bedroom4n256", "LSUN bedroom, $N$ = 4"),
            ("cin4n256", "ImageNet, $N$ = 4"), ("dpm8n64", "LSUN church, DPM-Solver++, $N$ = 8"),
            ("bedroom8n256", "LSUN bedroom, $N$ = 8"), ("sd4", "SD 1.5, guidance 7.5, $N$ = 4"),
            ("sd8", "SD 1.5, guidance 7.5, $N$ = 8"), ("church12n256", "LSUN church, $N$ = 12"),
            ("church8n256", "LSUN church, $N$ = 8"), ("sdgthree8", "SD 1.5, guidance 3, $N$ = 8"),
            ("church4n256", "LSUN church, $N$ = 4")]
    caps = [D[k + ".cap"][0] for k, _ in rows]
    if caps != sorted(caps, reverse=True):
        raise SystemExit("fig_omega: rows are not sorted by the shared part")
    n = len(rows)
    W, H = TEXTW - 0.8, 2.95
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, n - 0.4)
    fig.canvas.draw()
    bb = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    aspect = ((n - 0.4 + 0.6) / bb.height) / (100 / bb.width)
    h, gap, r = 0.56, 0.45, 0.9
    for i, (k, label) in enumerate(rows):
        y = n - 1 - i
        cap = D[k + ".cap"][0]
        parts = [(0, cap, BLUE), (cap, 100, ORANGE)]
        for j, (a, b, color) in enumerate(parts):
            x0 = a + (gap / 2 if j == 1 else 0)
            x1 = b - (gap / 2 if j == 0 else 0)
            ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h,
                                        boxstyle=f"round,pad=0,rounding_size={r}",
                                        mutation_aspect=aspect, fc=color, ec="none", zorder=3))
            share = b - a
            if share >= 9:
                ax.text(x0 + 1.4, y, f"{share:.0f}%", ha="left", va="center", fontsize=7,
                        color="#ffffff", zorder=4, fontweight="bold")
        om = D[k + ".om"][0]
        ax.text(101.5, y, f"{om:.2f}", ha="left", va="center", fontsize=7.5, color=INK2)
    ax.text(101.5, n - 0.25, r"$\Omega_N$", ha="left", va="bottom", fontsize=7.5, color=INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in rows][::-1])
    ax.tick_params(axis="y", length=0, labelcolor=INK, pad=6)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    handles = [Patch(fc=BLUE, ec="none"), Patch(fc=ORANGE, ec="none")]
    ax.legend(handles, ["Provided by the tuned shared schedule", "Added by per-sample placement"],
              loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.3,
              handlelength=1.1, handleheight=0.9)
    save(fig, d, "fig_omega")


def fig_menu(D, d):
    """Fraction of the per-prompt gain recovered by K fixed schedules (refined shared schedule)."""
    K = np.arange(1, 9)
    fin, fcv = D["menu.ref.in"], D["menu.ref.cv"]
    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    for vals, color, label in ((fin, BLUE, "Prompts used for selection"),
                               (fcv, ORANGE, "Held-out prompts")):
        ax.plot(K, vals, color=color, lw=LW, zorder=2, solid_joinstyle="round")
        for x, v in zip(K, vals):
            point(ax, x, v, color)
        ax.text(8.25, vals[-1], f"{vals[-1]:.0f}%", va="center", ha="left", fontsize=7.5,
                color=INK2)
    ax.set_xlim(0.6, 8.9)
    ax.set_ylim(0, 100)
    ax.set_xticks(K)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{t}%" for t in (0, 25, 50, 75, 100)])
    ax.set_xlabel(r"Schedules in the set, $m$")
    ax.set_ylabel("Per-prompt gain recovered")
    hgrid(ax, "y")
    handles = [Line2D([], [], color=BLUE, lw=LW, marker="o", ms=MS, mfc=BLUE, mec=BLUE),
               Line2D([], [], color=ORANGE, lw=LW, marker="o", ms=MS, mfc=ORANGE, mec=ORANGE)]
    ax.legend(handles, ["Prompts used for selection", "Held-out prompts"], loc="upper left",
              borderaxespad=0.2)
    save(fig, d, "fig_menu")


def fig_convex(D, d):
    """Mean LPIPS of the shared schedule against the budget, 64 images."""
    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    for ds, color, label in (("church", BLUE, "LSUN church"), ("bedroom", ORANGE, "LSUN bedroom")):
        pts = np.array(D[f"pts.{ds}"])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=LW, zorder=2, solid_joinstyle="round")
        for x, v in pts:
            point(ax, x, v, color)
    ax.set_xlim(3, 22)
    ax.set_xticks([4, 8, 12, 16, 20])
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel(r"Evaluations $N$")
    ax.set_ylabel("Mean LPIPS of the shared schedule")
    hgrid(ax, "y")
    handles = [Line2D([], [], color=BLUE, lw=LW, marker="o", ms=MS, mfc=BLUE, mec=BLUE),
               Line2D([], [], color=ORANGE, lw=LW, marker="o", ms=MS, mfc=ORANGE, mec=ORANGE)]
    ax.legend(handles, ["LSUN church", "LSUN bedroom"], loc="upper right", borderaxespad=0.2)
    save(fig, d, "fig_convex")


N8_ROWS = [("bedroom8n256", "LSUN bedroom"), ("church8n256", "LSUN church"),
           ("sdgthree8", "SD 1.5, guidance 3"), ("sd8", "SD 1.5, guidance 7.5"),
           ("cin8n256", "ImageNet"), ("dpm8n64", "LSUN church, DPM-Solver++")]


def fig_schedules(D, d):
    """Interior timesteps of every per-image schedule at N=8, around the shared schedule."""
    uniform = [int(round(999 * (1 - j / 8))) for j in range(1, 8)]
    n = len(N8_ROWS)
    fig, ax = plt.subplots(figsize=(TEXTW - 0.2, 2.7))
    rng = np.random.default_rng(0)
    for i, (k, label) in enumerate(N8_ROWS):
        y = n - 1 - i
        S = D[f"sched.{k}"]
        per = np.asarray(S["per"])[:, 1:-1]
        jit = rng.uniform(-0.3, 0.3, size=per.shape)
        ax.scatter(per.ravel(), (y + jit).ravel(), s=2.2, color=BLUE, alpha=0.16, lw=0,
                   zorder=2, rasterized=True)
        for t in S["shared"][1:-1]:
            ax.plot([t, t], [y - 0.36, y + 0.36], color=INK, lw=1.3, solid_capstyle="round",
                    zorder=3)
    ax.set_xlim(1000, 0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks(uniform)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in N8_ROWS][::-1])
    ax.tick_params(axis="y", length=0, labelcolor=INK, pad=6)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Timestep (sampling runs from left to right; gridlines mark uniform spacing)")
    handles = [Line2D([], [], ls="", marker="o", ms=3, mfc=BLUE, mec="none", alpha=0.6),
               Line2D([], [], color=INK, lw=1.3)]
    ax.legend(handles, ["Per-image schedules", "Shared schedule of the main search"],
              loc="lower right", ncol=2, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    save(fig, d, "fig_schedules")


def fig_pergain(D, d):
    """Each image's gain over the refined shared schedule, in units of one extra evaluation."""
    fig, axes = plt.subplots(2, 3, figsize=(TEXTW - 0.2, 3.3), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.1, hspace=0.42)
    hi = 8.0
    bins = np.linspace(0, hi, 33)
    for ax, (k, label) in zip(axes.ravel(), N8_ROWS):
        g = np.asarray(D[f"pergain.{k}"])
        w = np.full(len(g), 100.0 / len(g))
        ax.hist(np.clip(g, 0, hi - 1e-9), bins=bins, weights=w, color=BLUE, alpha=0.85,
                edgecolor=SURFACE, linewidth=0.6, zorder=2)
        m = float(g.mean())
        ax.axvline(m, color=INK, lw=1.0, zorder=3)
        ax.text(m + 0.15, 0.96, f"$R_8$ = {m:.2f}", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=7.2, color=INK)
        ax.set_title(label, loc="left", fontsize=8)
        hgrid(ax, "y")
    fig.supxlabel("Per-image gain, in extra evaluations", fontsize=8, y=0.0)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of images")
    axes[0, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    axes[0, 0].set_xlim(0, hi)
    axes[0, 0].set_xticks([0, 2, 4, 6, 8])
    axes[0, 0].set_xticklabels(["0", "2", "4", "6", "8+"])
    save(fig, d, "fig_pergain")


def fig_profiles(d, steps_root):
    """Local-error profiles of the five models (Section 5), with the shared B=8 optimum."""
    import sys
    sys.path.insert(0, os.path.join(steps_root, "src"))
    from steps import load, mesh  # the package in the step-placement repository
    grid = np.linspace(999, 0, 201).round().astype(int)
    models = [("church256", "LSUN church"), ("bedroom256", "LSUN bedroom"),
              ("celeb256", "CelebA-HQ"), ("ffhq256", "FFHQ"), ("cin256", "ImageNet")]
    fig, axes = plt.subplots(1, 5, figsize=(TEXTW - 0.2, 1.9), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    for ax, (ds, label) in zip(axes, models):
        prof, nodes = load.profile(ds)
        shapes = mesh.normalise(prof)
        t = grid[nodes.astype(int) + 1]          # column i is centred at grid[i + 1]
        lo, med, up = np.quantile(shapes, [0.1, 0.5, 0.9], axis=0)
        ax.fill_between(t, lo, up, color=BLUE, alpha=0.18, lw=0, zorder=2)
        ax.plot(t, med, color=BLUE, lw=0.9, zorder=3)
        shared = mesh.best_mesh(nodes, shapes.mean(0), budget=8)
        for s in np.asarray(shared, int):
            ax.plot([t[s]] * 2, [0, 0.07], color=INK, lw=1.1, transform=ax.get_xaxis_transform(),
                    zorder=4, clip_on=False)
        ax.set_xlim(1000, 0)
        ax.set_yscale("log")
        ax.set_xticks([750, 500, 250])
        ax.set_title(label, loc="left", fontsize=8)
        ax.grid(True, axis="y", which="major", color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        if ax is not axes[0]:
            ax.tick_params(axis="y", which="both", length=0)
    axes[0].set_ylabel("Error profile (normalised)")
    axes[2].set_xlabel("Timestep")
    save(fig, d, "fig_profiles")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder holding figdata.json")
    ap.add_argument("--steps-root", default=None,
                    help="step-placement folder (src/steps, data/profiles) for the profile figure")
    args = ap.parse_args()
    with open(os.path.join(args.dir, "figdata.json")) as fh:
        D = json.load(fh)
    setup()
    for f in (fig_shares, fig_paired, fig_budget, fig_omega, fig_menu, fig_convex,
              fig_schedules, fig_pergain):
        f(D, args.dir)
        print("wrote", f.__name__)
    if args.steps_root:
        fig_profiles(args.dir, args.steps_root)
        print("wrote fig_profiles")
    # research summary (files/research_summary.tex): its navy accent, A4 text width
    with plt.rc_context({"font.size": 8.8, "axes.labelsize": 8.8, "xtick.labelsize": 8.4,
                         "ytick.labelsize": 8.4, "legend.fontsize": 8.4}):
        fig_shares(D, args.dir, color="#1F4E79", width=6.45, height=2.1,
                   name="fig_shares_summary")
    print("wrote fig_shares_summary")


if __name__ == "__main__":
    main()
