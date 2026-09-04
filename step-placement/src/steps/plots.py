"""pgfplots fragments for the two figures."""

import numpy as np

from . import load
from .config import DATASETS, LABELS, MAX_T, MEASUREMENT_STEPS
from .mesh import excess, normalise

PROFILE_STYLES = (
    "profileBlue, solid, mark=*",
    "profileVermillion, dashed, mark=square*",
    "profileGreen, dashdotted, mark=triangle*",
    "profileOrange, densely dotted, mark=diamond*",
    "profileBlack, densely dashdotdotted, mark=x",
)
COST_STYLES = (
    ("blue!70!black", "*"),
    ("red!75!black", "square*"),
    ("teal!80!black", "triangle*"),
    ("orange!85!black", "diamond*"),
    ("black", "x"),
)


def _curve(style, points, label):
    return f"\\addplot[{style}] coordinates {{{points}}};\n\\addlegendentry{{{label}}}\n"


def profiles():
    """Mean normalised profile per model over the retained columns."""
    schedule = np.linspace(MAX_T - 1, 0, MEASUREMENT_STEPS + 1).round().astype(int)
    groups = None
    curves = []
    band_x = band_values = None
    for dataset in DATASETS:
        profile, nodes = load.profile(dataset)
        profile = normalise(profile)
        if groups is None:
            groups = np.array_split(np.arange(profile.shape[1]), 24)
        x = schedule[nodes.astype(int) + 1]
        centres = np.array([x[group].mean() for group in groups])
        values = 100 * np.stack(
            [profile[:, group].mean(axis=1) for group in groups], axis=1
        )
        curves.append((centres, values.mean(axis=0)))
        if dataset == "church256":
            band_x, band_values = centres, values

    lo = np.quantile(band_values, 0.10, axis=0)
    hi = np.quantile(band_values, 0.90, axis=0)
    fill = (
        " ".join(f"({x:.1f},{y:.5f})" for x, y in zip(band_x, hi))
        + " "
        + " ".join(f"({x:.1f},{y:.5f})" for x, y in zip(band_x[::-1], lo[::-1]))
    )
    lines = [
        f"\\addplot[draw=none, fill=profileBlue!14, forget plot] "
        f"coordinates {{{fill}}} \\closedcycle;"
    ]
    for style, dataset, (x, mean) in zip(PROFILE_STYLES, DATASETS, curves):
        points = " ".join(f"({a:.1f},{b:.5f})" for a, b in zip(x, mean))
        lines.append(
            f"\\addplot[{style}, very thick, mark repeat=4, mark size=1.2pt, "
            f"mark options={{solid}}] coordinates {{{points}}};\n"
            f"\\addlegendentry{{{LABELS[dataset]}}}"
        )
    lines.append(
        "\\addlegendimage{area legend, draw=none, fill=profileBlue!14}\n"
        "\\addlegendentry{church 10--90\\%}"
    )

    return rf"""\definecolor{{profileBlue}}{{HTML}}{{0072B2}}
\definecolor{{profileVermillion}}{{HTML}}{{D55E00}}
\definecolor{{profileGreen}}{{HTML}}{{009E73}}
\definecolor{{profileOrange}}{{HTML}}{{E69F00}}
\definecolor{{profileBlack}}{{HTML}}{{222222}}
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.96\linewidth, height=2.05in,
  xmin=50, xmax=994, x dir=reverse,
  ymode=log, log basis y=10, ymin=0.02, ymax=3.2,
  xtick={{994,750,500,250,50}}, ytick={{0.03,0.1,0.3,1,3}},
  yticklabels={{0.03,0.1,0.3,1,3}},
  xlabel={{DDIM timestep $t$ (noise $\rightarrow$ image)}},
    ylabel={{normalised profile weight per point (\%)}},
  xlabel style={{font=\footnotesize}}, ylabel style={{font=\footnotesize}},
  tick label style={{font=\footnotesize}},
  legend style={{font=\scriptsize, draw=none, fill=none,
                at={{(0.5,1.03)}}, anchor=south, legend columns=3,
                /tikz/every even column/.append style={{column sep=7pt}}}},
  legend cell align={{left}}, ymajorgrids=true, grid style={{black!12}},
  axis line style={{black!65}}, tick style={{black!65}},
]
{chr(10).join(lines)}
\end{{axis}}
\end{{tikzpicture}}
"""


def _cdf(values, xmax=12.0):
    values = np.sort(values)
    x = np.unique(values[values <= xmax])
    y = np.searchsorted(values, x, side="right") / len(values)
    if not len(x) or x[-1] < xmax:
        x = np.append(x, xmax)
        y = np.append(y, np.searchsorted(values, xmax, side="right") / len(values))
    return x, y


def _survival(values, xmin=5.0):
    values = np.sort(values)
    x = np.unique(values[values >= xmin])
    x = np.concatenate(([xmin], x[x > xmin]))
    y = (len(values) - np.searchsorted(values, x, side="left")) / len(values)
    return x, y


def _points(x, y):
    return " ".join(f"({a:.3f},{b:.5f})" for a, b in zip(x, y))


def cost(budget=8):
    """Central CDF and log-scale upper tail of shared-mesh excess."""
    central, tail = [], []
    for (color, marker), dataset in zip(COST_STYLES, DATASETS):
        prof, nodes = load.profile(dataset)
        values = excess(normalise(prof), nodes, budget)
        style = (
            f"{color}, solid, thick, mark={marker}, mark size=1.25pt, "
            "mark repeat=18, mark options={solid}, const plot"
        )
        x, y = _cdf(values)
        central.append(
            f"\\addplot[{style}] coordinates {{{_points(x, y)}}};\n"
            f"\\addlegendentry{{{LABELS[dataset]}}}"
        )
        x, y = _survival(values)
        tail.append(f"\\addplot[{style}] coordinates {{{_points(x, y)}}};")

    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.96\linewidth, height=1.65in,
  xmin=0, xmax=12, ymin=0, ymax=1.03,
  xlabel={{excess cost $g_k$ (\%)}}, ylabel={{$\Pr(g_k\leq x)$}},
  xlabel style={{font=\footnotesize}}, ylabel style={{font=\footnotesize}},
  tick label style={{font=\footnotesize}},
  legend style={{font=\scriptsize, draw=none, fill=none,
                at={{(0.5,1.04)}}, anchor=south, legend columns=3,
                /tikz/every even column/.append style={{column sep=5pt}}}},
  xtick={{0,2,4,6,8,10,12}}, ytick={{0,0.25,0.5,0.75,1}},
  grid=major, grid style={{gray!20}},
]
{chr(10).join(central)}
\node[anchor=north west, font=\footnotesize, fill=white, inner sep=1pt]
  at (rel axis cs:0.01,0.98) {{(a) central range}};
\end{{axis}}
\end{{tikzpicture}}
\par\vspace{{0.25em}}
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.96\linewidth, height=1.65in,
  xmin=5, xmax=88, ymode=log, ymin=0.0035, ymax=0.4,
  xlabel={{excess cost $g_k$ (\%)}}, ylabel={{$\Pr(g_k\geq x)$}},
  xlabel style={{font=\footnotesize}}, ylabel style={{font=\footnotesize}},
  tick label style={{font=\footnotesize}}, xtick={{5,20,40,60,80}},
  ytick={{0.00390625,0.01,0.1}}, yticklabels={{$1/256$,$0.01$,$0.1$}},
  grid=major, grid style={{gray!20}},
]
{chr(10).join(tail)}
\node[anchor=north west, font=\footnotesize, fill=white, inner sep=1pt]
  at (rel axis cs:0.01,0.98) {{(b) upper tail}};
\end{{axis}}
\end{{tikzpicture}}
"""


def write(destination):
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in (("fig_profile.tex", profiles()), ("fig_cost.tex", cost())):
        path = destination / name
        path.write_text(text)
        written.append(path)
    return written
