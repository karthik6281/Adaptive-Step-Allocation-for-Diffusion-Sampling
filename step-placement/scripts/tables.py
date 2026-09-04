#!/usr/bin/env python3
"""Print the result tables, as text or as LaTeX bodies.

    python scripts/tables.py [--latex]
"""

import argparse
import sys

from steps import results
from steps.config import LABELS


def rule(header):
    print("-" * len(header))
    print(header)
    print("-" * len(header))


def cost(latex):
    rows = results.cost_table()
    if latex:
        print("% shared-mesh cost")
        for r in rows:
            print(
                f"{LABELS[r.dataset]:<13} & ${r.budget}$ & ${r.median:.2f}\\%$ & "
                f"${r.p90:.2f}\\%$ & ${r.p99:.2f}\\%$ & ${r.worst:.2f}\\%$ & "
                f"${r.distinct}$ & ${r.matching_shared}$ \\\\"
            )
        return

    print("\nexcess error of the optimal shared mesh")
    rule(f"{'model':<14}{'B':>3}{'median':>9}{'p90':>9}{'p99':>9}{'max':>9}{'uniq':>7}{'shared':>8}")
    for r in rows:
        print(
            f"{LABELS[r.dataset]:<14}{r.budget:>3}{r.median:>8.2f}%{r.p90:>8.2f}%"
            f"{r.p99:>8.2f}%{r.worst:>8.2f}%{r.distinct:>7}{r.matching_shared:>8}"
        )


def agreement(latex):
    rows = results.shape_table()
    if latex:
        print("\n% shape agreement")
        for r in rows:
            print(
                f"{LABELS[r.dataset]:<14} & ${r.magnitude:.2f}\\times$ & "
                f"${r.mean_rho:+.3f}$ & ${r.tail:+.3f}$ & ${r.minimum:+.3f}$ \\\\"
            )
        return

    print("\nprofile magnitude and shape agreement")
    rule(f"{'model':<14}{'magnitude':>11}{'mean rho':>10}{'1% tail':>10}{'min':>9}")
    for r in rows:
        print(
            f"{LABELS[r.dataset]:<14}{r.magnitude:>10.2f}x{r.mean_rho:>+10.3f}"
            f"{r.tail:>+10.3f}{r.minimum:>+9.3f}"
        )


def advantage(latex):
    rows = results.advantage_table()
    if latex:
        print("\n% per-image LPIPS advantage")
        for r in rows:
            lo, hi = r.interval
            print(
                f"{LABELS[r.dataset]:<13} & ${r.budget}$ & ${r.gain:.4f}$ & "
                f"${r.recovered:.3f}$ & $[{lo:.2f},\\,{hi:.2f}]$ & "
                f"${r.improved}/256$ & ${r.distinct}$ \\\\"
            )
        return

    print("\nper-image LPIPS advantage recovered by coordinate descent")
    rule(
        f"{'model':<14}{'N':>3}{'raw':>9}{'recovered':>11}"
        f"{'interval':>15}{'improved':>10}{'uniq':>6}"
    )
    for r in rows:
        lo, hi = r.interval
        print(
            f"{LABELS[r.dataset]:<14}{r.budget:>3}{r.gain:>9.4f}{r.recovered:>11.3f}"
            f"{f'[{lo:.2f}, {hi:.2f}]':>15}{f'{r.improved}/256':>10}{r.distinct:>6}"
        )


def proxy(latex):
    rows = results.proxy_table()
    if latex:
        print("\n% solver proxy")
        for r in rows:
            print(
                f"{LABELS[r.dataset]:<13} & ${r.raw_second:+.3f}$ & "
                f"${r.raw_third:+.3f}$ & ${r.excised_second:+.3f}$ & "
                f"${r.excised_third:+.3f}$ & ${r.third_wins}/64$ \\\\"
            )
        return

    print("\nsolver error against second- and third-difference proxies")
    rule(
        f"{'model':<14}{'raw 2nd':>10}{'raw 3rd':>10}"
        f"{'exc 2nd':>10}{'exc 3rd':>10}{'3rd wins':>10}"
    )
    for r in rows:
        print(
            f"{LABELS[r.dataset]:<14}{r.raw_second:>+10.3f}{r.raw_third:>+10.3f}"
            f"{r.excised_second:>+10.3f}{r.excised_third:>+10.3f}"
            f"{f'{r.third_wins}/64':>10}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex", action="store_true")
    latex = parser.parse_args(argv).latex

    for section in (cost, agreement, advantage, proxy):
        section(latex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
