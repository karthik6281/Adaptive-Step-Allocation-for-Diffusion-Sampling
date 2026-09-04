#!/usr/bin/env python3
"""Recompute every published value and compare against what the paper prints.

Exits non-zero on the first disagreement beyond printed precision, so it can
gate a build.

    python scripts/check.py [-v]
"""

import argparse
import sys

from steps import config, results
from steps.load import profile
from steps.shape import mean_shape_share

FAILURES = []
CHECKS = 0
VERBOSE = False


def near(want, got, places, what):
    """Compare to the precision the paper prints."""
    global CHECKS
    CHECKS += 1
    if abs(round(got, places) - want) > 0.5 * 10.0**-places:
        FAILURES.append(f"{what}: paper {want:.{places}f}, recomputed {got:.{places}f}")
    elif VERBOSE:
        print(f"  ok  {what:<44} {got:.{places}f}")


def same(want, got, what):
    """Counts and intervals, where rounding does not apply."""
    global CHECKS
    CHECKS += 1
    if want != got:
        FAILURES.append(f"{what}: paper {want}, recomputed {got}")
    elif VERBOSE:
        print(f"  ok  {what:<44} {got}")


def check_cost():
    print("\nshared-mesh cost")
    for row in results.cost_table():
        want = config.COST[(row.dataset, row.budget)]
        cell = f"{config.LABELS[row.dataset]} B={row.budget}"
        for name, w, g, dp in zip(
            ("median", "p90", "p99", "max"), want, row[2:6], (2, 2, 2, 2)
        ):
            near(w, g, dp, f"{cell} {name}")
        same(want[4], row.distinct, f"{cell} distinct")
        same(want[5], row.matching_shared, f"{cell} shared")


def check_shape():
    print("\nshape agreement")
    for row in results.shape_table():
        want = config.SHAPE[row.dataset]
        for name, w, g, dp in zip(
            ("magnitude", "mean rho", "1% tail", "min"), want, row[1:], (2, 3, 3, 3)
        ):
            near(w, g, dp, f"{config.LABELS[row.dataset]} {name}")


def check_advantage():
    print("\nper-image LPIPS advantage")
    for row in results.advantage_table():
        gain, recovered, ci, improved, distinct = config.ADVANTAGE[
            (row.dataset, row.budget)
        ]
        cell = f"{config.LABELS[row.dataset]} N={row.budget}"
        near(gain, row.gain, 4, f"{cell} raw gap")
        near(recovered, row.recovered, 3, f"{cell} recovered")
        same(ci, row.interval, f"{cell} interval")
        same(improved, row.improved, f"{cell} improved")
        same(distinct, row.distinct, f"{cell} distinct schedules")


def check_proxy():
    print("\nsolver proxy")
    for row in results.proxy_table():
        *want, want_wins = config.PROXY[row.dataset]
        names = ("raw vs 2nd", "raw vs 3rd", "excised vs 2nd", "excised vs 3rd")
        for name, w, g in zip(names, want, row[1:5]):
            near(w, g, 3, f"{config.LABELS[row.dataset]} {name}")
        same(want_wins, row.third_wins, f"{config.LABELS[row.dataset]} 3rd wins")


def check_derived():
    print("\nmean-shape share and search calibration")
    lo, hi = config.MEAN_SHAPE_SHARE
    shares = [100 * mean_shape_share(profile(d)[0]) for d in config.DATASETS]
    near(lo, min(shares), 1, "mean-shape share, lowest model")
    near(hi, max(shares), 1, "mean-shape share, highest model")

    names = ("lowest cell mean", "highest cell mean", "worst sample")
    for name, w, g in zip(names, config.CALIBRATION, results.calibration()):
        near(w, g, 2, f"calibration, {name}")


def main(argv=None):
    global VERBOSE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    VERBOSE = parser.parse_args(argv).verbose

    FAILURES.clear()
    for step in (check_cost, check_shape, check_advantage, check_proxy, check_derived):
        step()

    print("\n" + "=" * 66)
    if FAILURES:
        print(f"FAIL  {len(FAILURES)} of {CHECKS} disagree")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"OK    all {CHECKS} published values reproduce from the shipped arrays")
    return 0


if __name__ == "__main__":
    sys.exit(main())
