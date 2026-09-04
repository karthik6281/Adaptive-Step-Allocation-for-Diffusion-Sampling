#!/usr/bin/env python3
"""Regenerate the two pgfplots fragments.

    python scripts/figures.py --out build
"""

import argparse
import sys
from pathlib import Path

from steps import plots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build"))
    args = parser.parse_args(argv)

    for path in plots.write(args.out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
