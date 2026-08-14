"""Generate the benchmark input matrices.

These are not committed to the repository: a 3072x3072 matrix written as text is
225 MB, which is over GitHub's hard 100 MB per-file limit, and the full set is
~1.4 GB. They are inputs, and inputs a script can rebuild do not belong in
version control.

The seed is the important part of this file. Without one, every regeneration
produced a different pair of matrices, so a timing measured today was taken on
different input from one measured last week. Matrix multiply is O(n^3) regardless
of the values, so the numbers stayed *comparable* - they were just not
*repeatable*, and "repeatable" is the claim a benchmark makes.

Usage:
    python datasets/generate_matrices.py                  # all sizes (~1.4 GB)
    python datasets/generate_matrices.py --sizes 256 512  # just the small ones
    python datasets/generate_matrices.py --force          # rewrite existing files
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_SIZES = [256, 512, 1024, 1536, 2048, 3072]
# Any fixed value works; it only has to stay fixed. Changing it invalidates every
# timing in results/ that was measured against the old matrices.
SEED = 20250429


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write into (default: alongside this script)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite files that already exist instead of skipping them",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for n in sorted(args.sizes):
        paths = {
            "A": args.out / f"matrixA_{n}.txt",
            "B": args.out / f"matrixB_{n}.txt",
        }
        if not args.force and all(p.exists() for p in paths.values()):
            print(f"{n:>5}x{n}: already present, skipping (use --force to rewrite)")
            continue

        # Seeded per size, so generating only the 512 matrices produces a
        # byte-identical file to generating the whole set. Seeding once up front
        # would make each matrix depend on which sizes were requested before it.
        rng = np.random.default_rng(args.seed + n)
        for path in paths.values():
            np.savetxt(path, rng.random((n, n)))
            print(f"{n:>5}x{n}: wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

    print("\nDatasets generated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
