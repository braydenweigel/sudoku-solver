"""Split a record file into training and test files by puzzle.

Usage:
    python -m sudoku.split --input data/manual.parquet --test-count 2 --seed 0

Writes <input>_train.parquet and <input>_test.parquet next to the input, keeping
its source label. Generate augmented records from each file separately, so no
puzzle appears in both splits in any form.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .data import read_kinds, read_records, write_records


def split_indices(n, test_count, rng):
    """Randomly pick `test_count` of `n` indices for testing; returns (train, test), each sorted."""
    if not 1 <= test_count < n:
        raise ValueError(f"test_count must be between 1 and {n - 1}")
    order = rng.permutation(n)
    return sorted(order[test_count:].tolist()), sorted(order[:test_count].tolist())


def split_records(records, test_count, rng):
    """Randomly pick `test_count` records for testing; the rest are for training."""
    train, test = split_indices(len(records), test_count, rng)
    return [records[i] for i in train], [records[i] for i in test]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="data/manual.parquet")
    parser.add_argument("--test-count", type=int, required=True, help="number of puzzles to hold out for testing")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    sources = set(pq.read_table(args.input, columns=["source"]).column("source").to_pylist())
    if len(sources) != 1:
        raise ValueError(f"expected one source label in {args.input}, found {sorted(sources)}")
    source = sources.pop()

    path = Path(args.input)
    records, kinds = read_records(path), read_kinds(path)
    train, test = split_indices(len(records), args.test_count, np.random.default_rng(args.seed))
    for name, indices in (("train", train), ("test", test)):
        out = path.with_name(f"{path.stem}_{name}{path.suffix}")
        write_records(
            out, [records[i] for i in indices], source=source, kinds=kinds and [kinds[i] for i in indices]
        )
        print(f"wrote {len(indices)} records to {out}")


if __name__ == "__main__":
    main()
