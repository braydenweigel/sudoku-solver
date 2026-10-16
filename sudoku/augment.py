"""Generate new records by applying random validity-preserving operations to existing ones.

Usage:
    python -m sudoku.augment --input data/manual.parquet --output data/generated.parquet --count 1000

Each operation is a function `rng -> (perm, digit_map)` in the same form as
`random_symmetry`: new cell i takes its value from old cell perm[i], and digits
are relabelled through digit_map (with 0 fixed). `apply_symmetry` applies that
pair to the puzzle, solution and moves together, so the move order is kept and
every intermediate board is transformed the same way as the puzzle.

To add an operation, write a function of that form and add it to OPERATIONS.
"""

import argparse

import numpy as np

from .data import apply_symmetry, decode, read_kinds, read_records, validate_record, write_records

_IDENTITY_CELLS = np.arange(81)
_IDENTITY_DIGITS = np.arange(10, dtype=np.uint8)


def relabel_digits(rng):
    """Swap digits for a random permutation of 1-9."""
    digit_map = _IDENTITY_DIGITS.copy()
    digit_map[1:] = rng.permutation(np.arange(1, 10, dtype=np.uint8))
    return _IDENTITY_CELLS, digit_map


def transpose(rng):
    """Reflect the grid across its main diagonal."""
    return _IDENTITY_CELLS.reshape(9, 9).T.ravel(), _IDENTITY_DIGITS


def swap_rows(rng):
    """Swap two random rows within the same band (rows 0-2, 3-5 or 6-8)."""
    a, b = int(rng.integers(3)) * 3 + rng.choice(3, size=2, replace=False)
    rows = np.arange(9)
    rows[[a, b]] = rows[[b, a]]
    return _IDENTITY_CELLS.reshape(9, 9)[rows].ravel(), _IDENTITY_DIGITS


def swap_columns(rng):
    """Swap two random columns within the same stack (columns 0-2, 3-5 or 6-8)."""
    a, b = int(rng.integers(3)) * 3 + rng.choice(3, size=2, replace=False)
    cols = np.arange(9)
    cols[[a, b]] = cols[[b, a]]
    return _IDENTITY_CELLS.reshape(9, 9)[:, cols].ravel(), _IDENTITY_DIGITS


def _swap_blocks(rng):
    """Line order that swaps two random groups of three lines (e.g. bands)."""
    blocks = np.arange(3)
    a, b = rng.choice(3, size=2, replace=False)
    blocks[[a, b]] = blocks[[b, a]]
    return (blocks[:, None] * 3 + np.arange(3)).ravel()


def swap_bands(rng):
    """Swap two random bands: the 3-row blocks of rows 0-2, 3-5 and 6-8."""
    return _IDENTITY_CELLS.reshape(9, 9)[_swap_blocks(rng)].ravel(), _IDENTITY_DIGITS


def swap_stacks(rng):
    """Swap two random stacks: the 3-column blocks of columns 0-2, 3-5 and 6-8."""
    return _IDENTITY_CELLS.reshape(9, 9)[:, _swap_blocks(rng)].ravel(), _IDENTITY_DIGITS


def rotate(rng):
    """Rotate the grid 90 degrees clockwise."""
    return np.rot90(_IDENTITY_CELLS.reshape(9, 9), k=-1).ravel(), _IDENTITY_DIGITS


def mirror_horizontal(rng):
    """Flip the grid left to right, reversing the order of the columns."""
    return _IDENTITY_CELLS.reshape(9, 9)[:, ::-1].ravel(), _IDENTITY_DIGITS


def mirror_vertical(rng):
    """Flip the grid top to bottom, reversing the order of the rows."""
    return _IDENTITY_CELLS.reshape(9, 9)[::-1].ravel(), _IDENTITY_DIGITS


OPERATIONS = [
    relabel_digits,
    transpose,
    swap_rows,
    swap_columns,
    swap_bands,
    swap_stacks,
    rotate,
    mirror_horizontal,
    mirror_vertical,
]


def augment_record(record, rng, min_ops=100, max_ops=150, operations=None):
    """Apply between `min_ops` and `max_ops` randomly chosen operations to a record.

    Each operation is a small change, so it takes about 100 of them before a
    record stops resembling its base puzzle.
    """
    if not 1 <= min_ops <= max_ops:
        raise ValueError("need 1 <= min_ops <= max_ops")
    operations = operations or OPERATIONS
    for _ in range(int(rng.integers(min_ops, max_ops + 1))):
        op = operations[rng.integers(len(operations))]
        record = apply_symmetry(*record, *op(rng))
    validate_record(*record)
    return record


def generate_with_bases(records, count, rng, min_ops=100, max_ops=150, operations=None, max_attempts=None):
    """Like `generate`, but also returns the index of each new record's base record."""
    seen = {decode(puzzle) for puzzle, _, _ in records}
    out, bases = [], []
    max_attempts = max_attempts or count * 20
    for attempt in range(max_attempts):
        if len(out) == count:
            break
        base = attempt % len(records)
        new = augment_record(records[base], rng, min_ops, max_ops, operations)
        key = decode(new[0])
        if key not in seen:
            seen.add(key)
            out.append(new)
            bases.append(base)
    if len(out) < count:
        raise RuntimeError(f"only {len(out)} distinct records after {max_attempts} attempts")
    return out, bases


def generate(records, count, rng, min_ops=100, max_ops=150, operations=None, max_attempts=None):
    """Make `count` new records, cycling through `records` as bases.

    Records whose puzzle matches an input or an earlier output are skipped, so
    every generated puzzle is distinct.
    """
    return generate_with_bases(records, count, rng, min_ops, max_ops, operations, max_attempts)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="data/manual.parquet")
    parser.add_argument("--output", default="data/generated.parquet")
    parser.add_argument("--count", type=int, required=True, help="number of records to generate")
    parser.add_argument("--min-ops", type=int, default=100, help="fewest operations applied to one record")
    parser.add_argument("--max-ops", type=int, default=150, help="most operations applied to one record")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--source", default="generated", help="label for the source column")
    args = parser.parse_args(argv)

    records = read_records(args.input)
    out, bases = generate_with_bases(records, args.count, np.random.default_rng(args.seed), args.min_ops, args.max_ops)
    # Operations keep the move order, so each record's move kinds are its base's.
    kinds = read_kinds(args.input)
    write_records(args.output, out, source=args.source, kinds=kinds and [kinds[b] for b in bases])
    print(f"wrote {len(out)} records from {len(records)} bases to {args.output}")


if __name__ == "__main__":
    main()
