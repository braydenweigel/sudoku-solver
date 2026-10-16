"""Make logically ordered move lists for puzzles that only have a solution.

Usage:
    python -m sudoku.moves --input puzzles.csv --output data/public.parquet --source public

The input CSV has a header row and the puzzle and solution as its first two
columns, as 81-character strings ('0' or '.' for blanks). Rows whose puzzle
can't be solved with the techniques below are skipped and counted.

Each move is a naked single (the only digit left for a cell) or a hidden single
(the only cell left for a digit in a row, column or box), judged on the
candidates at that point. When there are none, candidates are eliminated with
locked candidates, naked pairs and triples, and hidden pairs, until a single
appears. Nothing is guessed, so every move is deducible from the board before it.

With --hard-only it does the opposite and keeps only the puzzles those
techniques can't finish, for a harder test set (see `guided_moves`):
    python -m sudoku.moves --input puzzles.csv --output data/hard_test.parquet --source hard --hard-only --count 2000
"""

import argparse
import csv
from itertools import combinations, islice

import numpy as np

from .data import encode, write_records

ROWS = [list(range(r * 9, r * 9 + 9)) for r in range(9)]
COLS = [list(range(c, 81, 9)) for c in range(9)]
BOXES = [[(br + i) * 9 + bc + j for i in range(3) for j in range(3)] for br in (0, 3, 6) for bc in (0, 3, 6)]
UNITS = ROWS + COLS + BOXES
PEERS = [set().union(*(u for u in UNITS if c in u)) - {c} for c in range(81)]


class _Board:
    def __init__(self, puzzle):
        self.values = [int(v) for v in puzzle]
        self.cands = [set() if v else set(range(1, 10)) for v in self.values]
        for c, v in enumerate(self.values):
            if v:
                self._remove_from_peers(c, v)

    def _remove_from_peers(self, cell, digit):
        for p in PEERS[cell]:
            self.cands[p].discard(digit)

    def place(self, cell, digit):
        self.values[cell] = digit
        self.cands[cell] = set()
        self._remove_from_peers(cell, digit)

    def places(self, unit, digit):
        return [c for c in unit if digit in self.cands[c]]

    def singles(self):
        found = {(c, next(iter(cs))) for c, cs in enumerate(self.cands) if len(cs) == 1}
        for unit in UNITS:
            for d in range(1, 10):
                cells = self.places(unit, d)
                if len(cells) == 1:
                    found.add((cells[0], d))
        return sorted(found)

    def remove(self, cells, digits):
        changed = False
        for c in cells:
            if self.cands[c] & digits:
                self.cands[c] -= digits
                changed = True
        return changed

    def eliminate(self):
        """Apply the simplest technique that removes a candidate. False if none does."""
        return (
            self._locked_candidates()
            or self._naked_subsets(2)
            or self._hidden_pairs()
            or self._naked_subsets(3)
        )

    def _locked_candidates(self):
        # A digit confined to one line within a box (or one box within a line)
        # can't appear elsewhere in that line (or box).
        for a in UNITS:
            for d in range(1, 10):
                cells = set(self.places(a, d))
                if len(cells) < 2:
                    continue
                for b in UNITS:
                    if b is not a and cells <= set(b):
                        if self.remove(set(b) - cells, {d}):
                            return True
        return False

    def _naked_subsets(self, n):
        # n cells in a unit sharing only n candidates own those digits.
        for unit in UNITS:
            open_cells = [c for c in unit if 2 <= len(self.cands[c]) <= n]
            for group in combinations(open_cells, n):
                digits = set().union(*(self.cands[c] for c in group))
                if len(digits) == n and self.remove(set(unit) - set(group), digits):
                    return True
        return False

    def _hidden_pairs(self):
        # Two digits that fit only the same two cells of a unit fill those cells.
        for unit in UNITS:
            where = {d: self.places(unit, d) for d in range(1, 10)}
            pairs = [d for d, cells in where.items() if len(cells) == 2]
            for d1, d2 in combinations(pairs, 2):
                if where[d1] == where[d2]:
                    for c in where[d1]:
                        if self.cands[c] - {d1, d2}:
                            self.cands[c] &= {d1, d2}
                            return True
        return False


def _solve(puzzle, solution, rng, guess):
    board = _Board(puzzle)
    moves, guesses = [], 0
    while 0 in board.values:
        singles = board.singles()
        if singles:
            cell, digit = singles[int(rng.integers(len(singles)))] if rng is not None else singles[0]
            if digit != solution[cell]:
                raise ValueError(f"deduced {digit} at cell {cell}, but the solution has {solution[cell]}")
        elif board.eliminate():
            continue
        elif guess:
            blanks = [c for c in range(81) if board.values[c] == 0]
            cell = min(blanks, key=lambda c: len(board.cands[c]))
            digit = int(solution[cell])
            guesses += 1
        else:
            return None, guesses
        board.place(cell, digit)
        moves.append((cell, digit))
    return np.array(moves, dtype=np.uint8).reshape(-1, 2), guesses


def logical_moves(puzzle, solution, rng=None):
    """Ordered (cell, digit) moves solving `puzzle`, or None if the techniques run out.

    Among the singles available at each step, one is chosen at random with `rng`
    (or the lowest cell if `rng` is None). Raises ValueError if a deduced digit
    disagrees with `solution`, which means the solution is wrong.
    """
    return _solve(puzzle, solution, rng, guess=False)[0]


def guided_moves(puzzle, solution, rng=None):
    """Like `logical_moves`, but never gives up; returns (moves, guesses).

    When the techniques run out, it places the solution's digit in the blank cell
    with the fewest candidates (a guided guess) and carries on. The guess count
    is a rough measure of how far a puzzle is beyond these techniques. Moves
    after a guess aren't deducible from the board alone, so these move lists
    suit evaluation, not training with random intermediate boards.
    """
    return _solve(puzzle, solution, rng, guess=True)


def count_solutions(puzzle, limit=2):
    """Number of solutions of `puzzle`, counting no further than `limit`."""
    values = [int(v) for v in puzzle]
    used = [0] * 27  # bitmask of digits in each row, column and box
    units_of = [(c // 9, 9 + c % 9, 18 + (c // 27) * 3 + (c % 9) // 3) for c in range(81)]
    for c, v in enumerate(values):
        if v:
            for u in units_of[c]:
                used[u] |= 1 << v

    def search():
        best, best_mask, best_n = None, 0, 10
        for c in range(81):
            if values[c] == 0:
                r, k, b = units_of[c]
                mask = ~(used[r] | used[k] | used[b]) & 0x3FE
                n = bin(mask).count("1")
                if n < best_n:
                    best, best_mask, best_n = c, mask, n
                    if n == 0:
                        return 0
        if best is None:
            return 1
        found = 0
        for d in range(1, 10):
            if best_mask & (1 << d):
                values[best] = d
                for u in units_of[best]:
                    used[u] |= 1 << d
                found += search()
                for u in units_of[best]:
                    used[u] &= ~(1 << d)
                values[best] = 0
                if found >= limit:
                    break
        return found

    return search()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="CSV with puzzle and solution as its first two columns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True, help="label for the source column, e.g. the dataset's name")
    parser.add_argument("--limit", type=int, default=None, help="read only the first N rows")
    parser.add_argument(
        "--hard-only",
        action="store_true",
        help="keep only puzzles beyond these techniques that have exactly one solution, "
        "with guided guesses in their moves; for test sets",
    )
    parser.add_argument("--count", type=int, default=None, help="stop after writing N records")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    records, skipped, guesses = [], 0, []
    with open(args.input, newline="") as f:
        rows = csv.reader(f)
        next(rows)  # header
        for p, s, *_ in islice(rows, args.limit):
            if args.count is not None and len(records) >= args.count:
                break
            puzzle, solution = encode(p.strip()), encode(s.strip())
            moves = logical_moves(puzzle, solution, rng)
            if args.hard_only:
                if moves is not None or count_solutions(puzzle) != 1:
                    skipped += 1
                    continue
                moves, n = guided_moves(puzzle, solution, rng)
                guesses.append(n)
            if moves is None:
                skipped += 1
            else:
                records.append((puzzle, solution, moves))
    write_records(args.output, records, source=args.source)
    reason = "were easy enough or had several solutions" if args.hard_only else "needed harder techniques"
    print(f"wrote {len(records)} records to {args.output}; skipped {skipped} that {reason}")
    if guesses:
        g = sorted(guesses)
        print(f"guided guesses per puzzle: median {g[len(g) // 2]}, range {g[0]}-{g[-1]}")


if __name__ == "__main__":
    main()
