"""Make ordered move lists for puzzles that only have a solution.

Usage:
    python -m sudoku.moves --input puzzles.csv --output data/public.parquet --source public

The input CSV has a header row and the puzzle and solution as its first two
columns, as 81-character strings ('0' or '.' for blanks).

Every move places the correct digit; wrong guesses and dead ends are never
recorded. Each move is chosen with the simplest reasoning that finds one:

1. basic: a naked single (the only digit left for a cell) or a hidden single
   (the only cell left for a digit in a row, column or box). When there are
   none, candidates are eliminated with locked candidates, naked pairs and
   triples, and hidden pairs, until a single appears.
2. trial: if that runs out, try each candidate of a cell (fewest candidates
   first) and follow its consequences with the basic reasoning. A candidate that
   leads to a contradiction is ruled out, and step 1 resumes.
3. search: if no trial gives a contradiction, place the solution's digit in
   the blank cell with the fewest candidates. Because the puzzle has exactly
   one solution, a full search would rule out every other candidate there.

--level sets the most advanced step allowed; puzzles that need more are
skipped. With --level search (the default) every puzzle with exactly one
solution gets a move list; puzzles with several solutions are skipped.

With --hard-only it keeps only the puzzles that basic reasoning can't finish,
for a harder test set:
    python -m sudoku.moves --input puzzles.csv --output data/hard_test.parquet --source hard --hard-only --count 2000
"""

import argparse
import csv
from itertools import combinations, islice

import numpy as np

from .data import MOVE_BASIC, MOVE_SEARCH, MOVE_TRIAL, decode, encode, read_records, write_records

ROWS = [list(range(r * 9, r * 9 + 9)) for r in range(9)]
COLS = [list(range(c, 81, 9)) for c in range(9)]
BOXES = [[(br + i) * 9 + bc + j for i in range(3) for j in range(3)] for br in (0, 3, 6) for bc in (0, 3, 6)]
UNITS = ROWS + COLS + BOXES
PEERS = [set().union(*(u for u in UNITS if c in u)) - {c} for c in range(81)]
LEVELS = ("basic", "trial", "search")
_DIGITS = set(range(1, 10))


class _Board:
    def __init__(self, puzzle):
        self.values = [int(v) for v in puzzle]
        self.cands = [set() if v else set(_DIGITS) for v in self.values]
        for c, v in enumerate(self.values):
            if v:
                self._remove_from_peers(c, v)

    def copy(self):
        other = _Board.__new__(_Board)
        other.values = self.values.copy()
        other.cands = [set(cs) for cs in self.cands]
        return other

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

    def contradiction(self):
        """True if the board can't be completed: a blank cell has no candidates,
        or a row, column or box has nowhere left for one of its missing digits."""
        if any(v == 0 and not cs for v, cs in zip(self.values, self.cands)):
            return True
        for unit in UNITS:
            placed = {self.values[c] for c in unit} - {0}
            possible = set().union(*(self.cands[c] for c in unit))
            if not _DIGITS - placed <= possible:
                return True
        return False

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


def _leads_to_contradiction(board):
    """Follow basic reasoning from `board` (modified in place) until it is
    solved, stuck, or contradicts itself. True only for a contradiction."""
    while True:
        if board.contradiction():
            return True
        if 0 not in board.values:
            return False
        singles = board.singles()
        if singles:
            for cell, digit in singles:
                if board.values[cell] == digit:
                    continue
                if board.values[cell] or digit not in board.cands[cell]:
                    return True  # two singles clash
                board.place(cell, digit)
        elif not board.eliminate():
            return False


def _trial_elimination(board, solution):
    """Rule out one candidate whose consequences contradict themselves. False if none does."""
    blanks = sorted((c for c in range(81) if board.values[c] == 0), key=lambda c: (len(board.cands[c]), c))
    for cell in blanks:
        # The solution's digit never leads to a contradiction, so trying it would
        # only waste time; skipping it doesn't change what this finds.
        for digit in sorted(board.cands[cell] - {int(solution[cell])}):
            trial = board.copy()
            trial.place(cell, digit)
            if _leads_to_contradiction(trial):
                board.cands[cell].discard(digit)
                return True
    return False


def make_moves(puzzle, solution, rng=None, level="search"):
    """Ordered (cell, digit) moves solving `puzzle`, and how each move was found.

    Returns (moves, kinds): moves is uint8[N, 2], or None if reasoning up to
    `level` runs out; kinds is uint8[N] with MOVE_BASIC for moves found by basic
    reasoning alone, MOVE_TRIAL for moves found only after at least one trial
    elimination since the previous move, and MOVE_SEARCH for placements
    justified by search. Among the singles available at a step, one is chosen
    at random with `rng` (or the lowest cell if None). Raises ValueError if a
    deduced digit disagrees with `solution`.

    The search step relies on the puzzle having exactly one solution; check
    with `count_solutions` before trusting its moves.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    values = [int(v) for v in puzzle]
    moves, kinds = [], []
    while 0 in values:
        # Start each move from the digits alone, not candidates ruled out for
        # earlier moves: the model only sees the board, so a move's kind should
        # say what reasoning the board itself needs.
        board = _Board(values)
        kind = MOVE_BASIC
        singles = board.singles()
        while not singles:
            if board.eliminate():
                pass
            elif level != "basic" and _trial_elimination(board, solution):
                kind = MOVE_TRIAL
            else:
                break
            singles = board.singles()
        if singles:
            cell, digit = singles[int(rng.integers(len(singles)))] if rng is not None else singles[0]
            if digit != solution[cell]:
                raise ValueError(f"deduced {digit} at cell {cell}, but the solution has {solution[cell]}")
        elif level == "search":
            blanks = [c for c in range(81) if values[c] == 0]
            cell = min(blanks, key=lambda c: len(board.cands[c]))
            digit = int(solution[cell])
            kind = MOVE_SEARCH
        else:
            return None, np.array(kinds, dtype=np.uint8)
        values[cell] = digit
        moves.append((cell, digit))
        kinds.append(kind)
    return np.array(moves, dtype=np.uint8).reshape(-1, 2), np.array(kinds, dtype=np.uint8)


def logical_moves(puzzle, solution, rng=None, level="basic"):
    """Just the moves from `make_moves`, or None if reasoning up to `level` runs out."""
    return make_moves(puzzle, solution, rng, level)[0]


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
    parser.add_argument("--level", choices=LEVELS, default="search", help="most advanced step allowed")
    parser.add_argument(
        "--hard-only", action="store_true", help="keep only puzzles that basic reasoning can't finish"
    )
    parser.add_argument(
        "--exclude", nargs="+", default=[], help="record files whose puzzles to skip, e.g. test sets"
    )
    parser.add_argument("--limit", type=int, default=None, help="read only the first N rows")
    parser.add_argument("--count", type=int, default=None, help="stop after writing N records")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    excluded = {decode(p) for path in args.exclude for p, _, _ in read_records(path)}
    rng = np.random.default_rng(args.seed)
    records, all_kinds = [], []
    skipped = {"excluded": 0, "too hard for --level": 0, "several solutions": 0, "not hard": 0}
    used = {"trial": 0, "search": 0}
    with open(args.input, newline="") as f:
        rows = csv.reader(f)
        next(rows)  # header
        for p, s, *_ in islice(rows, args.limit):
            if args.count is not None and len(records) >= args.count:
                break
            puzzle, solution = encode(p.strip()), encode(s.strip())
            if decode(puzzle) in excluded:
                skipped["excluded"] += 1
                continue
            moves, kinds = make_moves(puzzle, solution, rng, args.level)
            trial, search = (kinds == MOVE_TRIAL).any(), (kinds == MOVE_SEARCH).any()
            if moves is None:
                skipped["too hard for --level"] += 1
            elif search and count_solutions(puzzle) != 1:
                skipped["several solutions"] += 1
            elif args.hard_only and not (trial or search):
                skipped["not hard"] += 1
            else:
                records.append((puzzle, solution, moves))
                all_kinds.append(kinds)
                used["trial"] += bool(trial)
                used["search"] += bool(search)
    write_records(args.output, records, source=args.source, kinds=all_kinds)
    print(f"wrote {len(records)} records to {args.output}")
    print(f"  needed trial eliminations: {used['trial']}, needed search placements: {used['search']}")
    reasons = ", ".join(f"{n} {why}" for why, n in skipped.items() if n)
    print(f"  skipped: {reasons or 'none'}")


if __name__ == "__main__":
    main()
