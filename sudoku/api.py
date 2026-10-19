"""Solve sudoku puzzles with a trained model.

    from sudoku.api import SudokuSolver

    solver = SudokuSolver.from_checkpoint("checkpoints/model.pt")
    solver.solve("530070000600195000098000060800060003400803001700020006060000280000419005000080079")
    # -> "534678912672195348198342567859761423426853791713924856961537284287419635345286179"

Puzzles can be an 81-character string ('0' or '.' for blanks; spaces, newlines
and the separators | - + are ignored) or a 9x9 list of ints (0 for blanks).
Results come back in the same form. Every result is checked against the rules,
so a wrong answer from the model raises SolveError instead of being returned.

From the command line:
    python -m sudoku.api 530070000600195000098000060800060003400803001700020006060000280000419005000080079
"""

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from .evaluate import conflicts, fill_all_with_retries
from .solve import fill_all, fill_one_at_a_time
from .train import autocast, load_model

METHODS = {
    "one_at_a_time": fill_one_at_a_time,  # most accurate
    "retries": fill_all_with_retries,  # about 10x faster, a little less accurate
    "all_at_once": fill_all,  # fastest, least accurate
}
_IGNORED = set(" \t\r\n|-+")


class SolveError(ValueError):
    """The model's grid breaks the rules. `grid` holds it, in the puzzle's form."""

    def __init__(self, message, grid):
        super().__init__(message)
        self.grid = grid


@dataclass
class Move:
    """A suggested next move: put `digit` at `row`, `col` (both 0-8)."""

    row: int
    col: int
    digit: int
    confidence: float


def _cells(puzzle):
    """An 81-char string or 9x9 list -> int64[81] array, checking only its shape."""
    if isinstance(puzzle, str):
        chars = [c for c in puzzle if c not in _IGNORED]
        if len(chars) != 81 or any(c not in "0123456789." for c in chars):
            raise ValueError("a puzzle string needs exactly 81 digits or '.' (spaces and | - + are ignored)")
        cells = [0 if c == "." else int(c) for c in chars]
    else:
        rows = [list(r) for r in puzzle]
        if len(rows) != 9 or any(len(r) != 9 for r in rows):
            raise ValueError("a puzzle grid needs 9 rows of 9 cells")
        cells = [int(v) for r in rows for v in r]
        if any(not 0 <= v <= 9 for v in cells):
            raise ValueError("grid cells must be 0 (blank) to 9")
    return np.array(cells, dtype=np.int64)


def _breaks_rules(board):
    return bool(conflicts(torch.from_numpy(board)[None]).any())


def parse(puzzle):
    """An 81-char string or 9x9 list -> int64[81] array, checking it is a legal puzzle."""
    board = _cells(puzzle)
    if _breaks_rules(board):
        raise ValueError("the puzzle repeats a digit in a row, column or box")
    return board


def _like(puzzle, board):
    """Return `board` in the same form as `puzzle` was given."""
    if isinstance(puzzle, str):
        return "".join(map(str, board.tolist()))
    return [board[r * 9 : r * 9 + 9].tolist() for r in range(9)]


def is_solution(puzzle, grid):
    """True if `grid` is completely filled, keeps every given, and breaks no rules."""
    p, g = parse(puzzle), _cells(grid)
    return bool((g != 0).all() and (g[p != 0] == p[p != 0]).all() and not _breaks_rules(g))


class SudokuSolver:
    def __init__(self, model, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()

    @classmethod
    def from_checkpoint(cls, path="checkpoints/model.pt", device=None):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return cls(load_model(path, device), device)

    def solve(self, puzzle, method="one_at_a_time"):
        """The finished grid, in the same form as `puzzle`. Raises SolveError if the model fails."""
        grid = self.solve_many([puzzle], method=method, strict=False)[0]
        if _breaks_rules(_cells(grid)):
            raise SolveError("the model's solution breaks the rules", grid)
        return grid

    def solve_many(self, puzzles, method="one_at_a_time", batch_size=1024, strict=True):
        """Solve a list of puzzles together, which is much faster than one by one.

        Each result is in the same form as its puzzle. With `strict`, a puzzle
        the model gets wrong gives None instead of a grid that breaks the rules.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}")
        boards = [parse(p) for p in puzzles]
        results = []
        for i in range(0, len(boards), batch_size):
            batch = torch.from_numpy(np.stack(boards[i : i + batch_size])).to(self.device)
            filled = METHODS[method](self.model, batch)
            bad = conflicts(filled).any(1).cpu().numpy()
            for j, board in enumerate(filled.cpu().numpy()):
                results.append(None if strict and bad[j] else _like(puzzles[i + j], board))
        return results

    @torch.no_grad()
    def next_move(self, puzzle):
        """The single move the model is most sure of, e.g. for a hint. None if the puzzle is full."""
        board = torch.from_numpy(parse(puzzle)).to(self.device)[None]
        if not (board == 0).any():
            return None
        with autocast(self.device):
            confidence, digit = self.model(board).float().softmax(-1).max(-1)
        confidence = confidence.masked_fill(board != 0, -1)[0]
        cell = int(confidence.argmax())
        return Move(cell // 9, cell % 9, int(digit[0, cell]) + 1, float(confidence[cell]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Solve a sudoku puzzle with a trained model.")
    parser.add_argument("puzzle", help="81 characters, '0' or '.' for blanks")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--method", choices=sorted(METHODS), default="one_at_a_time")
    args = parser.parse_args(argv)

    solver = SudokuSolver.from_checkpoint(args.checkpoint)
    try:
        grid = solver.solve(args.puzzle, method=args.method)
    except SolveError as e:
        print(f"{e}; its attempt:")
        grid = e.grid
    for r in range(9):
        row = grid[r * 9 : r * 9 + 9]
        print(" ".join(row[0:3]), "|", " ".join(row[3:6]), "|", " ".join(row[6:9]))
        if r in (2, 5):
            print("------+-------+------")


if __name__ == "__main__":
    main()
