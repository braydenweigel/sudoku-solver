"""Sudoku record format: encoding, move replay, validation, augmentation, Parquet I/O.

A record is (puzzle, solution, moves):
    puzzle    uint8[81]    row-major, 0 = blank, 1-9 = given
    solution  uint8[81]    fully solved grid
    moves     uint8[N, 2]  ordered (cell_index, digit), one per blank cell
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_BOXES = np.arange(81).reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
_DIGITS = np.arange(1, 10)


def encode(grid: str) -> np.ndarray:
    """81-char string -> uint8[81]. '0' or '.' is a blank."""
    if len(grid) != 81:
        raise ValueError(f"expected 81 characters, got {len(grid)}")
    arr = np.frombuffer(grid.replace(".", "0").encode("ascii"), dtype=np.uint8) - ord("0")
    if arr.max() > 9:
        raise ValueError("grid may only contain digits 0-9 or '.'")
    return arr


def decode(grid: np.ndarray) -> str:
    """uint8[81] -> 81-char string with '0' for blanks."""
    return "".join(map(str, np.asarray(grid).tolist()))


def apply_moves(puzzle: np.ndarray, moves: np.ndarray, t: int) -> np.ndarray:
    """Board after the first `t` moves."""
    board = puzzle.copy()
    board[moves[:t, 0]] = moves[:t, 1]
    return board


def is_solved(grid: np.ndarray) -> bool:
    rows = grid.reshape(9, 9)
    units = np.concatenate([rows, rows.T, grid[_BOXES]])
    return bool((np.sort(units, axis=1) == _DIGITS).all())


def validate_record(puzzle: np.ndarray, solution: np.ndarray, moves: np.ndarray) -> None:
    """Raise ValueError if the record breaks a format invariant."""
    if puzzle.shape != (81,) or solution.shape != (81,):
        raise ValueError("puzzle and solution must have shape (81,)")
    if moves.ndim != 2 or moves.shape[1] != 2:
        raise ValueError("moves must have shape (N, 2)")
    if not is_solved(solution):
        raise ValueError("solution is not a valid solved grid")
    given = puzzle != 0
    if not (puzzle[given] == solution[given]).all():
        raise ValueError("puzzle givens disagree with solution")
    cells = moves[:, 0]
    if cells.size and cells.max() > 80:
        raise ValueError("move cell index out of range")
    if not np.array_equal(np.sort(cells), np.flatnonzero(~given)):
        raise ValueError("moves must fill every blank cell exactly once")
    if not (moves[:, 1] == solution[cells]).all():
        raise ValueError("move digit disagrees with solution")


def random_symmetry(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random validity-preserving transform as (perm, digit_map).

    `perm` is int[81]: new cell i takes its value from old cell perm[i]. It
    combines band swaps, row swaps within a band, the same for columns, and an
    optional transpose, which together cover all rotations and reflections.
    `digit_map` is uint8[10] relabelling the digits, with 0 fixed.
    """

    def line_order():
        return np.concatenate([band * 3 + rng.permutation(3) for band in rng.permutation(3)])

    idx = np.arange(81).reshape(9, 9)[line_order()][:, line_order()]
    if rng.integers(2):
        idx = idx.T
    digit_map = np.zeros(10, dtype=np.uint8)
    digit_map[1:] = rng.permutation(_DIGITS)
    return idx.ravel(), digit_map


def apply_symmetry(puzzle, solution, moves, perm, digit_map):
    """Apply a transform from `random_symmetry` to a whole record."""
    inverse = np.empty(81, dtype=np.uint8)
    inverse[perm] = np.arange(81, dtype=np.uint8)
    new_moves = np.stack([inverse[moves[:, 0]], digit_map[moves[:, 1]]], axis=1)
    return digit_map[puzzle[perm]], digit_map[solution[perm]], new_moves


def write_records(path, records, source: str) -> None:
    """Write an iterable of (puzzle, solution, moves) records to a Parquet file.

    `source` labels how the records were produced (e.g. "manual") and is stored
    in a column of the same name.
    """
    puzzles, solutions, moves_col = [], [], []
    for puzzle, solution, moves in records:
        validate_record(puzzle, solution, moves)
        puzzles.append(decode(puzzle))
        solutions.append(decode(solution))
        moves_col.append(moves.tolist())
    table = pa.table(
        {
            "puzzle": pa.array(puzzles, pa.string()),
            "solution": pa.array(solutions, pa.string()),
            "moves": pa.array(moves_col, pa.list_(pa.list_(pa.uint8(), 2))),
            "source": pa.array([source] * len(puzzles), pa.string()),
        }
    )
    pq.write_table(table, path)


def read_records(path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read a Parquet file written by `write_records`."""
    table = pq.read_table(path).to_pydict()
    return [
        (encode(p), encode(s), np.array(m, dtype=np.uint8).reshape(-1, 2))
        for p, s, m in zip(table["puzzle"], table["solution"], table["moves"])
    ]
