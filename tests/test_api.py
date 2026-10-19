import pytest
import torch

from sudoku.api import Move, SolveError, SudokuSolver, is_solution, main, parse
from sudoku.data import encode

from test_data import PUZZLE, SOLUTION


class Oracle(torch.nn.Module):
    """Predicts `answer` for every cell, most confident on the lowest cell index."""

    def __init__(self, answer):
        super().__init__()
        self.answer = torch.from_numpy(encode(answer).astype("int64"))

    def forward(self, board):
        logits = torch.nn.functional.one_hot(self.answer - 1, 9).float().expand(len(board), 81, 9).clone()
        return logits * (100 - torch.arange(81)).float()[None, :, None] / 10


def rows(grid):
    return [[int(c) for c in grid[r * 9 : r * 9 + 9]] for r in range(9)]


@pytest.fixture
def solver():
    return SudokuSolver(Oracle(SOLUTION), device="cpu")


def test_parse_accepts_strings_with_separators_and_grids():
    spaced = "\n".join(" ".join(PUZZLE[r * 9 : r * 9 + 9]) for r in range(9))
    dotted = PUZZLE.replace("0", ".")
    for form in (PUZZLE, spaced, dotted, "|".join(dotted), rows(PUZZLE)):
        assert parse(form).tolist() == encode(PUZZLE).tolist()


@pytest.mark.parametrize(
    "bad",
    [PUZZLE[:-1], PUZZLE[:-1] + "x", rows(PUZZLE)[:8], [r[:8] for r in rows(PUZZLE)], [[10] * 9] * 9],
)
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse(bad)


def test_parse_rejects_repeated_givens():
    with pytest.raises(ValueError, match="repeats"):
        parse("55" + PUZZLE[2:])


def test_solve_returns_same_form(solver):
    assert solver.solve(PUZZLE) == SOLUTION
    assert solver.solve(rows(PUZZLE)) == rows(SOLUTION)
    assert solver.solve(PUZZLE.replace("0", "."), method="all_at_once") == SOLUTION


def test_solve_raises_when_model_breaks_rules():
    always_one = SudokuSolver(Oracle("1" * 81), device="cpu")
    with pytest.raises(SolveError) as info:
        always_one.solve(PUZZLE)
    grid = info.value.grid
    assert len(grid) == 81 and "0" not in grid
    assert all(grid[i] == PUZZLE[i] for i in range(81) if PUZZLE[i] != "0")  # givens kept


def test_solve_many_marks_failures(solver):
    other = SudokuSolver(Oracle("1" * 81), device="cpu")
    assert solver.solve_many([PUZZLE, rows(PUZZLE)], batch_size=1) == [SOLUTION, rows(SOLUTION)]
    assert other.solve_many([PUZZLE]) == [None]
    assert other.solve_many([PUZZLE], strict=False)[0] != SOLUTION


def test_unknown_method(solver):
    with pytest.raises(ValueError):
        solver.solve(PUZZLE, method="guess")


def test_next_move(solver):
    move = solver.next_move(PUZZLE)
    # The oracle is most confident on the lowest blank cell, which is cell 2 (row 0, col 2).
    assert move == Move(0, 2, int(SOLUTION[2]), pytest.approx(1.0, abs=0.01))
    assert solver.next_move(SOLUTION) is None


def test_is_solution():
    assert is_solution(PUZZLE, SOLUTION)
    assert not is_solution(PUZZLE, PUZZLE)  # not filled
    changed = SOLUTION.replace("5", "x").replace("3", "5").replace("x", "3")  # swap two digits everywhere
    assert not is_solution(PUZZLE, changed)  # valid grid, but disagrees with the givens
    broken = SOLUTION[:2] + SOLUTION[1] + SOLUTION[3:]  # cell 2 (blank in PUZZLE) copies cell 1
    assert not is_solution(PUZZLE, broken)  # breaks the rules: returns False rather than raising


def test_main_prints_grid(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(SudokuSolver, "from_checkpoint", classmethod(lambda cls, path: cls(Oracle(SOLUTION), "cpu")))
    main([PUZZLE])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "5 3 4 | 6 7 8 | 9 1 2" and len(out) == 11
