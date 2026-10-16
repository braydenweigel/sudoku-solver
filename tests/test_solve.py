import numpy as np
import torch

from sudoku.data import encode
from sudoku.evaluate import score
from sudoku.solve import fill_all, fill_one_at_a_time

from test_data import PUZZLE, SOLUTION


class Oracle(torch.nn.Module):
    """Predicts the true solution, most confident on the lowest blank cell index."""

    def __init__(self, solution):
        super().__init__()
        self.solution = solution
        self.calls = []

    def forward(self, board):
        self.calls.append(board.clone())
        logits = torch.nn.functional.one_hot(self.solution - 1, 9).float().expand(len(board), 81, 9).clone()
        return logits * (100 - torch.arange(81)).float()[None, :, None] / 10


def boards():
    puzzle = torch.from_numpy(encode(PUZZLE).astype(np.int64))
    solution = torch.from_numpy(encode(SOLUTION).astype(np.int64))
    return puzzle[None], solution


def test_fill_all_solves_with_one_run():
    board, solution = boards()
    model = Oracle(solution)
    assert torch.equal(fill_all(model, board)[0], solution)
    assert len(model.calls) == 1


def test_fill_one_at_a_time_fills_most_confident_cell_each_run():
    board, solution = boards()
    model = Oracle(solution)
    out = fill_one_at_a_time(model, board)
    assert torch.equal(out[0], solution)
    blanks = np.flatnonzero(encode(PUZZLE) == 0)
    assert len(model.calls) == len(blanks)
    for step, seen in enumerate(model.calls):
        filled = (seen[0] != 0).sum().item()
        assert filled == 81 - len(blanks) + step  # exactly one new cell per run
        if step:
            assert seen[0, blanks[step - 1]] != 0  # lowest index first, as the oracle prefers it


def test_fill_one_at_a_time_never_changes_givens():
    board, _ = boards()
    wrong = Oracle(torch.ones(81, dtype=torch.long))  # always predicts 1
    out = fill_one_at_a_time(wrong, board)[0]
    given = board[0] != 0
    assert torch.equal(out[given], board[0][given])
    assert (out[~given] == 1).all()


def test_fill_one_at_a_time_handles_boards_finishing_at_different_times():
    board, solution = boards()
    nearly_done = solution.clone()
    nearly_done[:3] = 0
    batch = torch.stack([board[0], nearly_done])
    out = fill_one_at_a_time(Oracle(solution), batch)
    assert torch.equal(out, solution.expand(2, 81))


def test_score_counts_cells_and_puzzles():
    board, solution = boards()
    mask = board == 0
    loader = [(board, solution[None] - 1, mask)]
    result = score(fill_all, Oracle(solution), loader, torch.device("cpu"))
    assert result == {"cell_acc": 1.0, "puzzle_acc": 1.0}
