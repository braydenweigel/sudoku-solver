import numpy as np
import torch

from sudoku.data import encode
from sudoku.evaluate import conflicts, fill_all_with_retries

from test_data import PUZZLE, SOLUTION


def boards():
    puzzle = torch.from_numpy(encode(PUZZLE).astype(np.int64))
    solution = torch.from_numpy(encode(SOLUTION).astype(np.int64))
    return puzzle[None], solution


def test_conflicts_finds_repeated_digits_only():
    board, solution = boards()
    assert not conflicts(solution[None]).any()
    assert not conflicts(board).any()  # blanks never count
    wrong = solution.clone()[None]
    wrong[0, 0] = wrong[0, 1]  # row 0 now has two of the same digit
    bad = conflicts(wrong)[0]
    assert bad[0] and bad[1]
    # Also flagged: any cell in cell 0's column or box that already held that digit.
    col0 = [r * 9 for r in range(9)]
    box0 = [0, 1, 2, 9, 10, 11, 18, 19, 20]
    expected = {0, 1} | {c for c in col0 + box0 if c != 0 and wrong[0, c] == wrong[0, 0]}
    assert set(torch.nonzero(bad).flatten().tolist()) == expected


class Model(torch.nn.Module):
    """Predicts the solution confidently, except that while fewer than `clues`
    cells are filled it puts a confident wrong digit in `cell`, and is unsure there."""

    def __init__(self, solution, cell=None, wrong=None, clues=0):
        super().__init__()
        self.solution, self.cell, self.wrong, self.clues = solution, cell, wrong, clues
        self.calls = 0

    def forward(self, board):
        self.calls += 1
        logits = 20 * torch.nn.functional.one_hot(self.solution - 1, 9).float().expand(len(board), 81, 9).clone()
        if self.cell is not None:
            early = (board != 0).sum(1) < self.clues
            logits[early, self.cell] = 0
            logits[early, self.cell, self.wrong - 1] = 1  # wrong, and only about 25% sure
        return logits


def test_valid_first_try_runs_once():
    board, solution = boards()
    model = Model(solution)
    assert torch.equal(fill_all_with_retries(model, board)[0], solution)
    assert model.calls == 1


def test_retry_recovers_from_a_wrong_cell():
    board, solution = boards()
    # Cell 2's answer is 4; a 5 there repeats the 5 in cell 0.
    model = Model(solution, cell=2, wrong=5, clues=40)
    out = fill_all_with_retries(model, board)[0]
    assert torch.equal(out, solution)
    assert model.calls == 2  # the second try had the confident cells as extra clues


def test_gives_up_after_tries_and_keeps_givens():
    board, solution = boards()
    model = Model(solution, cell=2, wrong=5, clues=99)  # never gets it right
    out = fill_all_with_retries(model, board, tries=3)[0]
    assert model.calls == 3
    assert out[2] == 5 and (out != 0).all()
    given = board[0] != 0
    assert torch.equal(out[given], board[0][given])


def test_batch_mixes_solved_and_retried_puzzles():
    board, solution = boards()
    nearly_done = solution.clone()
    nearly_done[2] = 0  # 80 clues: the model is never fooled on this one
    batch = torch.stack([board[0], nearly_done])
    model = Model(solution, cell=2, wrong=5, clues=40)
    assert torch.equal(fill_all_with_retries(model, batch), solution.expand(2, 81))
