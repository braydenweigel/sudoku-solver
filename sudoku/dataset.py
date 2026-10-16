import numpy as np
import torch
from torch.utils.data import Dataset

from .data import MOVE_BASIC, apply_moves, apply_symmetry, random_symmetry, read_kinds, read_records


class SudokuDataset(Dataset):
    """Yields (input, target, mask) for whole-solution prediction.

    input   int64[81] tokens 0-9 (0 = blank), or float32[10, 9, 9] if one_hot
    target  int64[81] classes 0-8 (solution digit - 1)
    mask    bool[81]  True where the input cell is blank (cells to score)

    With `random_step`, each access replays a random number of the puzzle's
    moves, so the model sees a different intermediate board every epoch.
    Without it the original puzzle is returned, which is what evaluation wants.

    `hard_fraction` (needs `random_step` and a file with move kinds) is the
    chance that, for a puzzle with any moves that needed more than basic
    reasoning, the board is instead the one just before such a move. Those
    boards are the few decisive points of a hard puzzle, which uniform sampling
    almost never picks.
    """

    def __init__(self, path, random_step=False, augment=False, one_hot=False, hard_fraction=0.0):
        self.records = read_records(path)
        kinds = read_kinds(path) if hard_fraction else None
        if hard_fraction and kinds is None:
            raise ValueError(f"hard_fraction needs move kinds, but {path} has none; regenerate it with sudoku.moves")
        self.hard_steps = [np.flatnonzero(k != MOVE_BASIC) for k in kinds] if kinds else None
        self.random_step = random_step
        self.augment = augment
        self.one_hot = one_hot
        self.hard_fraction = hard_fraction

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        puzzle, solution, moves = self.records[i]
        # Seed from torch so DataLoader workers get distinct random streams.
        rng = np.random.default_rng(int(torch.randint(0, 2**63 - 1, (1,))))
        if self.augment:
            puzzle, solution, moves = apply_symmetry(puzzle, solution, moves, *random_symmetry(rng))
        t = 0
        if self.random_step and len(moves):
            hard = self.hard_steps[i] if self.hard_steps is not None else ()
            if len(hard) and rng.random() < self.hard_fraction:
                t = int(rng.choice(hard))
            else:
                t = int(rng.integers(len(moves)))
        board = torch.from_numpy(apply_moves(puzzle, moves, t).astype(np.int64))
        target = torch.from_numpy(solution.astype(np.int64) - 1)
        mask = board == 0
        if self.one_hot:
            board = torch.nn.functional.one_hot(board, 10).T.reshape(10, 9, 9).float()
        return board, target, mask
