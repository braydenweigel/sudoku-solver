import numpy as np
import torch
from torch.utils.data import Dataset

from .data import apply_moves, apply_symmetry, random_symmetry, read_records


class SudokuDataset(Dataset):
    """Yields (input, target, mask) for whole-solution prediction.

    input   int64[81] tokens 0-9 (0 = blank), or float32[10, 9, 9] if one_hot
    target  int64[81] classes 0-8 (solution digit - 1)
    mask    bool[81]  True where the input cell is blank (cells to score)

    With `random_step`, each access replays a random number of the puzzle's
    moves, so the model sees a different intermediate board every epoch.
    Without it the original puzzle is returned, which is what evaluation wants.
    """

    def __init__(self, path, random_step=False, augment=False, one_hot=False):
        self.records = read_records(path)
        self.random_step = random_step
        self.augment = augment
        self.one_hot = one_hot

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        puzzle, solution, moves = self.records[i]
        # Seed from torch so DataLoader workers get distinct random streams.
        rng = np.random.default_rng(int(torch.randint(0, 2**63 - 1, (1,))))
        if self.augment:
            puzzle, solution, moves = apply_symmetry(puzzle, solution, moves, *random_symmetry(rng))
        t = int(rng.integers(len(moves))) if self.random_step and len(moves) else 0
        board = torch.from_numpy(apply_moves(puzzle, moves, t).astype(np.int64))
        target = torch.from_numpy(solution.astype(np.int64) - 1)
        mask = board == 0
        if self.one_hot:
            board = torch.nn.functional.one_hot(board, 10).T.reshape(10, 9, 9).float()
        return board, target, mask
