"""Ways to turn model predictions into a filled board.

Both functions take int64[B, 81] boards (0 = blank) and return int64[B, 81]
boards with every cell filled; givens are never changed.
"""

import torch

from .train import autocast


@torch.no_grad()
def fill_all(model, board):
    """Fill every blank cell from a single run of the model."""
    with autocast(board.device):
        logits = model(board)
    return torch.where(board > 0, board, logits.argmax(-1) + 1)


@torch.no_grad()
def fill_one_at_a_time(model, board):
    """Repeatedly fill only the blank cell the model is most confident about.

    Each run sees the cells filled so far, so later choices can build on
    earlier ones. Takes one model run per blank cell.
    """
    board = board.clone()
    rows = torch.arange(len(board), device=board.device)
    while True:
        blank = board == 0
        unfinished = blank.any(1)
        if not unfinished.any():
            return board
        with autocast(board.device):
            logits = model(board)
        confidence, digit = logits.float().softmax(-1).max(-1)
        cell = confidence.masked_fill(~blank, -1).argmax(1)
        r, c = rows[unfinished], cell[unfinished]
        board[r, c] = digit[r, c] + 1
