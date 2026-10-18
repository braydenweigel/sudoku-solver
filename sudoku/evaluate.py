"""Compare ways of solving with a trained model.

Usage:
    python -m sudoku.evaluate --checkpoint checkpoints/model.pt --data data/generated_test.parquet

Every method starts from the original puzzles in each data file and must fill
every blank cell. Reports the share of blank cells filled correctly and the
share of puzzles solved completely.
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader, Subset

from .dataset import SudokuDataset
from .moves import UNITS
from .solve import fill_all, fill_one_at_a_time
from .train import autocast, load_model

# _UNIT_CELLS[u, c] is 1 if cell c belongs to unit u (a row, column or box).
_UNIT_CELLS = torch.zeros(27, 81)
for _u, _cells in enumerate(UNITS):
    _UNIT_CELLS[_u, _cells] = 1


def conflicts(board):
    """bool[B, 81]: True where a filled cell's digit is repeated in its row, column or box."""
    unit_cells = _UNIT_CELLS.to(board.device)
    digits = torch.nn.functional.one_hot(board, 10)[..., 1:].float()  # [B, 81, 9]
    counts = torch.einsum("uc,bcd->bud", unit_cells, digits)  # [B, 27, 9]
    repeated = torch.einsum("uc,bud->bcd", unit_cells, (counts > 1).float()) > 0  # [B, 81, 9]
    return (repeated & (digits > 0)).any(-1)


@torch.no_grad()
def fill_all_with_retries(model, board, tries=5, keep_above=0.9):
    """Solve all at once, and try again if the result breaks the rules.

    A filled grid with no repeated digit in any row, column or box is a valid
    solution, and for a puzzle with exactly one solution it is the right one, so
    this never needs the answer. When a grid isn't valid, the next try starts
    from the puzzle plus every predicted digit that the model was at least
    `keep_above` sure of and that isn't part of a repeat, and solves all at once
    again from that fuller board. Stops after `tries` runs, returning the last
    attempt for puzzles that never became valid.
    """
    current, result, retry = board, None, None
    for attempt in range(tries):
        with autocast(board.device):
            confidence, digit = model(current).float().softmax(-1).max(-1)
        filled = torch.where(current > 0, current, digit + 1)
        result = filled if result is None else torch.where(retry[:, None], filled, result)
        bad = conflicts(result)
        retry = bad.any(1)
        if not retry.any() or attempt == tries - 1:
            return result
        keep = (board == 0) & ~bad & (confidence >= keep_above)
        current = torch.where(keep, result, board)


METHODS = {
    "all at once": fill_all,
    "all at once, retries": fill_all_with_retries,
    "one cell at a time": fill_one_at_a_time,
}


def score(method, model, loader, device):
    cells = correct = solved = puzzles = 0
    for board, target, mask in loader:
        board, target, mask = board.to(device), target.to(device), mask.to(device)
        right = method(model, board) == target + 1
        cells += mask.sum().item()
        correct += (right & mask).sum().item()
        solved += right.all(1).sum().item()
        puzzles += len(board)
    return {"cell_acc": correct / cells, "puzzle_acc": solved / puzzles}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--data", nargs="+", default=["data/generated_test.parquet"])
    parser.add_argument("--limit", type=int, default=2000, help="puzzles to use from each file")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    for path in args.data:
        dataset = SudokuDataset(path)
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        loader = DataLoader(dataset, batch_size=args.batch_size)
        print(f"{path} ({len(dataset)} puzzles)")
        for name, method in METHODS.items():
            start = time.perf_counter()
            result = score(method, model, loader, device)
            print(
                f"  {name:20s} cell acc {result['cell_acc']:.3f}  "
                f"puzzle acc {result['puzzle_acc']:.3f}  {time.perf_counter() - start:.1f}s"
            )


if __name__ == "__main__":
    main()
