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
from .solve import fill_all, fill_one_at_a_time
from .train import load_model

METHODS = {"all at once": fill_all, "one cell at a time": fill_one_at_a_time}


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
