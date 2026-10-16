"""Train a SudokuTransformer on generated records.

Usage:
    python -m sudoku.train --epochs 10

Training uses random intermediate boards (random_step), so each epoch sees
different partly solved puzzles. Evaluation always uses the original puzzles.
The checkpoint with the lowest test loss is saved to --out.
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import SudokuDataset
from .model import SudokuTransformer


def masked_loss(logits, target, mask):
    """Cross-entropy over the cells that were blank in the input only."""
    return F.cross_entropy(logits[mask].float(), target[mask])


def autocast(device):
    """bfloat16 mixed precision on the GPU; a no-op on the CPU."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def evaluate(model, loader, device):
    """Loss and accuracy on blank cells, and the share of puzzles fully solved."""
    model.eval()
    loss_sum = cells = correct = solved = puzzles = 0
    for board, target, mask in loader:
        board, target, mask = board.to(device), target.to(device), mask.to(device)
        with autocast(device):
            logits = model(board)
        loss_sum += F.cross_entropy(logits[mask].float(), target[mask], reduction="sum").item()
        right = logits.argmax(-1) == target
        cells += mask.sum().item()
        correct += (right & mask).sum().item()
        solved += (right | ~mask).all(1).sum().item()
        puzzles += len(board)
    model.train()
    return {"loss": loss_sum / cells, "cell_acc": correct / cells, "puzzle_acc": solved / puzzles}


def warmup_cosine(total_steps, warmup_steps):
    """Learning-rate multiplier: linear warmup, then cosine decay to zero."""

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return factor


def load_model(path, device="cpu"):
    """Rebuild a model from a checkpoint written by this script."""
    checkpoint = torch.load(path, map_location=device)
    model = SudokuTransformer(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", default="data/generated_train.parquet")
    parser.add_argument("--test", default="data/generated_test.parquet")
    parser.add_argument("--out", default="checkpoints/model.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loops", type=int, default=1, help="run the layers this many times in a row")
    parser.add_argument("--augment", action="store_true", help="also apply random symmetries while training")
    parser.add_argument(
        "--hard-fraction",
        type=float,
        default=0.0,
        help="chance of training on the board just before a move that needed more than basic reasoning, "
        "for puzzles that have one (needs move kinds in the training file)",
    )
    parser.add_argument("--limit", type=int, default=None, help="use only the first N training records")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_set = SudokuDataset(args.train, random_step=True, augment=args.augment, hard_fraction=args.hard_fraction)
    if args.limit:
        train_set = Subset(train_set, range(min(args.limit, len(train_set))))
    test_set = SudokuDataset(args.test)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, drop_last=len(train_set) > args.batch_size, **loader_args)
    test_loader = DataLoader(test_set, **loader_args)

    model_args = dict(
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, dropout=args.dropout, loops=args.loops
    )
    model = SudokuTransformer(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(total_steps, warmup_steps=max(1, min(500, total_steps // 10)))
    )
    print(
        f"{len(train_set)} train / {len(test_set)} test records, "
        f"{sum(p.numel() for p in model.parameters()):,} parameters, device {device}"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = math.inf
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        loss_sum = batches = 0
        for board, target, mask in train_loader:
            board, target, mask = board.to(device), target.to(device), mask.to(device)
            with autocast(device):
                logits = model(board)
            loss = masked_loss(logits, target, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            batches += 1

        metrics = evaluate(model, test_loader, device)
        saved = metrics["loss"] < best
        if saved:
            best = metrics["loss"]
            torch.save(
                {"model_args": model_args, "state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics},
                out,
            )
        print(
            f"epoch {epoch:3d}  train loss {loss_sum / batches:.4f}  "
            f"test loss {metrics['loss']:.4f}  cell acc {metrics['cell_acc']:.3f}  "
            f"puzzle acc {metrics['puzzle_acc']:.3f}  {time.perf_counter() - start:.0f}s"
            + ("  saved" if saved else "")
        )


if __name__ == "__main__":
    main()
