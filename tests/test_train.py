import numpy as np
import torch

from sudoku.augment import generate
from sudoku.data import write_records
from sudoku.train import evaluate, load_model, main, masked_loss, warmup_cosine

from test_data import make_record


def test_masked_loss_ignores_given_cells():
    target = torch.zeros(1, 81, dtype=torch.long)
    mask = torch.zeros(1, 81, dtype=torch.bool)
    mask[0, :10] = True
    logits = torch.zeros(1, 81, 9)
    logits[0, :10, 0] = 20.0  # confident and right on the blank cells
    logits[0, 10:, 5] = 20.0  # confident and wrong on the givens
    assert masked_loss(logits, target, mask).item() < 1e-6


def test_evaluate_scores_perfect_and_wrong_predictions():
    class Oracle(torch.nn.Module):
        def __init__(self, target, offset):
            super().__init__()
            self.target, self.offset = target, offset

        def forward(self, board):
            return torch.nn.functional.one_hot((self.target + self.offset) % 9, 9).float()

    target = torch.randint(0, 9, (4, 81))
    mask = torch.rand(4, 81) < 0.5
    loader = [(torch.zeros(4, 81, dtype=torch.long), target, mask)]
    good = evaluate(Oracle(target, 0), loader, torch.device("cpu"))
    assert good["cell_acc"] == 1.0 and good["puzzle_acc"] == 1.0
    bad = evaluate(Oracle(target, 1), loader, torch.device("cpu"))
    assert bad["cell_acc"] == 0.0 and bad["puzzle_acc"] == 0.0


def test_warmup_cosine_shape():
    f = warmup_cosine(total_steps=100, warmup_steps=10)
    assert f(0) == 0.1 and f(9) == 1.0
    assert f(10) == 1.0 and abs(f(55) - 0.5) < 1e-9 and f(100) == 0.0


def test_main_trains_and_saves_checkpoint(tmp_path):
    records = generate([make_record()], 12, np.random.default_rng(0), min_ops=20, max_ops=30)
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    write_records(train, records[:8], source="test")
    write_records(test, records[8:], source="test")
    out = tmp_path / "model.pt"
    main([
        "--train", str(train), "--test", str(test), "--out", str(out), "--epochs", "2",
        "--batch-size", "4", "--d-model", "32", "--layers", "1", "--heads", "2", "--loops", "2", "--device", "cpu",
    ])
    checkpoint = torch.load(out)
    assert checkpoint["model_args"]["d_model"] == 32 and checkpoint["model_args"]["loops"] == 2
    assert 0.0 <= checkpoint["metrics"]["cell_acc"] <= 1.0
    model = load_model(out)
    assert model.loops == 2
    assert model(torch.zeros(1, 81, dtype=torch.long)).shape == (1, 81, 9)


def test_load_model_accepts_checkpoints_without_loops(tmp_path):
    from sudoku.model import SudokuTransformer

    args = dict(d_model=32, n_layers=1, n_heads=2, dropout=0.0)  # saved before loops existed
    path = tmp_path / "old.pt"
    torch.save({"model_args": args, "state_dict": SudokuTransformer(**args).state_dict()}, path)
    assert load_model(path).loops == 1
