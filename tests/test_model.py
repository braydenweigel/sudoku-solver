import torch

from sudoku.model import SudokuTransformer


def small_model():
    return SudokuTransformer(d_model=32, n_layers=2, n_heads=2, dropout=0.0)


def test_output_shape():
    board = torch.randint(0, 10, (4, 81))
    logits = small_model()(board)
    assert logits.shape == (4, 81, 9) and logits.dtype == torch.float32


def test_position_embeddings_match_grid():
    model = small_model()
    assert model.rows[:10].tolist() == [0] * 9 + [1]
    assert model.cols[:10].tolist() == list(range(9)) + [0]
    # Cell 40 is the centre: row 4, column 4, box 4. Cell 80 is box 8.
    assert (model.rows[40], model.cols[40], model.boxes[40]) == (4, 4, 4)
    assert model.boxes[80] == 8 and model.boxes[30] == 4 and model.boxes[6] == 2


def test_every_cell_sees_every_other_cell():
    model = small_model().eval()
    board = torch.zeros(1, 81, dtype=torch.long)
    changed = board.clone()
    changed[0, 0] = 5  # top-left cell only
    with torch.no_grad():
        diff = (model(board) - model(changed)).abs().sum(-1)[0]
    assert (diff > 0).all()  # even the bottom-right cell's prediction moves


def test_gradients_reach_all_parameters():
    model = small_model()
    model(torch.randint(0, 10, (2, 81))).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_one_loop_matches_plain_encoder():
    model = small_model().eval()
    board = torch.randint(0, 10, (3, 81))
    x = model.digit(board) + model.row(model.rows) + model.col(model.cols) + model.box(model.boxes)
    with torch.no_grad():
        assert torch.allclose(model(board), model.head(model.encoder(x)), atol=1e-6)


def test_loops_reuse_the_same_layers():
    one = SudokuTransformer(d_model=32, n_layers=2, n_heads=2, dropout=0.0, loops=1)
    three = SudokuTransformer(d_model=32, n_layers=2, n_heads=2, dropout=0.0, loops=3)
    assert sum(p.numel() for p in one.parameters()) == sum(p.numel() for p in three.parameters())
    three.load_state_dict(one.state_dict())  # same weights fit either way
    board = torch.randint(0, 10, (2, 81))
    with torch.no_grad():
        out = three.eval()(board)
        assert out.shape == (2, 81, 9)
        assert not torch.allclose(out, one.eval()(board))  # but more rounds compute something different


def test_loops_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        SudokuTransformer(loops=0)
