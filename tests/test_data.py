import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from sudoku.data import (
    apply_moves,
    apply_symmetry,
    decode,
    encode,
    is_solved,
    random_symmetry,
    read_records,
    validate_record,
    write_records,
)
from sudoku.dataset import SudokuDataset

PUZZLE = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


def make_record():
    puzzle, solution = encode(PUZZLE), encode(SOLUTION)
    cells = np.flatnonzero(puzzle == 0)
    moves = np.stack([cells, solution[cells]], axis=1).astype(np.uint8)
    return puzzle, solution, moves


def test_encode_decode_round_trip():
    assert decode(encode(PUZZLE)) == PUZZLE
    assert decode(encode(PUZZLE.replace("0", "."))) == PUZZLE


def test_encode_rejects_bad_input():
    with pytest.raises(ValueError):
        encode("123")
    with pytest.raises(ValueError):
        encode("x" * 81)


def test_apply_moves():
    puzzle, solution, moves = make_record()
    assert np.array_equal(apply_moves(puzzle, moves, 0), puzzle)
    assert np.array_equal(apply_moves(puzzle, moves, len(moves)), solution)
    partial = apply_moves(puzzle, moves, 10)
    assert (partial == 0).sum() == len(moves) - 10
    assert np.array_equal(puzzle, encode(PUZZLE))  # input not mutated


def test_validate_record_accepts_valid():
    validate_record(*make_record())


def test_validate_record_rejects_missing_move():
    puzzle, solution, moves = make_record()
    with pytest.raises(ValueError):
        validate_record(puzzle, solution, moves[:-1])


def test_validate_record_rejects_duplicate_move():
    puzzle, solution, moves = make_record()
    moves[1] = moves[0]
    with pytest.raises(ValueError):
        validate_record(puzzle, solution, moves)


def test_validate_record_rejects_wrong_digit():
    puzzle, solution, moves = make_record()
    moves[0, 1] = moves[0, 1] % 9 + 1
    with pytest.raises(ValueError):
        validate_record(puzzle, solution, moves)


def test_validate_record_rejects_bad_solution():
    puzzle, solution, moves = make_record()
    solution[[0, 1]] = solution[[1, 0]]
    with pytest.raises(ValueError):
        validate_record(puzzle, solution, moves)


def test_symmetry_keeps_record_valid():
    record = make_record()
    rng = np.random.default_rng(0)
    changed = 0
    for _ in range(200):
        puzzle, solution, moves = apply_symmetry(*record, *random_symmetry(rng))
        assert is_solved(solution)
        validate_record(puzzle, solution, moves)
        changed += not np.array_equal(solution, record[1])
    assert changed > 190


def test_symmetry_preserves_move_order():
    puzzle, solution, moves = make_record()
    perm, digit_map = random_symmetry(np.random.default_rng(1))
    new_puzzle, _, new_moves = apply_symmetry(puzzle, solution, moves, perm, digit_map)
    for t in (1, 17, len(moves)):
        expected = digit_map[apply_moves(puzzle, moves, t)[perm]]
        assert np.array_equal(apply_moves(new_puzzle, new_moves, t), expected)


@pytest.fixture
def split(tmp_path):
    path = tmp_path / "train.parquet"
    rng = np.random.default_rng(2)
    base = make_record()
    records = [base] + [apply_symmetry(*base, *random_symmetry(rng)) for _ in range(2)]
    write_records(path, records, source="test")
    return path


def test_source_column(split):
    assert pq.read_table(split).column("source").to_pylist() == ["test"] * 3


def test_parquet_round_trip(split):
    records = read_records(split)
    assert len(records) == 3
    for got, want in zip(records[0], make_record()):
        assert got.dtype == np.uint8
        assert np.array_equal(got, want)


def test_dataset_eval_mode(split):
    dataset = SudokuDataset(split)
    assert len(dataset) == 3
    board, target, mask = dataset[0]
    assert board.shape == target.shape == mask.shape == (81,)
    assert board.dtype == target.dtype == torch.int64 and mask.dtype == torch.bool
    assert decode(board.numpy()) == PUZZLE
    assert decode(target.numpy() + 1) == SOLUTION
    assert torch.equal(mask, board == 0)


def test_dataset_train_mode(split):
    dataset = SudokuDataset(split, random_step=True, augment=True)
    blanks = set()
    for _ in range(50):
        board, target, mask = dataset[0]
        assert mask.any()  # never a fully solved board
        assert torch.equal(board[~mask], target[~mask] + 1)
        assert is_solved(target.numpy() + 1)
        blanks.add(int(mask.sum()))
    assert len(blanks) > 5


def test_dataset_one_hot(split):
    board, _, mask = SudokuDataset(split, one_hot=True)[0]
    assert board.shape == (10, 9, 9) and board.dtype == torch.float32
    assert torch.equal(board.argmax(0).reshape(81), torch.from_numpy(encode(PUZZLE).astype(np.int64)))
    assert torch.equal(board[0].reshape(81).bool(), mask)
