import numpy as np
import pyarrow.parquet as pq
import pytest

from sudoku.augment import (
    OPERATIONS,
    augment_record,
    generate,
    main,
    mirror_horizontal,
    mirror_vertical,
    rotate,
    swap_bands,
    swap_columns,
    swap_rows,
    swap_stacks,
)
from sudoku.data import apply_moves, apply_symmetry, decode, read_records, validate_record, write_records

from test_data import make_record


@pytest.mark.parametrize("op", OPERATIONS, ids=lambda op: op.__name__)
def test_operation_keeps_record_valid(op):
    record = make_record()
    rng = np.random.default_rng(0)
    for _ in range(20):
        perm, digit_map = op(rng)
        assert sorted(perm.tolist()) == list(range(81))
        assert digit_map[0] == 0 and sorted(digit_map[1:].tolist()) == list(range(1, 10))
        new = apply_symmetry(*record, perm, digit_map)
        validate_record(*new)
        # Every intermediate board is the transformed original board.
        for t in (0, 10, len(record[2])):
            assert np.array_equal(apply_moves(new[0], new[2], t), digit_map[apply_moves(record[0], record[2], t)[perm]])


def test_swap_rows_swaps_two_rows_in_one_band():
    rng = np.random.default_rng(0)
    pairs = set()
    for _ in range(300):
        perm, digit_map = swap_rows(rng)
        assert np.array_equal(digit_map, np.arange(10))
        rows = perm.reshape(9, 9)[:, 0] // 9  # source row of each new row
        moved = np.flatnonzero(rows != np.arange(9))
        assert len(moved) == 2
        a, b = moved
        assert a // 3 == b // 3 and rows[a] == b and rows[b] == a
        assert np.array_equal(perm.reshape(9, 9), np.arange(81).reshape(9, 9)[rows])
        pairs.add((int(a), int(b)))
    assert len(pairs) == 9  # every pair in every band is reachable


def test_swap_columns_swaps_two_columns_in_one_stack():
    rng = np.random.default_rng(0)
    pairs = set()
    for _ in range(300):
        perm, digit_map = swap_columns(rng)
        assert np.array_equal(digit_map, np.arange(10))
        cols = perm.reshape(9, 9)[0]  # source column of each new column
        moved = np.flatnonzero(cols != np.arange(9))
        assert len(moved) == 2
        a, b = moved
        assert a // 3 == b // 3 and cols[a] == b and cols[b] == a
        assert np.array_equal(perm.reshape(9, 9), np.arange(81).reshape(9, 9)[:, cols])
        pairs.add((int(a), int(b)))
    assert len(pairs) == 9  # every pair in every stack is reachable


@pytest.mark.parametrize("op, axis", [(swap_bands, "rows"), (swap_stacks, "columns")])
def test_swap_blocks_swaps_two_whole_blocks(op, axis):
    rng = np.random.default_rng(0)
    pairs = set()
    for _ in range(100):
        perm, digit_map = op(rng)
        assert np.array_equal(digit_map, np.arange(10))
        grid = perm.reshape(9, 9)
        lines = grid[:, 0] // 9 if axis == "rows" else grid[0]  # source line of each new line
        expected = np.arange(81).reshape(9, 9)
        assert np.array_equal(grid, expected[lines] if axis == "rows" else expected[:, lines])
        blocks = lines.reshape(3, 3) // 3  # source block of each new block
        assert (blocks == blocks[:, :1]).all()  # each block moves whole
        assert np.array_equal(lines.reshape(3, 3) % 3, np.tile(np.arange(3), (3, 1)))  # order kept inside
        moved = np.flatnonzero(blocks[:, 0] != np.arange(3))
        assert len(moved) == 2
        a, b = moved
        assert blocks[a, 0] == b and blocks[b, 0] == a
        pairs.add((int(a), int(b)))
    assert pairs == {(0, 1), (0, 2), (1, 2)}


def test_rotate_turns_grid_clockwise():
    perm, digit_map = rotate(np.random.default_rng(0))
    assert np.array_equal(digit_map, np.arange(10))
    grid = perm.reshape(9, 9)
    # New row r is old column r read from the bottom up.
    for r in range(9):
        assert grid[r].tolist() == [row * 9 + r for row in range(8, -1, -1)]
    # Four quarter turns bring every cell back.
    cells = np.arange(81)
    for _ in range(4):
        cells = cells[perm]
    assert np.array_equal(cells, np.arange(81))


def test_mirror_horizontal_reverses_each_row():
    perm, digit_map = mirror_horizontal(np.random.default_rng(0))
    assert np.array_equal(digit_map, np.arange(10))
    for r in range(9):
        assert perm.reshape(9, 9)[r].tolist() == [r * 9 + c for c in range(8, -1, -1)]
    assert np.array_equal(perm[perm], np.arange(81))  # flipping twice undoes it


def test_mirror_vertical_reverses_row_order():
    perm, digit_map = mirror_vertical(np.random.default_rng(0))
    assert np.array_equal(digit_map, np.arange(10))
    for r in range(9):
        assert perm.reshape(9, 9)[r].tolist() == [(8 - r) * 9 + c for c in range(9)]
    assert np.array_equal(perm[perm], np.arange(81))


def test_augment_record_applies_random_number_of_ops():
    calls = []

    def counting_op(rng):
        calls.append(1)
        return np.arange(81), np.arange(10, dtype=np.uint8)

    rng = np.random.default_rng(0)
    counts = set()
    for _ in range(50):
        calls.clear()
        augment_record(make_record(), rng, min_ops=2, max_ops=4, operations=[counting_op])
        counts.add(len(calls))
    assert counts == {2, 3, 4}


def test_augment_record_rejects_bad_op_range():
    for lo, hi in ((0, 5), (6, 5)):
        with pytest.raises(ValueError):
            augment_record(make_record(), np.random.default_rng(0), min_ops=lo, max_ops=hi)


def test_generate_makes_distinct_valid_records():
    base = make_record()
    out = generate([base], 30, np.random.default_rng(1))
    assert len(out) == 30
    keys = {decode(p) for p, _, _ in out}
    assert len(keys) == 30 and decode(base[0]) not in keys
    for record in out:
        validate_record(*record)
        assert len(record[2]) == len(base[2])


def test_generate_fails_when_too_few_distinct():
    def identity(rng):
        return np.arange(81), np.arange(10, dtype=np.uint8)

    with pytest.raises(RuntimeError):
        generate([make_record()], 2, np.random.default_rng(0), operations=[identity], max_attempts=10)


def test_main_writes_file(tmp_path):
    src, dst = tmp_path / "manual.parquet", tmp_path / "generated.parquet"
    write_records(src, [make_record()], source="manual")
    main(["--input", str(src), "--output", str(dst), "--count", "5", "--seed", "0"])
    assert len(read_records(dst)) == 5
    assert pq.read_table(dst).column("source").to_pylist() == ["generated"] * 5


def test_main_carries_move_kinds(tmp_path):
    from sudoku.data import read_kinds

    src, dst = tmp_path / "base.parquet", tmp_path / "generated.parquet"
    base = make_record()
    kinds = np.zeros(len(base[2]), dtype=np.uint8)
    kinds[5] = 1
    write_records(src, [base], source="public", kinds=[kinds])
    main(["--input", str(src), "--output", str(dst), "--count", "3", "--seed", "0", "--min-ops", "5", "--max-ops", "10"])
    assert all(np.array_equal(k, kinds) for k in read_kinds(dst))
