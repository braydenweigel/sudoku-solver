import numpy as np
import pyarrow.parquet as pq
import pytest

from sudoku.augment import generate
from sudoku.data import decode, read_records, write_records
from sudoku.split import main, split_records

from test_data import make_record


def make_records(n):
    return generate([make_record()], n, np.random.default_rng(0), min_ops=50, max_ops=60)


def test_split_records_partitions():
    records = make_records(10)
    train, test = split_records(records, 3, np.random.default_rng(0))
    assert len(train) == 7 and len(test) == 3
    keys = lambda rs: {decode(p) for p, _, _ in rs}
    assert keys(train).isdisjoint(keys(test))
    assert keys(train) | keys(test) == keys(records)


def test_split_records_seed_is_reproducible():
    records = make_records(10)
    a = split_records(records, 2, np.random.default_rng(5))[1]
    b = split_records(records, 2, np.random.default_rng(5))[1]
    assert [decode(p) for p, _, _ in a] == [decode(p) for p, _, _ in b]


@pytest.mark.parametrize("count", [0, 10])
def test_split_records_rejects_bad_count(count):
    with pytest.raises(ValueError):
        split_records(make_records(10), count, np.random.default_rng(0))


def test_main_writes_both_files(tmp_path):
    src = tmp_path / "manual.parquet"
    write_records(src, make_records(5), source="manual")
    main(["--input", str(src), "--test-count", "2", "--seed", "0"])
    train, test = tmp_path / "manual_train.parquet", tmp_path / "manual_test.parquet"
    assert len(read_records(train)) == 3 and len(read_records(test)) == 2
    for path in (train, test):
        assert set(pq.read_table(path).column("source").to_pylist()) == {"manual"}


def test_main_carries_move_kinds(tmp_path):
    from sudoku.data import read_kinds

    src = tmp_path / "public.parquet"
    records = make_records(5)
    kinds = [np.full(len(r[2]), i, dtype=np.uint8) % 3 for i, r in enumerate(records)]
    write_records(src, records, source="public", kinds=kinds)
    main(["--input", str(src), "--test-count", "2", "--seed", "0"])
    by_puzzle = {decode(r[0]): k for r, k in zip(records, kinds)}
    for name in ("train", "test"):
        path = tmp_path / f"public_{name}.parquet"
        for (puzzle, _, _), k in zip(read_records(path), read_kinds(path)):
            assert np.array_equal(k, by_puzzle[decode(puzzle)])
