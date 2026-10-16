import numpy as np
import pyarrow.parquet as pq
import pytest

from sudoku.data import apply_moves, encode, read_records, validate_record
from sudoku.moves import UNITS, _Board, count_solutions, guided_moves, logical_moves, main

from test_data import PUZZLE, SOLUTION

# Needs a naked triple partway through (one of the hand-solved puzzles).
TRIPLE_PUZZLE = "090023870010500000400000000300001000604000109000700002000000003000007060032840050"
TRIPLE_SOLUTION = "596423871217568394483179526329681745674235189851794632768952413945317268132846957"

# Far beyond singles and simple eliminations; needs guessing or advanced chains.
HARD_PUZZLE = "800000000003600000070090200050007000000045700000100030001000068008500010090000400"
HARD_SOLUTION = "812753649943682175675491283154237896369845721287169534521974368438526917796318452"

# The same puzzle as TRIPLE_PUZZLE with a given removed, leaving two solutions.
AMBIGUOUS_PUZZLE = "006008000000200000809000357070000500008090600001000040534000901000001000000000700"
AMBIGUOUS_SOLUTION = "716358294345279186829416357273864519458193672961527843534782961687931425192645738"


def assert_logical(puzzle, moves):
    """Every move is a single on the board before it, given only the candidates logic can see."""
    board = _Board(puzzle)
    for cell, digit in moves.tolist():
        while (cell, digit) not in board.singles():
            assert board.eliminate(), f"move {(cell, digit)} isn't deducible"
        board.place(cell, digit)


@pytest.mark.parametrize("puzzle, solution", [(PUZZLE, SOLUTION), (TRIPLE_PUZZLE, TRIPLE_SOLUTION)])
def test_logical_moves_solve_and_validate(puzzle, solution):
    p, s = encode(puzzle), encode(solution)
    moves = logical_moves(p, s, np.random.default_rng(0))
    validate_record(p, s, moves)
    assert np.array_equal(apply_moves(p, moves, len(moves)), s)
    assert_logical(p, moves)


def test_triple_puzzle_needs_eliminations():
    board = _Board(encode(TRIPLE_PUZZLE))
    for cell, digit in logical_moves(encode(TRIPLE_PUZZLE), encode(TRIPLE_SOLUTION)).tolist():
        if not board.singles():
            return  # reached a point where only eliminations could continue
        board.place(cell, digit)
    pytest.fail("expected the solve to need eliminations at some point")


def test_seed_changes_order_not_result():
    p, s = encode(PUZZLE), encode(SOLUTION)
    a = logical_moves(p, s, np.random.default_rng(1))
    b = logical_moves(p, s, np.random.default_rng(2))
    assert not np.array_equal(a, b)
    assert sorted(a.tolist()) == sorted(b.tolist())
    assert np.array_equal(logical_moves(p, s), logical_moves(p, s))  # no rng: deterministic


@pytest.mark.parametrize("puzzle, solution", [(HARD_PUZZLE, HARD_SOLUTION), (AMBIGUOUS_PUZZLE, AMBIGUOUS_SOLUTION)])
def test_gives_up_instead_of_guessing(puzzle, solution):
    assert logical_moves(encode(puzzle), encode(solution)) is None


def test_wrong_solution_raises():
    s = encode(SOLUTION)
    s[[2, 3]] = s[[3, 2]]  # cells 2 and 3 are both blank in PUZZLE
    with pytest.raises(ValueError):
        logical_moves(encode(PUZZLE), s)


def test_units_cover_grid():
    assert len(UNITS) == 27 and all(len(u) == 9 for u in UNITS)
    assert all(sorted(sum(UNITS[i : i + 9], [])) == list(range(81)) for i in (0, 9, 18))


def test_main_writes_records_and_skips_hard(tmp_path):
    csv = tmp_path / "puzzles.csv"
    csv.write_text(
        "quizzes,solutions\n"
        f"{PUZZLE},{SOLUTION}\n"
        f"{HARD_PUZZLE},{HARD_SOLUTION}\n"
        f"{TRIPLE_PUZZLE.replace('0', '.')},{TRIPLE_SOLUTION}\n"
    )
    out = tmp_path / "out.parquet"
    main(["--input", str(csv), "--output", str(out), "--source", "public"])
    records = read_records(out)
    assert len(records) == 2
    assert pq.read_table(out).column("source").to_pylist() == ["public", "public"]
    assert_logical(records[1][0], records[1][2])


def test_guided_moves_match_logical_when_no_guess_needed():
    p, s = encode(PUZZLE), encode(SOLUTION)
    moves, guesses = guided_moves(p, s)
    assert guesses == 0
    assert np.array_equal(moves, logical_moves(p, s))


def test_guided_moves_finish_hard_puzzle():
    p, s = encode(HARD_PUZZLE), encode(HARD_SOLUTION)
    moves, guesses = guided_moves(p, s, np.random.default_rng(0))
    assert guesses > 0
    validate_record(p, s, moves)


@pytest.mark.parametrize(
    "puzzle, expected",
    [(PUZZLE, 1), (HARD_PUZZLE, 1), (AMBIGUOUS_PUZZLE, 2), ("0" * 81, 2)],
)
def test_count_solutions(puzzle, expected):
    assert count_solutions(encode(puzzle)) == expected


def test_main_hard_only_keeps_unique_hard_puzzles(tmp_path):
    csv = tmp_path / "puzzles.csv"
    csv.write_text(
        "puzzle,solution\n"
        f"{PUZZLE},{SOLUTION}\n"
        f"{AMBIGUOUS_PUZZLE},{AMBIGUOUS_SOLUTION}\n"
        f"{HARD_PUZZLE},{HARD_SOLUTION}\n"
        f"{HARD_PUZZLE},{HARD_SOLUTION}\n"
    )
    out = tmp_path / "hard.parquet"
    main(["--input", str(csv), "--output", str(out), "--source", "hard", "--hard-only", "--count", "1"])
    records = read_records(out)
    assert len(records) == 1  # easy and ambiguous skipped, stopped at --count
    assert np.array_equal(records[0][0], encode(HARD_PUZZLE))
