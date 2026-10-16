import numpy as np
import pyarrow.parquet as pq
import pytest

from sudoku.data import MOVE_BASIC, MOVE_SEARCH, MOVE_TRIAL, apply_moves, encode, read_kinds, read_records, validate_record
import sudoku.moves as moves_module
from sudoku.moves import UNITS, _Board, count_solutions, logical_moves, main, make_moves

from test_data import PUZZLE, SOLUTION

# Needs a naked triple partway through (one of the hand-solved puzzles).
TRIPLE_PUZZLE = "090023870010500000400000000300001000604000109000700002000000003000007060032840050"
TRIPLE_SOLUTION = "596423871217568394483179526329681745674235189851794632768952413945317268132846957"

# Far beyond singles and simple eliminations; needs guessing or advanced chains.
HARD_PUZZLE = "800000000003600000070090200050007000000045700000100030001000068008500010090000400"
HARD_SOLUTION = "812753649943682175675491283154237896369845721287169534521974368438526917796318452"

# Beyond basic reasoning, but one trial elimination is enough.
TRIAL_PUZZLE = "380100000000050600000900003040000000005018000000090561060024780800000006004080020"
TRIAL_SOLUTION = "386172954419853672527946813741365298695218437238497561163524789852739146974681325"

# A different puzzle from TRIPLE_PUZZLE with a given removed, leaving two solutions.
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


def write_csv(path, rows):
    lines = ["puzzle,solution"] + [f"{p},{s}" for p, s in rows]
    path.write_text("\n".join(lines) + "\n")


def test_main_writes_every_unique_puzzle_by_default(tmp_path):
    csv = tmp_path / "puzzles.csv"
    write_csv(csv, [(PUZZLE, SOLUTION), (TRIAL_PUZZLE, TRIAL_SOLUTION), (AMBIGUOUS_PUZZLE, AMBIGUOUS_SOLUTION),
                    (TRIPLE_PUZZLE.replace("0", "."), TRIPLE_SOLUTION)])
    out = tmp_path / "out.parquet"
    main(["--input", str(csv), "--output", str(out), "--source", "public"])
    records = read_records(out)
    assert [r[0].tolist() for r in records] == [encode(x).tolist() for x in (PUZZLE, TRIAL_PUZZLE, TRIPLE_PUZZLE)]
    assert pq.read_table(out).column("source").to_pylist() == ["public"] * 3
    for record in records:
        validate_record(*record)
    assert_logical(records[2][0], records[2][2])


def test_main_basic_level_skips_hard(tmp_path):
    csv = tmp_path / "puzzles.csv"
    write_csv(csv, [(PUZZLE, SOLUTION), (TRIAL_PUZZLE, TRIAL_SOLUTION)])
    out = tmp_path / "out.parquet"
    main(["--input", str(csv), "--output", str(out), "--source", "public", "--level", "basic"])
    assert [r[0].tolist() for r in read_records(out)] == [encode(PUZZLE).tolist()]


@pytest.mark.parametrize(
    "puzzle, expected",
    [(PUZZLE, 1), (HARD_PUZZLE, 1), (AMBIGUOUS_PUZZLE, 2), ("0" * 81, 2)],
)
def test_count_solutions(puzzle, expected):
    assert count_solutions(encode(puzzle)) == expected


@pytest.mark.parametrize(
    "puzzle, solution, solved_at",
    [(PUZZLE, SOLUTION, "basic"), (TRIAL_PUZZLE, TRIAL_SOLUTION, "trial"), (HARD_PUZZLE, HARD_SOLUTION, "search")],
)
def test_levels(puzzle, solution, solved_at):
    p, s = encode(puzzle), encode(solution)
    for level in ("basic", "trial", "search"):
        moves, kinds = make_moves(p, s, np.random.default_rng(0), level)
        if ["basic", "trial", "search"].index(level) < ["basic", "trial", "search"].index(solved_at):
            assert moves is None
        else:
            validate_record(p, s, moves)  # every move is a correct placement, each cell once
            assert kinds.shape == (len(moves),)
            assert (kinds == MOVE_TRIAL).any() == (solved_at != "basic")
            assert (kinds == MOVE_SEARCH).any() == (solved_at == "search")


def test_kinds_mark_moves_basic_reasoning_cannot_find():
    """A move is MOVE_BASIC exactly when basic reasoning alone finds it on the board before it."""
    p, s = encode(HARD_PUZZLE), encode(HARD_SOLUTION)
    moves, kinds = make_moves(p, s, np.random.default_rng(0))
    for t, ((cell, digit), kind) in enumerate(zip(moves.tolist(), kinds.tolist())):
        board = _Board(apply_moves(p, moves, t))
        while not board.singles() and board.eliminate():
            pass
        assert bool(board.singles()) == (kind == MOVE_BASIC), f"step {t}"
    assert 0 < (kinds != MOVE_BASIC).sum() < 10


def test_trial_eliminations_are_sound(monkeypatch):
    """Every contradiction a trial finds is real: exhaustive search finds no solution either."""
    found = []
    original = moves_module._leads_to_contradiction

    def checked(board):
        start = board.values.copy()
        result = original(board)
        if result:
            found.append(start)
        return result

    monkeypatch.setattr(moves_module, "_leads_to_contradiction", checked)
    for puzzle, solution in ((TRIAL_PUZZLE, TRIAL_SOLUTION), (HARD_PUZZLE, HARD_SOLUTION)):
        make_moves(encode(puzzle), encode(solution), np.random.default_rng(0))
    assert len(found) > 10
    assert all(count_solutions(board, limit=1) == 0 for board in found)


def test_contradiction():
    assert not _Board(encode(PUZZLE)).contradiction()
    s = encode(SOLUTION)
    # A blank cell with no candidates: row 0 lacks only 6, but column 2 already has a 6.
    stuck = s.copy()
    stuck[2], stuck[3] = 0, 4
    assert _Board(stuck).contradiction()
    # Every blank has candidates, but column 6 has nowhere left for its 4.
    hidden = s.copy()
    hidden[[4, 15, 27, 36, 49, 51, 53, 69, 77]] = 0
    hidden[33] = 8
    board = _Board(hidden)
    assert all(board.cands[c] for c in range(81) if hidden[c] == 0)
    assert board.contradiction()


def test_main_hard_only_keeps_unique_hard_puzzles(tmp_path):
    csv = tmp_path / "puzzles.csv"
    write_csv(csv, [(PUZZLE, SOLUTION), (AMBIGUOUS_PUZZLE, AMBIGUOUS_SOLUTION), (TRIAL_PUZZLE, TRIAL_SOLUTION),
                    (TRIPLE_PUZZLE, TRIPLE_SOLUTION)])
    out = tmp_path / "hard.parquet"
    main(["--input", str(csv), "--output", str(out), "--source", "hard", "--hard-only", "--count", "1"])
    records = read_records(out)
    assert len(records) == 1  # easy and ambiguous skipped, stopped at --count
    assert np.array_equal(records[0][0], encode(TRIAL_PUZZLE))


def test_main_exclude_skips_puzzles_in_other_files(tmp_path):
    csv = tmp_path / "puzzles.csv"
    write_csv(csv, [(PUZZLE, SOLUTION), (TRIAL_PUZZLE, TRIAL_SOLUTION)])
    test_set = tmp_path / "test.parquet"
    main(["--input", str(csv), "--output", str(test_set), "--source", "hard", "--hard-only"])
    out = tmp_path / "train.parquet"
    main(["--input", str(csv), "--output", str(out), "--source", "public", "--exclude", str(test_set)])
    assert [r[0].tolist() for r in read_records(out)] == [encode(PUZZLE).tolist()]
