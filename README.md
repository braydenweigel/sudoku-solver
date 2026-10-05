# sudoku-solver
ML Model to solve sudoku puzzles

## Data Format

Training data is stored as one record per puzzle, in one Parquet file per split (train/val/test). Splits are made by puzzle, so no puzzle's boards appear in more than one split.

| Column | Parquet type | Meaning |
|---|---|---|
| `puzzle` | string, 81 chars | Starting grid, row by row; `0` is a blank cell |
| `solution` | string, 81 chars | Solved grid |
| `moves` | list of `[cell, digit]` pairs (`uint8`) | Every placement in the order it was made; `cell` is 0–80 (row × 9 + column), `digit` is 1–9 |
| `source` | string | How the record was produced, e.g. `manual` for puzzles solved by hand |

Intermediate boards are not stored. The board after step *t* is `puzzle` with the first *t* moves applied.

Every record satisfies:

- `solution` is a valid solved grid, and agrees with every given in `puzzle`.
- `moves` fills each blank cell of `puzzle` exactly once, with the digit `solution` has there.

`sudoku.data.write_records` checks these before writing and takes the `source` label for the file, and `read_records` loads a file back as `(puzzle, solution, moves)` arrays.

### Model tensors

`sudoku.dataset.SudokuDataset` turns a split into `(input, target, mask)` examples for predicting the whole solution in one pass:

| Tensor | Shape and type | Meaning |
|---|---|---|
| `input` | `int64[81]` | Board as tokens 0–9, `0` for blank (`one_hot=True` gives `float32[10, 9, 9]` instead) |
| `target` | `int64[81]` | Solution digit minus 1, so classes are 0–8 |
| `mask` | `bool[81]` | True where the input cell is blank; compute the loss on these cells only |

Options for training:

- `random_step=True` replays a random number of moves on each access, so the model sees a different partly solved board each epoch. Leave it off for evaluation to get the original puzzle.
- `augment=True` applies a random transform that keeps the puzzle valid: digit relabelling, row and column swaps within a band, band swaps, and transpose.

### Tests

```
pip install -r requirements.txt
python -m pytest
```
