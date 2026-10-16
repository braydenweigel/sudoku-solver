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

### Generating records

`sudoku.augment` makes new records from existing ones by applying random operations that keep a puzzle valid. Each operation changes the puzzle, solution and moves together, and keeps the move order, so a generated record is still a logically ordered solve.

To generate records, run this from the repository root, with the dependencies installed (see [Tests](#tests)):

```
python -m sudoku.augment --count 100000 --seed 0
```

This reads `data/manual.parquet` and writes `data/generated.parquet`, overwriting it if it exists. Generating 100,000 records takes about two minutes.

| Option | Default | Meaning |
|---|---|---|
| `--count` | required | Number of records to generate |
| `--input` | `data/manual.parquet` | Records to use as bases; the tool cycles through them in turn |
| `--output` | `data/generated.parquet` | File to write |
| `--min-ops`, `--max-ops` | 100, 150 | Each record gets a random number of operations in this range |
| `--seed` | random | Fixes the random choices, so the same seed and input give the same file |
| `--source` | `generated` | Value written to the `source` column |

Every generated puzzle is distinct from the inputs and from each other. Generated files are ignored by git, so use `--seed` to be able to recreate the same file.

**Operations.** Each record gets operations picked at random from this list, with repeats:

- relabel the digits 1–9 with a random permutation
- swap two rows within a band (rows 0–2, 3–5 or 6–8), or two columns within a stack
- swap two bands, or two stacks
- rotate 90° clockwise, transpose, or mirror left to right or top to bottom

Each operation is a small change, so it takes about 100 of them before a record stops resembling its base puzzle; fewer leave records close to the originals.

**Keeping test data separate.** Every generated record is a disguised copy of one base puzzle. To measure performance on puzzles the model has never seen, split the base puzzles first, then generate from each split separately:

```
python -m sudoku.split --test-count 2 --seed 0
python -m sudoku.augment --input data/manual_train.parquet --output data/generated_train.parquet --count 80000 --seed 0
python -m sudoku.augment --input data/manual_test.parquet --output data/generated_test.parquet --count 20000 --seed 1
```

`sudoku.split` picks `--test-count` puzzles at random from `data/manual.parquet` and writes them to `data/manual_test.parquet`, with the rest in `data/manual_train.parquet`. `data/manual.parquet` stays the full set: after adding hand-solved puzzles to it, run the split again. A new split can move puzzles between train and test, so regenerate both generated files afterwards.

**Adding an operation.** Write a function that takes a NumPy random generator and returns a cell permutation and a digit relabelling, then add it to `OPERATIONS` in `sudoku/augment.py`. The tests check every operation in that list automatically.

## Training

`sudoku.train` trains `sudoku.model.SudokuTransformer`, which reads a board and predicts a digit for all 81 cells at once. After generating `data/generated_train.parquet` and `data/generated_test.parquet` (see above):

```
python -m sudoku.train --epochs 10
```

Each epoch prints the training loss and, on the test file's original puzzles, the loss, the share of blank cells predicted correctly (`cell acc`) and the share of puzzles with every cell right (`puzzle acc`). Only blank cells count towards the loss and accuracy. The model with the lowest test loss is saved to `checkpoints/model.pt`, which git ignores; load it with `sudoku.train.load_model`.

It uses the GPU when PyTorch can see one (see [Tests](#tests) for installing the GPU build), with bfloat16 mixed precision. On an RTX 5070 an epoch of 80,000 records takes about 7 seconds.

Useful options:

- `--limit N` trains on the first N records only, for quick runs. The test file is still used in full.
- `--d-model`, `--layers`, `--heads`, `--dropout` set the model size (defaults 128, 6, 4, 0.1).
- `--batch-size`, `--lr`, `--weight-decay` set the optimiser (defaults 256, 3e-4, 0.01). The learning rate warms up, then decays to zero by the last epoch.
- `--augment` also applies random symmetries to each training example as it is loaded.

### Comparing solving methods

`sudoku.evaluate` runs a trained model on the original puzzles in one or more files and reports cell and puzzle accuracy for each way of filling the board (`sudoku/solve.py`):

- **all at once**: one run of the model fills every blank cell.
- **one cell at a time**: fill only the blank cell the model is most confident about, then run it again on the updated board, until the grid is full.

```
python -m sudoku.evaluate --data data/generated_test.parquet data/generated_train.parquet
```

It uses the first 2,000 puzzles of each file by default (`--limit`).

## Tests

```
pip install -r requirements.txt
python -m pytest
```

`pip install -r requirements.txt` installs the CPU-only build of PyTorch. To train on an NVIDIA GPU, replace it with a CUDA build afterwards (CUDA 13.0 here, which supports RTX 50-series cards):

```
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available())"
```
