"""REST API for solving sudoku puzzles with a trained model.

Run it from the repository root with:
    python -m server --checkpoint checkpoints/model.pt --port 8000

then open http://localhost:8000/docs for interactive documentation. The model is
loaded once at startup; set SUDOKU_CHECKPOINT and SUDOKU_DEVICE to choose the
checkpoint and device when running the app another way, e.g. `uvicorn server.app:app`.

Web pages on other origins may call it if their origin is allowed: by default
Vite's development server (http://localhost:5173), otherwise the comma-separated
list in --cors-origins or SUDOKU_CORS_ORIGINS.

Puzzles are an 81-character string ('0' or '.' for blanks) or a 9x9 list of ints
(0 for blanks), and results come back in the same form. A puzzle the model
can't solve is not an error: the response has "solved": false and the model's
attempt, which breaks the rules somewhere.
"""

import argparse
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sudoku.api import SudokuSolver, is_solution, parse

Puzzle = Union[str, list[list[int]]]
Method = Literal["one_at_a_time", "retries", "all_at_once"]
MAX_BATCH = 10_000
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"  # Vite's development server

_EXAMPLE = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"


class SolveRequest(BaseModel):
    puzzle: Puzzle = Field(examples=[_EXAMPLE])
    method: Method = "one_at_a_time"


class SolveResponse(BaseModel):
    solved: bool
    solution: Optional[Puzzle] = Field(None, description="The finished grid, if the model solved it")
    attempt: Optional[Puzzle] = Field(None, description="The model's grid, which breaks the rules, if it didn't")


class BatchRequest(BaseModel):
    puzzles: list[Puzzle] = Field(max_length=MAX_BATCH, examples=[[_EXAMPLE]])
    method: Method = "one_at_a_time"


class BatchResponse(BaseModel):
    results: list[SolveResponse]


class HintRequest(BaseModel):
    puzzle: Puzzle = Field(examples=[_EXAMPLE])


class MoveResponse(BaseModel):
    row: int = Field(description="0-8, top to bottom")
    col: int = Field(description="0-8, left to right")
    digit: int
    confidence: float


class HintResponse(BaseModel):
    move: Optional[MoveResponse] = Field(None, description="None if the puzzle is already full")


class CheckRequest(BaseModel):
    puzzle: Puzzle
    grid: Puzzle


class CheckResponse(BaseModel):
    valid: bool = Field(description="Filled, keeps every given, and breaks no rules")


def _result(puzzle, grid):
    if is_solution(puzzle, grid):
        return SolveResponse(solved=True, solution=grid)
    return SolveResponse(solved=False, attempt=grid)


def _check_puzzle(puzzle):
    try:
        parse(puzzle)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


def create_app(solver=None, cors_origins=None):
    """Build the app. Without `solver`, one is loaded from SUDOKU_CHECKPOINT at startup.

    `cors_origins` is a comma-separated list of web origins whose pages may call
    the API (default: SUDOKU_CORS_ORIGINS, or Vite's development server).
    """
    state = {"solver": solver}
    lock = threading.Lock()  # one model call at a time; requests run in a thread pool

    @asynccontextmanager
    async def lifespan(app):
        if state["solver"] is None:
            state["solver"] = SudokuSolver.from_checkpoint(
                os.environ.get("SUDOKU_CHECKPOINT", "checkpoints/model.pt"), os.environ.get("SUDOKU_DEVICE")
            )
        yield

    app = FastAPI(
        title="Sudoku solver",
        description="Solve sudoku puzzles with a trained transformer model.",
        lifespan=lifespan,
    )
    origins = cors_origins if cors_origins is not None else os.environ.get("SUDOKU_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health():
        solver = state["solver"]
        return {"status": "ok", "device": str(solver.device), "loops": getattr(solver.model, "loops", None)}

    @app.post("/solve", response_model=SolveResponse)
    def solve(request: SolveRequest):
        """Solve one puzzle."""
        _check_puzzle(request.puzzle)
        with lock:
            grid = state["solver"].solve_many([request.puzzle], method=request.method, strict=False)[0]
        return _result(request.puzzle, grid)

    @app.post("/solve/batch", response_model=BatchResponse)
    def solve_batch(request: BatchRequest):
        """Solve many puzzles together, much faster than one request each."""
        for i, puzzle in enumerate(request.puzzles):
            try:
                parse(puzzle)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"puzzle {i}: {e}") from None
        with lock:
            grids = state["solver"].solve_many(request.puzzles, method=request.method, strict=False)
        return BatchResponse(results=[_result(p, g) for p, g in zip(request.puzzles, grids)])

    @app.post("/hint", response_model=HintResponse)
    def hint(request: HintRequest):
        """The single move the model is most sure of."""
        _check_puzzle(request.puzzle)
        with lock:
            move = state["solver"].next_move(request.puzzle)
        return HintResponse(move=None if move is None else MoveResponse(**vars(move)))

    @app.post("/check", response_model=CheckResponse)
    def check(request: CheckRequest):
        """Whether `grid` is a correct solution of `puzzle`. Doesn't use the model."""
        _check_puzzle(request.puzzle)
        try:
            return CheckResponse(valid=is_solution(request.puzzle, request.grid))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"grid: {e}") from None

    return app


app = create_app()


def main(argv=None):
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the sudoku solver over HTTP.")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cors-origins",
        default=None,
        help=f"comma-separated web origins allowed to call the API, or * for any (default: {DEFAULT_CORS_ORIGINS})",
    )
    args = parser.parse_args(argv)
    os.environ["SUDOKU_CHECKPOINT"] = args.checkpoint
    if args.device:
        os.environ["SUDOKU_DEVICE"] = args.device
    uvicorn.run(create_app(cors_origins=args.cors_origins), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
