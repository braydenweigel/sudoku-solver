import pytest

pytest.importorskip("fastapi", reason="install server/requirements.txt to test the server")

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from server.app import MAX_BATCH, create_app  # noqa: E402
from sudoku.api import SudokuSolver  # noqa: E402
from sudoku.data import encode  # noqa: E402

PUZZLE = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


class Oracle(torch.nn.Module):
    """Predicts `answer` for every cell, most confident on the lowest cell index."""

    def __init__(self, answer):
        super().__init__()
        self.answer = torch.from_numpy(encode(answer).astype("int64"))

    def forward(self, board):
        logits = torch.nn.functional.one_hot(self.answer - 1, 9).float().expand(len(board), 81, 9).clone()
        return logits * (100 - torch.arange(81)).float()[None, :, None] / 10


def rows(grid):
    return [[int(c) for c in grid[r * 9 : r * 9 + 9]] for r in range(9)]


@pytest.fixture
def client():
    with TestClient(create_app(SudokuSolver(Oracle(SOLUTION), device="cpu"))) as c:
        yield c


@pytest.fixture
def failing_client():
    with TestClient(create_app(SudokuSolver(Oracle("1" * 81), device="cpu"))) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["device"] == "cpu"


def test_solve_string_and_grid(client):
    r = client.post("/solve", json={"puzzle": PUZZLE})
    assert r.status_code == 200
    assert r.json() == {"solved": True, "solution": SOLUTION, "attempt": None}
    r = client.post("/solve", json={"puzzle": rows(PUZZLE), "method": "all_at_once"})
    assert r.json()["solution"] == rows(SOLUTION)


def test_unsolved_returns_attempt(failing_client):
    body = failing_client.post("/solve", json={"puzzle": PUZZLE}).json()
    assert body["solved"] is False and body["solution"] is None
    assert len(body["attempt"]) == 81


@pytest.mark.parametrize("puzzle", [PUZZLE[:-1], "55" + PUZZLE[2:]])
def test_bad_puzzle_is_400(client, puzzle):
    r = client.post("/solve", json={"puzzle": puzzle})
    assert r.status_code == 400 and r.json()["detail"]


def test_bad_method_is_422(client):
    assert client.post("/solve", json={"puzzle": PUZZLE, "method": "guess"}).status_code == 422


def test_batch(client, failing_client):
    r = client.post("/solve/batch", json={"puzzles": [PUZZLE, rows(PUZZLE)]})
    assert [x["solution"] for x in r.json()["results"]] == [SOLUTION, rows(SOLUTION)]
    r = failing_client.post("/solve/batch", json={"puzzles": [PUZZLE]})
    assert r.json()["results"][0]["solved"] is False
    r = client.post("/solve/batch", json={"puzzles": [PUZZLE, "bad"]})
    assert r.status_code == 400 and r.json()["detail"].startswith("puzzle 1:")


def test_batch_size_limit(client):
    assert client.post("/solve/batch", json={"puzzles": [PUZZLE] * (MAX_BATCH + 1)}).status_code == 422


def test_hint(client):
    body = client.post("/hint", json={"puzzle": PUZZLE}).json()
    assert (body["move"]["row"], body["move"]["col"], body["move"]["digit"]) == (0, 2, int(SOLUTION[2]))
    assert client.post("/hint", json={"puzzle": SOLUTION}).json() == {"move": None}


def test_check(client):
    assert client.post("/check", json={"puzzle": PUZZLE, "grid": SOLUTION}).json() == {"valid": True}
    assert client.post("/check", json={"puzzle": PUZZLE, "grid": PUZZLE}).json() == {"valid": False}
    assert client.post("/check", json={"puzzle": PUZZLE, "grid": "123"}).status_code == 400


def preflight(client, origin):
    return client.options(
        "/solve",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )


def test_cors_allows_vite_dev_server_by_default(client, monkeypatch):
    r = preflight(client, "http://localhost:5173")
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    r = client.post("/solve", json={"puzzle": PUZZLE}, headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_other_origins(client):
    r = preflight(client, "http://evil.example")
    assert "access-control-allow-origin" not in r.headers


def test_cors_origins_are_configurable(monkeypatch):
    solver = SudokuSolver(Oracle(SOLUTION), device="cpu")
    with TestClient(create_app(solver, cors_origins="https://sudoku.example, http://localhost:3000")) as c:
        assert preflight(c, "https://sudoku.example").headers["access-control-allow-origin"] == "https://sudoku.example"
        assert "access-control-allow-origin" not in preflight(c, "http://localhost:5173").headers
    monkeypatch.setenv("SUDOKU_CORS_ORIGINS", "*")
    with TestClient(create_app(solver)) as c:
        assert preflight(c, "http://anything.example").headers["access-control-allow-origin"] == "*"
