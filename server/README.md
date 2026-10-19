# REST API

This folder holds the REST API: the app in `app.py` and its tests in `tests/`. It uses the model through the [Python API](../README.md#using-the-model).

You need a trained checkpoint (see [Training](../README.md#training) in the main README). Install the server's packages, which include the main ones, then run commands from the repository root:

```
pip install -r server/requirements.txt
```

The server offers the same functions over HTTP with FastAPI. The model is loaded once at startup:

```
python -m server --checkpoint checkpoints/model.pt --port 8000
```

Interactive documentation, where you can try each endpoint, is at http://localhost:8000/docs.

| Endpoint | Body | Response |
|---|---|---|
| `GET /health` | | `{"status": "ok", "device": "cuda", "loops": 4}` |
| `POST /solve` | `{"puzzle": ..., "method": "one_at_a_time"}` | `{"solved": true, "solution": ...}`, or `{"solved": false, "attempt": ...}` if the model's grid breaks the rules |
| `POST /solve/batch` | `{"puzzles": [...], "method": ...}` (up to 10,000) | `{"results": [...]}`, one `/solve` response per puzzle |
| `POST /hint` | `{"puzzle": ...}` | `{"move": {"row": 5, "col": 5, "digit": 4, "confidence": 0.99}}`, or `{"move": null}` if the puzzle is full |
| `POST /check` | `{"puzzle": ..., "grid": ...}` | `{"valid": true}` if `grid` solves `puzzle`; doesn't use the model |

Puzzles and grids are the same forms as for the [Python API](../README.md#using-the-model), and results come back in the form sent. A malformed puzzle, or one that already repeats a digit, gets HTTP 400 with the reason in `detail`. A puzzle the model can't solve is not an error: it gets `"solved": false`. For many puzzles, use `/solve/batch`, which solves them together on the GPU (500 hard puzzles take about 8 seconds), rather than one request each.

To run it another way, such as `uvicorn server.app:app`, set `SUDOKU_CHECKPOINT` (and optionally `SUDOKU_DEVICE`) instead of passing options. By default it only accepts connections from this computer; use `--host 0.0.0.0` to allow others on your network.
