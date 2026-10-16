import torch
from torch import nn

_CELLS = torch.arange(81)
_ROWS = _CELLS // 9
_COLS = _CELLS % 9
_BOXES = (_ROWS // 3) * 3 + _COLS // 3


class SudokuTransformer(nn.Module):
    """Predicts a digit for every cell of a board in one pass.

    Input:  int64[B, 81] board tokens 0-9 (0 = blank), as from SudokuDataset.
    Output: float[B, 81, 9] logits; class k means digit k + 1.

    Each cell's vector is the sum of embeddings for its digit, row, column and
    box, so the model knows from the start which cells constrain each other.
    A transformer encoder then lets every cell attend to every other cell.
    """

    def __init__(self, d_model=128, n_layers=6, n_heads=4, dropout=0.1):
        super().__init__()
        self.digit = nn.Embedding(10, d_model)
        self.row = nn.Embedding(9, d_model)
        self.col = nn.Embedding(9, d_model)
        self.box = nn.Embedding(9, d_model)
        self.register_buffer("rows", _ROWS.clone(), persistent=False)
        self.register_buffer("cols", _COLS.clone(), persistent=False)
        self.register_buffer("boxes", _BOXES.clone(), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 9)

    def forward(self, board):
        x = self.digit(board) + self.row(self.rows) + self.col(self.cols) + self.box(self.boxes)
        return self.head(self.encoder(x))
