"""Paper-inspired BiLSTM classifier with multi-head attention."""

from __future__ import annotations

import torch
from torch import nn


class GlypredBiLSTMAttention(nn.Module):
    """BiLSTM + LayerNorm + attention + GELU + dropout + residual classifier.

    The architecture follows the classifier components described for Glypred.
    Input tensors have shape ``[batch, 31, 21]`` and output one logit per
    sequence.
    """

    def __init__(self, input_size: int = 21, hidden_size: int = 64, heads: int = 4, dropout: float = 0.5):
        super().__init__()
        embedding_size = hidden_size * 2
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm_norm = nn.LayerNorm(embedding_size)
        self.attention = nn.MultiheadAttention(embedding_size, heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(embedding_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_output, _ = self.lstm(x)
        lstm_output = self.lstm_norm(lstm_output)
        attention_output, _ = self.attention(lstm_output, lstm_output, lstm_output, need_weights=False)
        contextual = self.attention_norm(lstm_output + attention_output)
        pooled = contextual.mean(dim=1)
        residual = pooled
        hidden = self.fc1(self.dropout(self.gelu(pooled)))
        hidden = self.gelu(hidden + residual)
        return self.fc2(self.dropout(hidden)).squeeze(1)
