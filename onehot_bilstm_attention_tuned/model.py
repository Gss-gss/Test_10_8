"""Glypred-style BiLSTM attention classifier for one-hot protein windows."""

from __future__ import annotations

import torch
from torch import nn


class GlypredBiLSTMAttention(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, input_size: int = 21, heads: int = 4):
        super().__init__()
        embedding_size = hidden_size * 2
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm_norm = nn.LayerNorm(embedding_size)
        self.attention = nn.MultiheadAttention(embedding_size, heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(embedding_size)
        self.fc1 = nn.Linear(embedding_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        sequence = self.lstm_norm(sequence)
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        sequence = self.attention_norm(sequence + attended)
        pooled = sequence.mean(dim=1)
        hidden = self.fc1(self.dropout(self.gelu(pooled)))
        hidden = self.gelu(hidden + pooled)
        return self.fc2(self.dropout(hidden)).squeeze(1)
