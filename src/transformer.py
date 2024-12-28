import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, dimensions: int, vocab_size: int):
        super().__init__()
        self.dimensions = dimensions
        self.vocabSize = vocab_size
        self.embedding = nn.Embedding(vocab_size, dimensions)  # method from pytorch for embedding

    def forward(self, pos):
        return self.embedding(pos) * math.sqrt(self.dimensions)  # scaling
        # scaling is crucial to prevent the values of getting to larger -> ensure stable gradients during training


class PositionalEncodings(nn.Module):
    def __init__(self, dimensions: int, max_sequence_length: int, dropout: float):
        super().__init__()
        self.dimensions = dimensions
        self.max_sequence_length = max_sequence_length
        self.dropout = nn.Dropout(dropout)

        # create positional Encodings
        positional_encodings = torch.zeros(max_sequence_length, dimensions)  # 2D tensor (len, dimensions)
        position = torch.arange(0, max_sequence_length, 1, dtype=torch.float).unsqueeze(1) # (maxSequenceLength, 1)

        # Compute the divisor as 10^(4^(2i/d))
        i = torch.arange(0, dimensions, 2, dtype=torch.float)  # Indices for even dimensions
        divisor = torch.pow(10_000, 2 * i / dimensions)  # 10^(4^(2i/d))
        # TODO: Use numerically stable logarithmic formula to avoid large number problems
        x = position / divisor

        positional_encodings[:, 0::2] = torch.sin(x)
        positional_encodings[:, 1::2] = torch.cos(x)

        positional_encodings = positional_encodings.unsqueeze(0)  # Add a batch dimension to the positional encoding
        self.register_buffer("positional_encodings", positional_encodings)
        # adds a new parameter self.positional_encodings which isn't learned but saved and loaded in training

    def forward(self, x):
        # x = x + positional_encodings of x
        x = x + (self.positional_encodings[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, dimensions)
        return self.dropout(x)
