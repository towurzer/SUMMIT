import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, dimensions: int, vocabSize: int, *args, **kwargs):
        super().__init__()
        self.dimensions = dimensions
        self.vocabSize = vocabSize
        self.embedding = nn.Embedding(vocabSize, dimensions)  # method from pytorch for embedding

    def forward(self, pos):
        return self.embedding(pos) * math.sqrt(self.dimensions)  # scaling
        # scaling is crucial to prevent the values of getting to larger -> ensure stable gradients during training


class PositionalEncodings(nn.Module):
    def __init__(self, dimensions: int, maxSequenceLength: int, dropout: float):
        super().__init__()
        self.dimensions = dimensions
        self.maxSequenceLength = maxSequenceLength
        self.dropout = nn.Dropout(dropout)

        # create positional Encodings
        positionalEncodings = torch.zeros(maxSequenceLength, dimensions)  # 2D tensor (len, dimensions)
        position = torch.arange(0, maxSequenceLength, 1, dtype=torch.float).unsqueeze(1) # (maxSequenceLength, 1)

        # Compute the divisor as 10^(4^(2i/d))
        i = torch.arange(0, dimensions, 2, dtype=torch.float)  # Indices for even dimensions
        divisor = torch.pow(10_000, 2 * i / dimensions)  # 10^(4^(2i/d))
        # TODO: Use numerically stable logarithmic formula to avoid large number problems
        x = position / divisor

        positionalEncodings[:, 0::2] = torch.sin(x)
        positionalEncodings[:, 1::2] = torch.cos(x)

        positionalEncodings = positionalEncodings.unsqueeze(0)  # Add a batch dimension to the positional encoding
        self.register_buffer("positionalEncodings", positionalEncodings)
        # adds a new parameter self.positionalEncodings which isn't learned but saved and loaded in training

    def forward(self, x):
        # x = x + positionalEncoding of x
        x = x + (self.positionalEncodings[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, dimensions)
        return self.dropout(x)
