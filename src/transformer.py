import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    """
    Module to create (scaled) InputEmbeddings
    ((batch_size, sequence_length) -> (batch_size, sequence_length, model_dimensions))

    This module creates embeddings for input tokens and scales them by the square root of the model dimensions
    to stabilize training

    Example:
        Given:
            input_tokens = [[1, 2, 3], [4, 5, 6]]  # shape: (2, 3)
            model_dimensions = 4
            vocab_size = 10

        The output will be a tensor of shape (2, 3, 4):
            tensor([[[ 0.25, -0.31,  0.13,  0.08],
                 [ 0.19,  0.01, -0.45,  0.27],
                 [ 0.05, -0.12,  0.09,  0.41]],
                [[-0.34,  0.22, -0.18,  0.04],
                 [ 0.11, -0.08,  0.39, -0.27],
                 [-0.20,  0.14,  0.07, -0.03]]]) * sqrt(4)
    """
    def __init__(self, model_dimensions: int, vocab_size: int):
        """
        Initialize the InputEmbeddings module.

        Args:
            model_dimensions (int): Dimensionality of the embeddings.
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__()
        self.model_dimensions = model_dimensions
        self.vocab_size = vocab_size

        # Pytorch Method - maps input tokens to vector representations (for embeddings)
        self.embedding = nn.Embedding(vocab_size, model_dimensions)

    def forward(self, input_tokens):
        """
        Forward pass to compute scaled embeddings.

        Args:
            input_tokens (Tensor): tokens of shape (batch_size, sequence_length).
            (Each token index corresponds to a vocabulary entry.)

        Returns:
            Tensor: Scaled embeddings of shape (batch_size, sequence_length, model_dimensions).
            (where model_dimensions is the size of the embedding vectors.)

        Note:
            - The embeddings are scaled by sqrt(model_dimensions) to normalize their
              magnitudes. This is to ensure stable gradients and balanced contributions
              of embeddings and positional encodings.
        """
        return self.embedding(input_tokens) * math.sqrt(self.model_dimensions)


class PositionalEncodings(nn.Module):
    """
    Module to generate and add positional encodings to input embeddings in a sequence.
    ((batch_size, sequence_length, model_dimensions) -> (batch_size, sequence_length, model_dimensions))

    This module computes positional encodings for each position in the input sequence using trigonometric functions
    since the paper this model is based upon hypothesised that this is a great way. "Trigonometric functions like
    cos and sin naturally represent a pattern that the model can recognize as continuous, so relative positions are
    easier to see for the model. By watching the plot of these functions, we can also see a regular pattern,
    so we can hope that the model will see it too."

    The goal is to inject positional information into the model, as Transformer models do not
    inherently understand token order since it processes all the tokens parallely.

    Example:
    Given:
        model_dimensions = 2
        max_sequence_length  = 1
        dropout = 0.0
        input_embeddings = [[0.1, 0.2]]  # shape: (1, 1, 2)
        (positional_encodings = [[0.01, 0.02]]  # shape: (1, 1, 2)) <- e.g. would be calculated in __init__

    The output will be a tensor of shape (1, 1, 2):
        tensor([[[0.11, 0.24]]])  # (0.1 + 0.01, 0.2 + 0.02)
    """
    # TODO: Add Example
    def __init__(self, model_dimensions: int, max_sequence_length: int, dropout: float):
        """
        Initialize the PositionalEncodings module.

        Args:
            model_dimensions (int): Dimensionality of the model (embedding size).
            max_sequence_length (int): Maximum sequence length for which positional encodings are generated.
            dropout (float): Dropout probability applied to the output.

        """
        super().__init__()

        # Initialize model dimensions, max sequence length, and dropout
        self.model_dimensions = model_dimensions
        self.max_sequence_length = max_sequence_length
        self.dropout = nn.Dropout(dropout)

        # Create a tensor to store the positional encodings (shape: max_sequence_length, model_dimensions)
        positional_encodings = torch.zeros(max_sequence_length, model_dimensions)  # 2D tensor (len, model_dimensions)
        # Generate positions (0, 1, 2, ..., max_sequence_length-1)
        position = torch.arange(0, max_sequence_length, 1, dtype=torch.float).unsqueeze(1)  # (maxSequenceLength, 1)

        # Compute divisors for sine and cosine functions (scaled by 10e4^(2i/d))
        i = torch.arange(0, model_dimensions, 2, dtype=torch.float)  # Indices for even model_dimensions
        divisor = torch.pow(10_000, 2 * i / model_dimensions)  # 10^(4^(2i/d))
        # TODO: Use numerically stable logarithmic formula to avoid large number problems
        x = position / divisor  # Shape: (max_sequence_length, model_dimensions/2)

        positional_encodings[:, 0::2] = torch.sin(x)  # Apply sine to even dimensions
        positional_encodings[:, 1::2] = torch.cos(x)  # Apply cosine to odd dimensions

        # Add batch dimension to the positional encodings (shape: 1, max_sequence_length, model_dimensions)
        positional_encodings = positional_encodings.unsqueeze(0)

        # Register the positional encodings as a buffer (not a learnable parameter)
        self.register_buffer("positional_encodings", positional_encodings)

    def forward(self, input_embeddings):
        """
        Forward pass to add positional encodings to the input sequence.

        Args:
            input_embeddings (Tensor): Input tensor of shape (batch_size, seq_length, model_dimensions).
            The tensor typically represents the input embeddings.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, model_dimensions),
                after adding the positional encodings and applying dropout.

        Note:
            - The positional encodings are added element-wise to the input tensor.
        """
        # Extract the positional encodings corresponding to the current input sequence length
        positional_encodings_for_input = self.positional_encodings[:, :input_embeddings.shape[1], :]

        # The positional encodings do not require gradients as they are fixed
        positional_encodings_for_input = positional_encodings_for_input.requires_grad_(False)

        # Add the positional encodings to the input embeddings
        input_embeddings = input_embeddings + positional_encodings_for_input

        # Apply dropout
        return self.dropout(input_embeddings)
