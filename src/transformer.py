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
    """
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


class MultiHeadAttentionSegment(nn.Module):
    """
        Module to perform multi-head attention.
        ((batch_size, sequence_length, model_dimensions) -> (batch_size, sequence_length, model_dimensions))

        This module implements multi-head attention, which allows the model to attend
        to information from different representation subspaces. By splitting the attention
        mechanism into multiple heads, the model can focus on different aspects of the input.

        The input queries, keys, and values are transformed into subspaces, scaled, and
        combined through attention weights computed using dot-product attention. The results
        are concatenated and passed through a final linear layer to produce the output.
        """
    def __init__(self, model_dimensions: int, head_count: int, dropout: float):
        """
        Initialize the MultiHeadAttentionSegment module.

        Args:
            model_dimensions (int): Total number of dimensions in the input embeddings.
            head_count (int): Number of attention heads.
            dropout (float): Dropout probability applied to the attention weights.
        """
        super().__init__()

        self.model_dimensions = model_dimensions
        self.head_count = head_count
        assert model_dimensions % head_count == 0, "ERROR!: dimension of model are not divisible by number of heads"

        # Compute the dimension of each attention head
        self.dimension_per_head = model_dimensions // head_count

        # Define linear transformations for query, key, value, and output projection
        self.w_q = nn.Linear(model_dimensions, model_dimensions)
        self.w_k = nn.Linear(model_dimensions, model_dimensions)
        self.w_v = nn.Linear(model_dimensions, model_dimensions)
        self.w_o = nn.Linear(model_dimensions, model_dimensions)

        self.dropout = nn.Dropout(dropout)

        # Placeholder for storing attention scores
        self.attention_scores = None

    @staticmethod
    def calculate_attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute scaled dot-product attention.

        Args:
            query (Tensor): Query tensor of shape (batch_size, head_count, seq_len, dim_per_head).
            key (Tensor): Key tensor of shape (batch_size, head_count, seq_len, dim_per_head).
            value (Tensor): Value tensor of shape (batch_size, head_count, seq_len, dim_per_head).
            mask (Tensor or None): Mask tensor of shape (batch_size, 1, seq_len, seq_len)
            dropout (nn.Dropout): Dropout layer applied to attention probabilities.

        Returns:
            Tensor: Weighted sum of values (attention output).
            Tensor: Attention scores (softmax probabilities).
        """
        dimension_per_head = query.shape[-1]

        key = key.transpose(-2, -1)  # Shape: [batch_size, head_count, dim_per_head, seq_len]
        attention_scores = (query @ key)  # Shape: [batch_size, head_count, seq_len, seq_len]

        attention_scores = attention_scores / math.sqrt(dimension_per_head)  # scaled attention scores

        if mask is not None:  # masked attention scores
            # if mask = 0 -> value = -infinity (around -2^31) so that softmax excludes them
            attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to convert attention scores into probabilities
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:  # needed since dropout gets handed over as nn.Dropout and not float
            attention_scores = dropout(attention_scores)

        # Compute the weighted sum of the value vectors using the dot product with the attention scores
        weighted_sum_of_values = attention_scores @ value  # weighted_sum_of_values
        return weighted_sum_of_values, attention_scores  # attention scores needed for visualization

    def split_heads(self, head_to_split):
        """
        Split a tensor into multiple heads and reorder dimensions for parallel attention computation.

        Args:
            head_to_split (Tensor): Input tensor of shape [batch_size, seq_len, model_dimensions].

        Returns:
            Tensor: Transformed tensor of shape [batch_size, head_count, seq_len, dim_per_head].
        """
        head_to_split = head_to_split.view(head_to_split.shape[0], head_to_split.shape[1],
                                           self.head_count, self.dimension_per_head)
        # Transpose to bring the head dimension forward
        head_to_split = head_to_split.transpose(1, 2)  # Shape: [batch_size, head_count, seq_len, dim_per_head]
        return head_to_split

    def forward(self, q, k, v, mask):
        """
        Forward pass of the multi-head attention mechanism.

        Args:
            q (Tensor): Query tensor of shape [batch_size, seq_len, model_dimensions].
            k (Tensor): Key tensor of shape [batch_size, seq_len, model_dimensions].
            v (Tensor): Value tensor of shape [batch_size, seq_len, model_dimensions].
            mask (Tensor or None): Mask tensor of shape [batch_size, 1, seq_len, seq_len].

        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, model_dimensions].
        """
        # Transform input embeddings into queries, keys, and values
        # Shape: [batch_size, seq_len, model_dimensions]
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split queries, keys, and values into multiple heads for parallel computation
        # and allowing focus on different aspects of the input.
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        attention_output, self.attention_scores = self.calculate_attention(query, key, value, mask, self.dropout)

        # Shape: [batch_size, seq_len, head_count * dim_per_head]
        attention_output = attention_output.transpose(1, 2)
        # Ensures that the tensor is stored in a contiguous block of memory(needed before reshaping)
        attention_output = attention_output.contiguous()

        # Reshape tensor back to original dimensions
        attention_output = attention_output.view(
            attention_output.shape[0], -1, self.head_count * self.dimension_per_head
        )

        # Apply the output projection and return
        return self.w_o(attention_output)


class LayerAdditionAndNormalization(nn.Module):
    """
    Module for applying layer normalization to an input tensor.
    (batch_size, sequence_length, features), (batch_size, sequence_length, features)

    Note:
        - Layer Normalization: Instead of normalizing across a batch, the normalization is done across features.
        For each individual sample the normalization is computed across all features.
        Another point, is that in Layer Normalization, zeros in an individual sample do not affect the normalization
        of other values in that sample, because the normalization process is done independently for each sample,
        and the presence of zeros doesn't cause a shift in the other values
        (other than their own relative contribution to the mean and variance).
    """
    def __init__(self, features: int, epsilon: float = 10e-6):
        """
        Initialize the LayerAdditionAndNormalization module.

        Args:
            features (int): The size of the feature dimension for the input tensor.
            epsilon (float): A small constant added to the denominator to prevent division by zero.
        """
        super().__init__()
        self.epsilon = epsilon  # Small constant to prevent division by zero during normalization
        self.gamma = nn.Parameter(torch.ones(features))  # creates a learnable parameter, which starts out as all ones
        self.bias = nn.Parameter(torch.zeros(features))  # creates a learnable parameter, which starts out as all zeros

    def forward(self, x):
        """
        Forward pass for layer normalization.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, features).
                represents a sequence or batch of data.

        Returns:
            Tensor: Output tensor of the same shape as input,
                after applying normalization and affine transformation.

        Note:
            - used the standard deviation instead of the root of the variance since I wasn't sure
            whether I should use a biased or unbiased variance, and the std is äquivalent the root of the std.
            And since epsilon is << the root of var+epsilon is not that far off the std + epsilon
        """
        # Calculate the mean of the input tensor along the feature dimension (dim=-1) and keeps the dimensions intact
        mean = x.mean(dim=-1, keepdim=True)
        standard_deviation = x.std(dim=-1, keepdim=True)  # calculates the standard deviation of x

        # Normalize the input tensor: (x - mean) / (std + epsilon)
        # Epsilon ensures numerical stability to prevent division by zero
        x_dash = (x - mean) / (standard_deviation + self.eps) + self.bias

        # Scale shift and return
        return self.gamma * x_dash + self.bias
