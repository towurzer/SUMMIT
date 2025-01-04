import math
from abc import ABC, abstractmethod

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

		# divisor = torch.pow(10_000, 2 * i / model_dimensions)  # 10^(4^(2i/d))
		# Use numerically stable logarithmic formula to avoid large number problems instead
		log_divisor = torch.log(torch.tensor(10_000)) * (2 * i / model_dimensions)
		divisor = torch.exp(log_divisor)

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
			input_embeddings (Tensor): Input tensor of shape (batch_size, sequence_length, model_dimensions).
			The tensor typically represents the input embeddings.

		Returns:
			Tensor: Output tensor of shape (batch_size, sequence_length, model_dimensions),
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
		are concatenated and transformed back using another matrix multiplication with the weight Matrix W_o,
		to get a matrix back, that has the same format as the one given as an input
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

		Note:
			Note:
			- Masking is mostly applied in the decoder, where it ensures that a token can only attend to itself
			  and previous tokens during generation.
			- In the encoder, masking is generally limited to padding masks to handle variable-length input sequences.
			- for further explanation of mask look in the encoder and decoder block - docstring
		"""
		dimension_per_head = query.shape[-1]

		key = key.transpose(-2, -1)  # Shape: [batch_size, head_count, dim_per_head, seq_len]
		attention_scores = (query @ key)  # Shape: [batch_size, head_count, seq_len, seq_len]

		attention_scores = attention_scores / math.sqrt(dimension_per_head)  # scaled attention scores

		if mask is not None:
			# if mask = 0 -> value = -infinity (around -2^31) so that softmax excludes them
			attention_scores.masked_fill_(mask == 0, -1e9)

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


class LayerNormalization(nn.Module):
	"""
	Module for applying layer normalization to an input tensor.
	Used in Add and Norm Module
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
			x (Tensor): Input tensor of shape (batch_size, sequence_length, features).
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
		x_dash = (x - mean) / (standard_deviation + self.epsilon)

		# Scale shift and return
		return self.gamma * x_dash + self.bias


class FeedForwardLayer(nn.Module):
	"""
	FeedForwardLayer module
	((batch_size, sequence_length, model_dimensions) -> (batch_size, sequence_length, model_dimensions))

	This module applies two linear transformations separated by a ReLU activation
	and a dropout layer. It serves as a position-wise feed-forward network
	to process individual token representations independently.

	This Layer is used in addition to a self attention block.
	While the latter captures relationships between tokens, the feed-forward
	layer processes each token independently. This adds non-linearity and depth to the model,
	enabling it to extract more information from the input.
	"""

	def __init__(self, dimensions_model: int, dimensions_feed_forward_layer: int, dropout: float):
		"""
		Initialize the FeedForwardLayer module.

		Args:
			dimensions_model (int): Dimensionality of the input and output features.
			dimensions_feed_forward_layer (int): Dimensionality of the intermediate feed-forward layer.
			dropout (float): Dropout probability applied after the activation.
		"""
		super().__init__()
		# removed to save memory
		# self.dimensions_model = dimensions_model
		# self.dimensions_feed_forward_layer = dimensions_feed_forward_layer

		# First linear transformation: maps input features to feed-forward dimension
		self.linear_1 = nn.Linear(dimensions_model, dimensions_feed_forward_layer)  # W1 and B1 in Paper
		# Second linear transformation: maps back to original model dimension
		self.linear_2 = nn.Linear(dimensions_feed_forward_layer, dimensions_model)  # W2 and B2 in Paper

		self.dropout = nn.Dropout(dropout)

	def forward(self, input_tensor):
		"""
		Forward pass to process input through the feed-forward layer.

		Args:
			input_tensor (Tensor): Input tensor of shape (batch_size, sequence_length, model_dimensions).

		Returns:
			Tensor: Output tensor of the same shape as the input tensor (batch_size, sequence_length, model_dimensions).
		"""
		input_tensor = self.linear_1(input_tensor)
		input_tensor = torch.relu(input_tensor)
		input_tensor = self.dropout(input_tensor)
		return self.linear_2(input_tensor)


class AddAndNormLayer(nn.Module):
	"""
	Addition and Normalization Layer using a residual connection by skipping the sublayer computation,
	adding its output back to the original input, and applying layer normalization.
	((batch_size, sequence_length, features) -> (batch_size, sequence_length, features))

	Residual connections allow for the effective training of much deeper networks by addressing
	the gradient flow issues and making it easier for the network to learn identity mappings.
	This enables the model to be very expressiv and capable of learning more complex functions.
	"""

	def __init__(self, features: int, dropout: float):
		"""
		Initialize the AddAndNormLayer module.

		Args:
			features (int): Dimensionality of the input and output features.
			dropout (float): Dropout probability applied to the sublayer output.
		"""
		super().__init__()
		self.dropout = nn.Dropout(dropout)

		self.normalization_layer = LayerNormalization(features)

	def forward(self, residual, sublayer_function):
		"""
		Forward pass to apply residual connection and normalization.

		Args:
			residual (Tensor): Input tensor (residual connection) of shape
								(batch_size, sequence_length, features).
			sublayer_function (Callable): A function representing the sublayer
											(e.g., attention or feed-forward).

		Returns:
			Tensor: Output tensor of shape (batch_size, sequence_length, features).
		"""
		sublayer_output = sublayer_function(residual)  # compute sublayer
		sublayer_output = self.dropout(sublayer_output)
		residual_added_output = residual + sublayer_output  # add ...
		return self.normalization_layer(residual_added_output)  # ... & norm


class EncoderBlock(nn.Module):
	"""
	   EncoderBlock module
	   ((batch_size, sequence_length, features) -> (batch_size, sequence_length, features))

	   An Encoder Block consists of two sub-layers:
	   1. A multi-head self-attention mechanism.
	   2. A position-wise feed-forward network.

	   Each sublayer is wrapped with an addition and normalization block,
	   addressing vanishing gradients and stabilize training.
	   """

	def __init__(self, model_dimensions: int, self_attention_layer: MultiHeadAttentionSegment,
				 feed_forward_block: FeedForwardLayer, dropout: float):
		"""
		Initialize the EncoderBlock module.

		Args:
			model_dimensions (int): Dimensionality of the input and output features.
			self_attention_layer (MultiHeadAttentionSegment): Multi-head self-attention mechanism.
			feed_forward_block (FeedForwardLayer): Position-wise feed-forward network.
			dropout (float): Dropout probability applied in the AddAndNormLayer.
		"""
		super().__init__()

		# Multi-head self-attention and feed-forward layers
		self.self_attention_layer = self_attention_layer
		self.feed_forward_block = feed_forward_block

		# Add and Normalize layers for each sublayer
		self.add_and_norm_layers = nn.ModuleList(
			[AddAndNormLayer(model_dimensions, dropout) for _ in range(2)]
		)

	def forward(self, residual_input, attention_mask):
		"""
		Forward pass through the EncoderBlock.

		Args:
			residual_input (Tensor): Input tensor of shape (batch_size, sequence_length, features).
			attention_mask (Tensor): attention mask to prevent attention to padding tokens.

		Returns:
			Tensor: Output tensor of shape (batch_size, sequence_length, features).

		Note:
			- In the self-attention mechanism, the same tensor is used for the query, key and value inputs.
			  This is done to allow each token to attend to all other tokens in teh sequence including itself.
			  Using the same tensor for all three inputs helps compute the attention weights and output based
			  on the relationships between the tokens in the sequence,
			  while preserving the structure of the input sequence.

			- The mask ensures that the attention mechanism ignores padding tokens,
			  preventing the model from attending to irrelevant or padded positions in the input.
		"""

		# the first addition and normalization layer adds the output of the self attention block and the input of it
		residual_input = self.add_and_norm_layers[0](
			residual_input,
			lambda attention_input:  # attention_input is used as query key and value, as explained in the docs-note.
			self.self_attention_layer(attention_input, attention_input, attention_input, attention_mask)
		)

		# the second addition and normalization layer adds
		# the output of the first add and norm layer with the output of the feed forward layer
		residual_input = self.add_and_norm_layers[1](residual_input, self.feed_forward_block)
		return residual_input


class Encoder(nn.Module):
	"""
	Encoder module consisting of multiple encoder blocks.
	((batch_size, sequence_length, features), (batch_size, sequence_length, features))

	The Encoder processes the input tensor through a series of encoder blocks,
	each containing self-attention and feed-forward sub-layers.
	"""

	def __init__(self, model_dimensions: int, number_of_encoder_and_decoder_blocks: int,
				 number_of_heads_in_multi_head_attention: int, feed_forward_hidden_layer_dimensions: int,
				 dropout: float):
		"""
		Initialize the Encoder module.

		Args:
			model_dimensions (int): Dimensionality of model layers.
			number_of_encoder_and_decoder_blocks (int): Number of encoder/decoder blocks.
			number_of_heads_in_multi_head_attention (int): Number of attention heads.
			dropout (float): Dropout probability.
			feed_forward_hidden_layer_dimensions (int): Hidden layer size in feed-forward sublayer.
		"""
		super().__init__()

		# create the encoder_blocs and save them in the encoder_module_list
		# Each encoder Block consists of a self attention layer, and a feed forward layer
		encoder_blocks = []
		for _ in range(number_of_encoder_and_decoder_blocks):
			encoder_self_attention_layer = (
				MultiHeadAttentionSegment(model_dimensions, number_of_heads_in_multi_head_attention, dropout)
			)
			feed_forward_layer = FeedForwardLayer(model_dimensions, feed_forward_hidden_layer_dimensions, dropout)
			encoder_block = EncoderBlock(model_dimensions, encoder_self_attention_layer, feed_forward_layer, dropout)
			encoder_blocks.append(encoder_block)

		self.encoder_module_list = nn.ModuleList(encoder_blocks)

	def forward(self, encoded_input, mask):
		"""
		Forward pass through the Encoder.

		Args:
			encoded_input (Tensor): Input tensor that gets encoded,
			which has a shape (batch_size, sequence_length, features).
			mask (Tensor): Attention mask to prevent attending to padding tokens.
			Besides that the masking mechanism is only needed for the decoder

		Returns:
			Tensor: Output tensor of shape (batch_size, sequence_length, features) after passing through all encoders.
		"""
		# Pass input tensor through each encoder block
		for encoder in self.encoder_module_list:
			encoded_input = encoder(encoded_input, mask)
		return encoded_input


class DecoderBlock(nn.Module):
	"""
	DecoderBlock module
	((batch_size, sequence_length, features) -> (batch_size, sequence_length, features))

	A Decoder Block consisting of three main components:
	1. Self-Attention Layer: Allows the decoder to focus on earlier tokens in the output sequence.
	2. Cross-Attention Layer: Enables the decoder to attend to the encoder's output.
	3. Feed-Forward Layer: Processes each token representation independently for richer features.

	Each sublayer is wrapped with an AddAndNormLayer to stabilize the gradient and therefore the training
	"""

	def __init__(self,
				 model_dimensions: int,
				 self_attention_layer: MultiHeadAttentionSegment,
				 cross_attention_layer: MultiHeadAttentionSegment,
				 feed_forward_layer: FeedForwardLayer,
				 dropout: float
				 ):
		"""
		Initialize the DecoderBlock.

		Args:
			self_attention_layer (MultiHeadAttentionSegment): Self-attention mechanism for decoder.
			cross_attention_layer (MultiHeadAttentionSegment): Cross-attention mechanism with encoder output.
			feed_forward_layer (FeedForwardLayer): Position-wise feed-forward network.
			model_dimensions (int): Dimensionality of the input and output features of the Feed Forward Layer.
			dropout (float): Dropout probability for regularization.
		"""
		super().__init__()

		self.self_attention_layer = self_attention_layer
		self.cross_attention_layer = cross_attention_layer
		self.feed_forward_layer = feed_forward_layer

		# Add and Normalize layers for self-attention, cross-attention, and feed-forward sublayer
		self.add_and_norm_layers = nn.ModuleList(
			[AddAndNormLayer(model_dimensions, dropout) for _ in range(3)]
		)

	def forward(self, residual_input, encoder_output, encoder_mask, decoder_mask):
		"""
		Forward pass through the DecoderBlock.

		Args:
			residual_input (Tensor): Input tensor from the previous decoder layer or initial embedding
									 of shape (batch_size, sequence_length, features).
			encoder_output (Tensor): Output tensor from the encoder of shape (batch_size, src_sequence_length, features)
			encoder_mask (Tensor): Attention mask for the encoder output, preventing attention to padding tokens.
			decoder_mask (Tensor): Attention mask for the decoder input, preventing attention to padding tokens
								   and future tokens in the sequence (causal masking).

		Returns:
			Tensor: Output tensor of shape (batch_size, sequence_length, features).

		Note:
			- The cross attention layer adds the decoder's understanding with the encoder's contextual information.
			  The decoder's output serves as the query because it represents the current decoding state, while the
			  encoder's output which serves as key and values to provide the contextual information.
			- The encoder mask ensures that the decoder focuses only on non-padded positions in the encoder output
		"""
		# the first addition and normalization layer adds the output of the self attention block and the input of it
		# Self-Attention: Focuses on the decoder's own input tokens
		residual_input = self.add_and_norm_layers[0](
			residual_input,
			lambda attention_input:  # Query, Key, and Value are all the decoder's input (self attention)
			self.self_attention_layer(attention_input, attention_input, attention_input, decoder_mask)
		)

		# Cross-Attention: Attends to the encoder's output, aligning decoder input with encoder context
		residual_input = self.add_and_norm_layers[1](
			residual_input,
			lambda attention_input:  # Decoder input as Query, encoder output as Key and Value (explanation in note)
			self.cross_attention_layer(attention_input, encoder_output, encoder_output, encoder_mask)
		)

		# the third add&norm layer adds output of feed forward layer and output of cross attention layer
		residual_input = self.add_and_norm_layers[2](residual_input, self.feed_forward_layer)

		return residual_input


class Decoder(nn.Module):
	"""
	Decoder module consisting of multiple decoder blocks.
	((batch_size, sequence_length, features), (batch_size, sequence_length, features))

	The Decoder processes the input tensor through a series of decoder blocks,
	each containing self-attention, cross_attention and feed-forward sub-layers.
	"""
	def __init__(self, model_dimensions: int, number_of_encoder_and_decoder_blocks: int,
				 number_of_heads_in_multi_head_attention: int, feed_forward_hidden_layer_dimensions: int,
				 dropout: float):
		"""
		Initialize the Decoder module.

		Args:
			model_dimensions (int): Dimensionality of model layers.
			number_of_encoder_and_decoder_blocks (int): Number of encoder/decoder blocks.
			number_of_heads_in_multi_head_attention int: Number of attention heads.
			dropout (float): Dropout probability.
			feed_forward_hidden_layer_dimensions (int): Hidden layer size in feed-forward sublayer.
		"""
		super().__init__()

		# create the decoder_blocks and save them in the decoder_module_list
		# Each decoder Block consists of a self attention layer, a cross attention layer and a feed forward layer
		decoder_blocks = []
		for _ in range(number_of_encoder_and_decoder_blocks):
			decoder_self_attention_layer = (
				MultiHeadAttentionSegment(model_dimensions, number_of_heads_in_multi_head_attention, dropout)
			)
			decoder_cross_attention_layer = (
				MultiHeadAttentionSegment(model_dimensions, number_of_heads_in_multi_head_attention, dropout)
			)
			feed_forward_layer = FeedForwardLayer(model_dimensions, feed_forward_hidden_layer_dimensions, dropout)
			decoder_block = DecoderBlock(model_dimensions, decoder_self_attention_layer,
										 decoder_cross_attention_layer, feed_forward_layer, dropout)
			decoder_blocks.append(decoder_block)

		self.decoder_module_list = nn.ModuleList(decoder_blocks)

	def forward(self, decoder_input, encoder_output, encoder_mask, decoder_mask):
		"""
		Forward pass through the Decoder.

		Args:
			decoder_input (Tensor): Input tensor from the previous decoder layer or the initial embeddings.
									Shape: (batch_size, tgt_sequence_length, features).
			encoder_output (Tensor): Output tensor from the encoder, used as the key and value for cross-attention.
									 Shape: (batch_size, src_sequence_length, features).
			encoder_mask (Tensor): Mask to prevent the decoder from attending to padding tokens in the encoder output.
			decoder_mask (Tensor): Mask to prevent the decoder from attending to padding tokens and future tokens.

		Returns:
			Tensor: Output tensor of shape (batch_size, sequence_length, features)
					after processing through all decoder layers.
		"""
		# Pass the input through each decoder block in sequence
		for decoder in self.decoder_module_list:
			decoder_input = decoder(decoder_input, encoder_output, encoder_mask, decoder_mask)

		return decoder_input


class ProjectionLayer(nn.Module, ABC):
	"""
	Placeholder class so that the projection layer used in the transformer can be specified as needed
	Is supposed to imitate the strategy design pattern
	"""

	class InterfaceException(Exception):
		"""Custom exception raised when trying to call a method from the class that simulates an interface."""
		pass

	@abstractmethod
	def __init__(self, model_dimensions: int, vocab_size: int):
		"""
		Fake initialization Function to force all subclasses to implement the correct constructor

		Args:
			model_dimensions (int): The dimensionality of the output from the decoder layers.
			vocab_size (int): The size of the output vocabulary.
		"""
		super().__init__()

		self.model_dimensions = model_dimensions
		self.vocab_size = vocab_size

		# raise self.InterfaceException(
		# 	"This class is supposed to act like an Interface, you can't have an instance of it."
		# )

	@abstractmethod
	def forward(self, x):
		"""Perform a forward pass (to be implemented by subclasses)."""
		# raise self.InterfaceException("Please use a concrete implementation for your model.")


class OutputProjectionLayerForNLLLoss(ProjectionLayer):
	"""
	Combines a linear layer and log-softmax to produce probabilities over the vocabulary.
	((batch_size, sequence_length, features), (batch_size, sequence_length, vocab_size))

	Used for Loss Calculation using NLLLoss since it expects the probabilities
	"""

	def __init__(self, model_dimensions: int, vocab_size: int):
		"""
		Initializes the OutputLayer with a linear transformation layer.

		Args:
			model_dimensions (int): The dimensionality of the output from the decoder layers.
			vocab_size (int): The size of the output vocabulary.
		"""
		# throws custom InterfaceException, but this class is a valid class and not a mock-interface,
		# shouldn't throw an exception
		try:
			super().__init__(model_dimensions, vocab_size)
		except ProjectionLayer.InterfaceException:
			pass
		self.linear_layer = nn.Linear(model_dimensions, vocab_size)

	def forward(self, decoder_output):
		"""
		Forward pass through the OutputLayer.

		Args:
			decoder_output (Tensor): Input tensor of shape (batch_size, sequence_length, model_dimensions),

		Returns:
			Tensor: Output tensor of shape (batch_size, sequence_length, vocab_size),
			after applying the linear transformation and log softmax to the input tensor.
		"""
		linear_output = self.linear_layer(decoder_output)
		return torch.log_softmax(linear_output, dim=-1)


class OutputProjectionLayerForCrossEntropyLoss(ProjectionLayer):
	"""
	Projection Layer that only consists of a linear layer to produce an output over the vocabulary.
	((batch_size, sequence_length, features), (batch_size, sequence_length, vocab_size))

	Used for Loss Calculation using CrossEntropyLoss since the softmax function application will happen internally.
	"""

	def __init__(self, model_dimensions: int, vocab_size: int):
		"""
		Initializes the OutputLayer with a linear transformation layer.

		Args:
			model_dimensions (int): The dimensionality of the output from the decoder layers.
			vocab_size (int): The size of the output vocabulary.
		"""
		# throws custom InterfaceException, but this class is a valid class and not a mock-interface,
		# shouldn't throw an exception
		try:
			super().__init__(model_dimensions, vocab_size)
		except ProjectionLayer.InterfaceException:
			pass
		self.linear_layer = nn.Linear(model_dimensions, vocab_size)

	def forward(self, decoder_output):
		"""
		Forward pass through the OutputLayer.

		Args:
			decoder_output (Tensor): Input tensor of shape (batch_size, sequence_length, model_dimensions),

		Returns:
			Tensor: Output tensor of shape (batch_size, sequence_length, vocab_size),
			after applying the linear transformation.
		"""
		return self.linear_layer(decoder_output)


class Transformer(nn.Module):
	"""
	Full Transformer module

	Methods:
		encode(source, source_mask) -> Tensor: Encodes the input sequence

		decode(encoder_output, encoder_mask, target, decoder_mask) -> Tensor: Decodes the target sequence

		project(output_tensor) -> Tensor: Projects the output back into the vocabulary space,
			specifics are dependent on the concrete Projection Layer Used (in- or excluding softmax)
	"""

	def __init__(self,
				 encoder: Encoder,
				 decoder: Decoder,
				 source_embedding_layer: InputEmbeddings,
				 target_embedding_layer: InputEmbeddings,
				 source_positional_encoding_layer: PositionalEncodings,
				 target_positional_encoding_layer: PositionalEncodings,
				 output_projection_layer: ProjectionLayer
				 ):
		"""
		Initializes the Transformer model.

		Args:
			encoder (Encoder): The encoder module for processing source sequences.
			decoder (Decoder): The decoder module for generating target sequences.
			source_embedding_layer (InputEmbeddings): Embedding layer for source sequences.
			target_embedding_layer (InputEmbeddings): Embedding layer for target sequences.
			source_positional_encoding_layer (PositionalEncodings): Positional encoding layer for source sequences.
			target_positional_encoding_layer (PositionalEncodings): Positional encoding layer for target sequences.
			output_projection_layer (ProjectionLayer): Layer that projects decoder outputs to the vocabulary space.
		"""
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.source_embedding_layer = source_embedding_layer
		self.target_embedding_layer = target_embedding_layer
		self.source_pe_layer = source_positional_encoding_layer
		self.target_pe_layer = target_positional_encoding_layer
		self.output_projection_layer = output_projection_layer

	def encode(self, input_tokens, encoder_mask):
		"""
		Encodes the source sequence.
		((batch size, source_sequence), (batch_size, source_sequence, model_dimensions))

		Args:
			input_tokens (Tensor): Input sequence of shape (batch_size, source_sequence).
			encoder_mask (Tensor): Attention mask to ignore padding tokens in the source sequence.

		Returns:
			Tensor: Encoded source sequence of shape (batch_size, source_sequence, dimensions_model).
		"""
		# Apply source embeddings and positional encodings
		input_tokens = self.source_embedding_layer(input_tokens)
		input_tokens = self.source_pe_layer(input_tokens)

		# Pass through the encoder
		return self.encoder(input_tokens, encoder_mask)

	def decode(self, encoder_output, encoder_mask, decoder_input_sequence, decoder_mask):
		"""
		Decodes the target sequence using encoder output.
		((batch_size, source_sequence, dimensions_model), (batch_size, target_sequence, dimensions_model))

		Args:
			encoder_output (Tensor): Output from the encoder of shape (batch_size, source_sequence, dimensions_model).
			encoder_mask (Tensor): Attention mask for the encoder output, ignoring padding tokens.
			decoder_input_sequence (Tensor): Input sequence of decoder in the shape (batch_size, target_sequence).
			decoder_mask (Tensor): Attention mask for the target sequence, including continuos token masking.

		Returns:
			Tensor: Decoded output of shape (batch_size, target_sequence, dimensions_model).
		"""
		decoder_input_sequence = self.target_embedding_layer(decoder_input_sequence)
		decoder_input_sequence = self.target_pe_layer(decoder_input_sequence)
		return self.decoder(decoder_input_sequence, encoder_output, encoder_mask, decoder_mask)

	def project(self, output_tensor):
		"""
		Projects the decoder output into the vocabulary space.

		Args:
			output_tensor (Tensor): Decoder output of shape (batch_size, sequence_length, model_dimensions).

		Returns:
			Tensor: of shape (batch_size, sequence_length, vocab_size). specifics are dependent on
				the concrete Projection Layer Used (in- or excluding softmax)
		"""
		return self.output_projection_layer(output_tensor)


class TransformerBuilder:
	"""
	A utility class for building and initializing a Transformer model.

	This class separates the model construction from the `Transformer` class to avoid bloating its `__init__` method.
	It handles the creation of embeddings, positional encodings, encoder/decoder blocks, and projection layers,
	making the Transformer model easier to maintain and extend.

	Methods:
		build_transformer: Builds and returns a fully initialized Transformer model with specified configurations.
	"""
	@staticmethod
	def build_transformer(source_vocab_size: int, target_vocab_size: int,
						  source_sequence_length: int, target_sequence_length: int,
						  loss_function_is_NLLLoss: bool,
						  loss_function_is_CrossEntropyLoss: bool,
						  model_dimensions: int = 512,
						  number_of_encoder_and_decoder_blocks: int = 6,
						  number_of_heads_in_multi_head_attention: int = 8,
						  dropout: float = 0.1,
						  feed_forward_hidden_layer_dimensions: int = 2048,
						  ) -> Transformer:
		"""
		Builds and returns a Transformer model.

		Args:
			source_vocab_size (int): Size of the source vocabulary.
			target_vocab_size (int): Size of the target vocabulary.
			source_sequence_length (int): Maximum sequence length for source input.
			target_sequence_length (int): Maximum sequence length for target input.
			loss_function_is_NLLLoss (bool): If the loss function, used with the output of the transformer is NLLLoss.
			loss_function_is_CrossEntropyLoss (bool): If the loss function,
				used with the output of the transformer is CrossEntropyLoss
			model_dimensions (int, optional): Dimensionality of model layers. Default is 512.
			number_of_encoder_and_decoder_blocks (int, optional): Number of encoder/decoder blocks. Default is 6.
			number_of_heads_in_multi_head_attention (int, optional): Number of attention heads. Default is 8.
			dropout (float, optional): Dropout probability. (for more information look at the note) Default is 0.1.
			feed_forward_hidden_layer_dimensions (int, optional): Hidden layer size in feed-forward sublayer.
				Default is 2048.

		Returns:
			Transformer: A fully initialized Transformer model.

		Raises:
			AssertionError: If an invalid loss function type is provided or the projection layer isn't set up correctly.

		Note:
			- Dropout is used during training to prevent overfitting by randomly setting a fraction of the input
			  units to zero at each update. This helps improve the generalization of the model by preventing it
			  from becoming overly reliant on certain features.
		"""
		# create embedding layers
		source_embedding_layer = InputEmbeddings(model_dimensions, source_vocab_size)
		target_embedding_layer = InputEmbeddings(model_dimensions, target_vocab_size)

		# create positional encoding layers
		source_positional_encoding_layer = PositionalEncodings(model_dimensions, source_sequence_length, dropout)
		target_positional_encoding_layer = PositionalEncodings(model_dimensions, target_sequence_length, dropout)

		# create encoder and decoder
		encoder = Encoder(model_dimensions, number_of_encoder_and_decoder_blocks,
						  number_of_heads_in_multi_head_attention, feed_forward_hidden_layer_dimensions, dropout)
		decoder = Decoder(model_dimensions, number_of_encoder_and_decoder_blocks,
						  number_of_heads_in_multi_head_attention, feed_forward_hidden_layer_dimensions, dropout)

		# create projection layer
		assert loss_function_is_CrossEntropyLoss != loss_function_is_NLLLoss, "select (only) one of the loss functions"
		projection_layer = None
		if loss_function_is_CrossEntropyLoss:
			projection_layer = OutputProjectionLayerForCrossEntropyLoss(model_dimensions, target_vocab_size)
		if loss_function_is_NLLLoss:
			projection_layer = OutputProjectionLayerForNLLLoss(model_dimensions, target_vocab_size)

		assert projection_layer is not None, "something went horribly wrong"

		# create transformer
		transformer = Transformer(encoder, decoder,
								  source_embedding_layer, target_embedding_layer,
								  source_positional_encoding_layer, target_positional_encoding_layer,
								  projection_layer
								  )

		# initialize parameters
		for parameter in transformer.parameters():
			if parameter.dim() > 1:
				nn.init.xavier_uniform_(parameter)
				# Xavier-initialization is used to maintain a good variance balance in weights, since it led to good
				# results in literature

		return transformer
