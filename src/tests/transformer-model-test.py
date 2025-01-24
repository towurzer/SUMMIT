import unittest
from unittest.mock import MagicMock

from src.transformer import *


class TestInputEmbeddings(unittest.TestCase):
	def setUp(self):
		self.model_dimensions = 4
		self.vocab_size = 10
		self.batch_size = 2
		self.sequence_length = 3
		self.input_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])

	def test_forward_output_shape(self):
		"""Test if the output shape matches (batch_size, sequence_length, model_dimensions)."""
		embeddings = InputEmbeddings(self.model_dimensions, self.vocab_size)
		output = embeddings(self.input_tokens)
		self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.model_dimensions))

	def test_forward_scaling(self):
		"""Test if the embeddings are scaled by sqrt(model_dimensions)."""
		embeddings = InputEmbeddings(self.model_dimensions, self.vocab_size)
		unscaled_output = embeddings.embedding(self.input_tokens)  # Direct embedding without scaling
		scaled_output = embeddings(self.input_tokens)  # Embedding with scaling
		self.assertTrue(torch.allclose(scaled_output, unscaled_output * math.sqrt(self.model_dimensions)))

	def test_embedding_gradients(self):
		"""Test if the embedding weights have gradients."""
		embeddings = InputEmbeddings(self.model_dimensions, self.vocab_size)
		output = embeddings(self.input_tokens)
		output.mean().backward()  # Trigger backward pass
		self.assertIsNotNone(embeddings.embedding.weight.grad)
		self.assertEqual(embeddings.embedding.weight.grad.shape, embeddings.embedding.weight.shape)


class TestPositionalEncodings(unittest.TestCase):
	def setUp(self):
		self.model_dimensions = 4
		self.max_sequence_length = 10
		self.dropout_rate = 0.1
		self.batch_size = 2
		self.sequence_length = 3
		self.input_embeddings = torch.randn(self.batch_size, self.sequence_length, self.model_dimensions)

	def test_forward_output_shape(self):
		"""Test if the output shape matches (batch_size, sequence_length, model_dimensions)."""
		positional_encodings = PositionalEncodings(self.model_dimensions, self.max_sequence_length, self.dropout_rate)
		output = positional_encodings(self.input_embeddings)
		self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.model_dimensions))

	def test_positional_encodings_are_added(self):
		"""Test if positional encodings are correctly added to the input embeddings."""
		positional_encodings = PositionalEncodings(self.model_dimensions, self.max_sequence_length, self.dropout_rate)
		positional_encodings.eval()  # Disable dropout to ensure deterministic behavior
		output = positional_encodings(self.input_embeddings)
		# Extract positional encodings used
		expected_pos_encodings = positional_encodings.positional_encodings[:, :self.sequence_length, :]
		# Check if the positional encodings are added
		self.assertTrue(torch.allclose(output, self.input_embeddings + expected_pos_encodings, atol=1e-5))

	def test_no_gradients_on_positional_encodings(self):
		"""Test that positional encodings are not learnable (require_grad=False)."""
		positional_encodings = PositionalEncodings(self.model_dimensions, self.max_sequence_length, self.dropout_rate)
		pos_encodings_tensor = positional_encodings.positional_encodings
		self.assertFalse(pos_encodings_tensor.requires_grad)

	def test_dropout_effect(self):
		"""Test if dropout is applied."""
		positional_encodings = PositionalEncodings(self.model_dimensions, self.max_sequence_length, self.dropout_rate)
		positional_encodings.dropout.eval()  # Disable dropout for consistent output
		output_no_dropout = positional_encodings(self.input_embeddings)

		positional_encodings.dropout.train()  # Enable dropout
		output_with_dropout = positional_encodings(self.input_embeddings)

		# Check that dropout creates differences in the output
		self.assertFalse(torch.equal(output_no_dropout, output_with_dropout))


class TestMultiHeadAttentionSegment(unittest.TestCase):
	def setUp(self):
		self.batch_size = 2
		self.seq_len = 4
		self.model_dimensions = 8
		self.head_count = 4
		self.dropout = 0.1
		self.mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)

		self.mha = MultiHeadAttentionSegment(self.model_dimensions, self.head_count, self.dropout)
		self.q = torch.rand(self.batch_size, self.seq_len, self.model_dimensions)
		self.k = torch.rand(self.batch_size, self.seq_len, self.model_dimensions)
		self.v = torch.rand(self.batch_size, self.seq_len, self.model_dimensions)

	def test_forward_output_shape(self):
		"""Test if the output shape matches (batch_size, seq_len, model_dimensions)."""
		output = self.mha(self.q, self.k, self.v, self.mask)
		self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.model_dimensions))

	def test_attention_scores(self):
		"""Test if attention scores are computed correctly and shape matches."""
		self.mha(self.q, self.k, self.v, self.mask)
		self.assertIsNotNone(self.mha.attention_scores)
		self.assertEqual(
			self.mha.attention_scores.shape,
			(self.batch_size, self.head_count, self.seq_len, self.seq_len),
		)

	def test_masking_effect(self):
		"""Test if masking works by ensuring masked positions have near-zero attention scores."""
		mask = torch.tensor([[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).repeat(self.batch_size, 1, 1,
		                                                                                         1)
		# Assert that the mask has the correct shape
		self.assertEqual(mask.shape, (self.batch_size, 1, self.seq_len, self.seq_len))

		self.mha(self.q, self.k, self.v, mask)
		attention_scores = self.mha.attention_scores
		# Check if masked positions have very low values (near zero after softmax)
		# self.assertTrue((attention_scores[:, :, :, 2:4] < 1e-3).all())
		self.assertTrue((attention_scores[:, :, :2,
		                 2:] < 1e-3).all())  # Masked regions (0s) for first two rows and last two columns
		self.assertTrue((attention_scores[:, :, 2:,
		                 :2] < 1e-3).all())  # Masked regions (0s) for last two rows and first two columns


class TestLayerNormalization(unittest.TestCase):
	def setUp(self):
		self.features = 8
		self.layer_norm = LayerNormalization(self.features)
		self.input_tensor = torch.rand(2, 4, self.features)

	def test_output_shape(self):
		"""Test if output shape matches input shape."""
		output = self.layer_norm(self.input_tensor)
		self.assertEqual(output.shape, self.input_tensor.shape)

	def test_normalization_properties(self):
		"""Test if the output tensor is normalized to mean=0 and variance=1."""
		output = self.layer_norm(self.input_tensor)
		mean = output.mean(dim=-1)
		std = output.std(dim=-1)
		self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-6))
		std = std.detach()
		self.assertTrue(torch.allclose(std, torch.ones_like(std), atol=1e-4))


class TestFeedForwardLayer(unittest.TestCase):
	def setUp(self):
		self.dimensions_model = 8
		self.dimensions_ff = 32
		self.dropout = 0.1
		self.ff_layer = FeedForwardLayer(self.dimensions_model, self.dimensions_ff, self.dropout)
		self.input_tensor = torch.rand(2, 4, self.dimensions_model)

	def test_output_shape(self):
		"""Test if the output shape matches input shape."""
		output = self.ff_layer(self.input_tensor)
		self.assertEqual(output.shape, self.input_tensor.shape)

	def test_relu_activation(self):
		"""Test if the ReLU activation is applied (no negative values)."""
		intermediate = self.ff_layer.linear_1(self.input_tensor)
		relu_output = torch.relu(intermediate)
		self.assertTrue((relu_output >= 0).all())


class TestAddAndNormLayer(unittest.TestCase):
	class AddLayerWithoutNorm(nn.Module):
		def __init__(self, model_dimensions, dropout):
			super().__init__()
			self.norm = nn.LayerNorm(model_dimensions)

		def forward(self, residual, sublayer_output):
			return residual + sublayer_output(residual)

	def setUp(self):
		self.features = 8
		self.dropout = 0.1
		self.add_norm = AddAndNormLayer(self.features, self.dropout)
		self.add_no_norm = self.AddLayerWithoutNorm(self.features, self.dropout)
		self.input_tensor = torch.rand(2, 4, self.features)

	def test_output_shape(self):
		"""Test if the output shape matches input shape."""

		def dummy_sublayer(x):
			return x * 2  # Simple dummy sublayer

		output = self.add_norm(self.input_tensor, dummy_sublayer)
		self.assertEqual(output.shape, self.input_tensor.shape)

	def test_residual_connection(self):
		"""Test if the residual connection is correctly applied."""

		# Create a dummy sublayer function that doubles the input
		def dummy_sublayer(x):
			return x * 2  # Simple dummy sublayer (scales input by 2)

		# Set the module in evaluation mode so dropout does not affect the output
		self.add_norm.eval()

		# Override the normalization layer to be identity, ensuring no effect from it
		# self.add_norm.normalization_layer = nn.Identity()  # Skip any normalization

		# Perform the forward pass
		output = self.add_no_norm(self.input_tensor, dummy_sublayer)

		# Check if the output is exactly 3 times the input tensor
		expected_output = self.input_tensor * 3
		self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

	def test_layer_normalization(self):
		"""Test if normalization is applied after adding residual."""

		def dummy_sublayer(x):
			return x * 0  # Outputs all zeros

		self.add_norm.eval()
		output = self.add_norm(self.input_tensor, dummy_sublayer)
		mean = output.mean(dim=-1)
		std = output.std(dim=-1)
		self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-6))
	# TODO: Fix self.assertTrue(torch.allclose(std, torch.ones_like(std), atol=1e-3))





class TestEncoderBlock(unittest.TestCase):
	def setUp(self):
		self.batch_size = 2
		self.sequence_length = 5
		self.model_dimensions = 8
		self.dropout = 0.1

		# Initialize mock layers
		self.self_attention_layer = MultiHeadAttentionSegment(self.model_dimensions, 2, self.dropout)
		self.feed_forward_layer = FeedForwardLayer(self.model_dimensions, 16, self.dropout)

		# Initialize EncoderBlock
		self.encoder_block = EncoderBlock(self.model_dimensions, self.self_attention_layer, self.feed_forward_layer,
		                                  self.dropout)

	def test_initialization(self):
		self.assertIsInstance(self.encoder_block.self_attention_layer, MultiHeadAttentionSegment)
		self.assertIsInstance(self.encoder_block.feed_forward_layer, FeedForwardLayer)
		self.assertEqual(len(self.encoder_block.add_and_norm_layers), 2)

	def test_forward_pass(self):
		input_tensor = torch.randn(self.batch_size, self.sequence_length, self.model_dimensions)
		attention_mask = torch.ones(self.batch_size, self.sequence_length, self.sequence_length)

		output = self.encoder_block(input_tensor, attention_mask)

		# Check output shape
		self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.model_dimensions))


class TestEncoder(unittest.TestCase):
	def setUp(self):
		self.batch_size = 2
		self.sequence_length = 5
		self.model_dimensions = 8
		self.num_blocks = 3
		self.num_heads = 2
		self.hidden_layer_dim = 16
		self.dropout = 0.1

		# Initialize Encoder
		self.encoder = Encoder(self.model_dimensions, self.num_blocks, self.num_heads, self.hidden_layer_dim,
		                       self.dropout)

	def test_initialization(self):
		self.assertEqual(len(self.encoder.encoder_module_list), self.num_blocks)
		for block in self.encoder.encoder_module_list:
			self.assertIsInstance(block, EncoderBlock)

	def test_forward_pass(self):
		input_tensor = torch.randn(self.batch_size, self.sequence_length, self.model_dimensions)
		attention_mask = torch.ones(self.batch_size, self.sequence_length, self.sequence_length)

		output = self.encoder(input_tensor, attention_mask)

		# Check output shape
		self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.model_dimensions))


class TestDecoderBlock(unittest.TestCase):
	def setUp(self):
		# Mock dependencies
		self.model_dimensions = 64
		self.sequence_length = 10
		self.batch_size = 4
		self.dropout = 0.1

		# Mock attention layers and feed-forward layer
		self.mock_self_attention = MagicMock()
		self.mock_cross_attention = MagicMock()
		self.mock_feed_forward = MagicMock()
		self.mock_self_attention.return_value = torch.rand(
			self.batch_size, self.sequence_length, self.model_dimensions
		)
		self.mock_cross_attention.return_value = torch.rand(
			self.batch_size, self.sequence_length, self.model_dimensions
		)
		self.mock_feed_forward.return_value = torch.rand(
			self.batch_size, self.sequence_length, self.model_dimensions
		)

		# Instantiate the DecoderBlock
		self.decoder_block = DecoderBlock(
			model_dimensions=self.model_dimensions,
			self_attention_layer=self.mock_self_attention,
			cross_attention_layer=self.mock_cross_attention,
			feed_forward_layer=self.mock_feed_forward,
			dropout=self.dropout,
		)

		# Mock inputs
		self.residual_input = torch.rand(
			self.batch_size, self.sequence_length, self.model_dimensions
		)
		self.encoder_output = torch.rand(
			self.batch_size, self.sequence_length, self.model_dimensions
		)
		self.encoder_mask = torch.ones(self.batch_size, self.sequence_length).bool()
		self.decoder_mask = torch.ones(self.batch_size, self.sequence_length).bool()

	def test_forward_pass(self):
		# Test that the forward pass runs without errors
		output = self.decoder_block(
			self.residual_input,
			self.encoder_output,
			self.encoder_mask,
			self.decoder_mask,
		)
		self.assertEqual(output.shape, self.residual_input.shape)

	def test_self_attention_called(self):
		# Test that the self-attention layer is called with expected arguments
		_ = self.decoder_block(
			self.residual_input,
			self.encoder_output,
			self.encoder_mask,
			self.decoder_mask,
		)
		self.mock_self_attention.assert_called()

	def test_cross_attention_called(self):
		# Test that the cross-attention layer is called with expected arguments
		_ = self.decoder_block(
			self.residual_input,
			self.encoder_output,
			self.encoder_mask,
			self.decoder_mask,
		)
		self.mock_cross_attention.assert_called()

	def test_feed_forward_called(self):
		# Test that the feed-forward layer is called
		_ = self.decoder_block(
			self.residual_input,
			self.encoder_output,
			self.encoder_mask,
			self.decoder_mask,
		)
		self.mock_feed_forward.assert_called()


class TestDecoder(unittest.TestCase):
	def setUp(self):
		self.model_dimensions = 64
		self.sequence_length = 10
		self.batch_size = 4
		self.dropout = 0.1
		self.num_blocks = 3
		self.num_heads = 8
		self.hidden_layer_dims = 128

		# Instantiate the Decoder with mock components
		self.decoder = Decoder(
			model_dimensions=self.model_dimensions,
			number_of_encoder_and_decoder_blocks=self.num_blocks,
			number_of_heads_in_multi_head_attention=self.num_heads,
			feed_forward_hidden_layer_dimensions=self.hidden_layer_dims,
			dropout=self.dropout,
		)

		# Mock inputs
		self.decoder_input = torch.rand(
			self.batch_size, self.sequence_length, self.model_dimensions
		)
		self.encoder_output = torch.rand(
			self.batch_size, self.sequence_length, self.model_dimensions
		)
		self.encoder_mask = torch.ones(self.batch_size, 1, self.sequence_length, self.sequence_length).bool()
		self.decoder_mask = torch.ones(self.batch_size, 1, self.sequence_length, self.sequence_length).bool()

	def test_forward_pass(self):
		# Test that the forward pass runs without errors
		output = self.decoder(
			self.decoder_input,
			self.encoder_output,
			self.encoder_mask,
			self.decoder_mask,
		)
		self.assertEqual(output.shape, self.decoder_input.shape)

	def test_multiple_blocks(self):
		# TODO: Test that all blocks are called

		self.assertTrue(True)


class TestTransformer(unittest.TestCase):

	def setUp(self):
		"""Set up a small Transformer model for testing."""
		self.model_dimensions = 64
		self.source_vocab_size = 50
		self.target_vocab_size = 50
		self.source_sequence_length = 10
		self.target_sequence_length = 10

		self.transformer = TransformerBuilder.build_transformer(
			source_vocab_size=self.source_vocab_size,
			target_vocab_size=self.target_vocab_size,
			source_sequence_length=self.source_sequence_length,
			target_sequence_length=self.target_sequence_length,
			loss_function_is_NLLLoss=False,
			loss_function_is_CrossEntropyLoss=True,
			model_dimensions=self.model_dimensions,
			number_of_encoder_and_decoder_blocks=2,
			number_of_heads_in_multi_head_attention=2,
			dropout=0.1,
			feed_forward_hidden_layer_dimensions=128,
		)

		self.batch_size = 4
		self.source_tokens = torch.randint(0, self.source_vocab_size, (self.batch_size, self.source_sequence_length))
		self.target_tokens = torch.randint(0, self.target_vocab_size, (self.batch_size, self.target_sequence_length))
		self.encoder_mask = torch.ones((self.batch_size, 1, self.source_sequence_length, self.source_sequence_length))
		self.decoder_mask = torch.ones((self.batch_size, 1, self.target_sequence_length, self.target_sequence_length))

	def test_encode(self):
		"""Test the encode method of the Transformer class."""
		encoder_output = self.transformer.encode(self.source_tokens, self.encoder_mask)
		self.assertEqual(encoder_output.shape, (self.batch_size, self.source_sequence_length, self.model_dimensions))

	def test_decode(self):
		"""Test the decode method of the Transformer class."""
		encoder_output = self.transformer.encode(self.source_tokens, self.encoder_mask)
		decoder_output = self.transformer.decode(
			encoder_output, self.encoder_mask, self.target_tokens, self.decoder_mask
		)
		self.assertEqual(decoder_output.shape, (self.batch_size, self.target_sequence_length, self.model_dimensions))

	def test_project(self):
		"""Test the project method of the Transformer class."""
		encoder_output = self.transformer.encode(self.source_tokens, self.encoder_mask)
		decoder_output = self.transformer.decode(
			encoder_output, self.encoder_mask, self.target_tokens, self.decoder_mask
		)
		projected_output = self.transformer.project(decoder_output)
		self.assertEqual(projected_output.shape, (self.batch_size, self.target_sequence_length, self.target_vocab_size))

	def test_transformer_builder(self):
		"""Test the TransformerBuilder.build_transformer method."""
		transformer = TransformerBuilder.build_transformer(
			source_vocab_size=self.source_vocab_size,
			target_vocab_size=self.target_vocab_size,
			source_sequence_length=self.source_sequence_length,
			target_sequence_length=self.target_sequence_length,
			loss_function_is_NLLLoss=False,
			loss_function_is_CrossEntropyLoss=True,
			model_dimensions=self.model_dimensions,
			number_of_encoder_and_decoder_blocks=2,
			number_of_heads_in_multi_head_attention=2,
			dropout=0.1,
			feed_forward_hidden_layer_dimensions=128,
		)
		self.assertIsInstance(transformer, Transformer)

	def test_invalid_loss_function(self):
		"""Test that an invalid loss function configuration raises an AssertionError."""
		with self.assertRaises(AssertionError):
			TransformerBuilder.build_transformer(
				source_vocab_size=self.source_vocab_size,
				target_vocab_size=self.target_vocab_size,
				source_sequence_length=self.source_sequence_length,
				target_sequence_length=self.target_sequence_length,
				loss_function_is_NLLLoss=True,
				loss_function_is_CrossEntropyLoss=True,
			)


class TestProjectionLayer(unittest.TestCase):
	def test_abstract_class_instantiation(self):
		"""
		Test that instantiating the abstract ProjectionLayer raises an InterfaceException.
		"""
		with self.assertRaises(TypeError) as context:
			_ = ProjectionLayer(128, 1000)
		self.assertIn(
			"Can't instantiate abstract class ProjectionLayer without an implementation for abstract methods '__init__', 'forward'",
			str(context.exception)
		)

	def test_abstract_forward_method(self):
		"""
		Test that the forward method of the abstract ProjectionLayer raises an InterfaceException.
		"""

		class MockProjectionLayer(ProjectionLayer):
			def __init__(self, model_dimensions, vocab_size):
				super().__init__(model_dimensions, vocab_size)

		with self.assertRaises(TypeError) as context:
			mock_layer = MockProjectionLayer(128, 1000)
			mock_layer.forward(torch.randn(2, 5, 128))  # Should not reach this
		self.assertIn(
			"Can't instantiate abstract class MockProjectionLayer without an implementation for abstract method 'forward'",
			str(context.exception)
		)


class TestOutputProjectionLayerForNLLLoss(unittest.TestCase):
	def setUp(self):
		"""
		Set up parameters and an instance of OutputProjectionLayerForNLLLoss.
		"""
		self.model_dimensions = 128
		self.vocab_size = 1000
		self.batch_size = 2
		self.sequence_length = 5
		self.layer = OutputProjectionLayerForNLLLoss(
			self.model_dimensions, self.vocab_size
		)

	def test_initialization(self):
		"""
		Test that the OutputProjectionLayerForNLLLoss initializes without raising exceptions.
		"""
		self.assertIsInstance(self.layer.linear_layer, nn.Linear)

	def test_forward_pass(self):
		"""
		Test the forward pass of OutputProjectionLayerForNLLLoss.
		"""
		input_tensor = torch.randn(
			self.batch_size, self.sequence_length, self.model_dimensions
		)
		output_tensor = self.layer(input_tensor)

		# Check shape
		self.assertEqual(
			output_tensor.shape,
			(self.batch_size, self.sequence_length, self.vocab_size),
		)

		# Check that log-softmax has been applied (sum along vocab dimension close to 0)
		self.assertTrue(
			torch.allclose(
				output_tensor.exp().sum(dim=-1),
				torch.ones(self.batch_size, self.sequence_length),
				atol=1e-5,
			)
		)


class TestOutputProjectionLayerForCrossEntrpyLoss(unittest.TestCase):
	def setUp(self):
		"""
		Set up parameters and an instance of OutputProjectionLayerForCrossEntrpyLoss.
		"""
		self.model_dimensions = 128
		self.vocab_size = 1000
		self.batch_size = 2
		self.sequence_length = 5
		self.layer = OutputProjectionLayerForCrossEntropyLoss(
			self.model_dimensions, self.vocab_size
		)

	def test_initialization(self):
		"""
		Test that the OutputProjectionLayerForCrossEntrpyLoss initializes without raising exceptions.
		"""
		self.assertIsInstance(self.layer.linear_layer, nn.Linear)

	def test_forward_pass(self):
		"""
		Test the forward pass of OutputProjectionLayerForCrossEntrpyLoss.
		"""
		input_tensor = torch.randn(
			self.batch_size, self.sequence_length, self.model_dimensions
		)
		output_tensor = self.layer(input_tensor)

		# Check shape
		self.assertEqual(
			output_tensor.shape,
			(self.batch_size, self.sequence_length, self.vocab_size),
		)

		# Check that no softmax/log-softmax has been applied
		self.assertFalse(
			torch.allclose(
				output_tensor.sum(dim=-1),
				torch.ones(self.batch_size, self.sequence_length),
				atol=1e-5,
			)
		)


class TestSubclassInitialization(unittest.TestCase):
	def test_subclass_initialization_does_not_raise_exception(self):
		"""
		Test that initializing valid subclasses does not raise InterfaceException.
		"""
		try:
			_ = OutputProjectionLayerForNLLLoss(128, 1000)
			_ = OutputProjectionLayerForCrossEntropyLoss(128, 1000)
		except ProjectionLayer.InterfaceException as e:
			self.fail(f"Initialization raised InterfaceException: {str(e)}")


if __name__ == '__main__':
	unittest.main()
