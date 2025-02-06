from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# tokenizers (using package)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# dataset
from config import Config
from dataset import TranslationDataset

# model
from transformer import TransformerBuilder

class Model():
	def __init__(self, config:Config):
		self.config = config

	def create_model(self):
		self.model = TransformerBuilder.build_transformer(self.tokenizer_source.get_vocab_size(), self.tokenizer_target.get_vocab_size(), self.config.train_config["max_sentence_tokens"], self.config.train_config["max_sentence_tokens"], False, True, self.config.train_config["model_dimensions"], self.config.train_config["num_encoder_blocks"], self.config.train_config["num_heads"], self.config.train_config["dropout"]).to(self.config.device)


	def load_latest_model(self):
		print("Loading tokenizers...")
		self.tokenizer_source = self.load_tokenizer(self.config.data_config["lang_source"])
		self.tokenizer_target = self.load_tokenizer(self.config.data_config["lang_target"])

		if self.tokenizer_source == None or self.tokenizer_target == None: raise FileNotFoundError("Tokenizer files not found!")

		print("Loading model")
		# TODO: make use of different configurations ?????
		self.create_model()

		model_path = self.config.get_latest_model_path()
		if model_path is None: raise FileNotFoundError("No existing model found!")

		state = torch.load(model_path, map_location=self.config.device)
		self.model.load_state_dict(state['model_states'])

	def save_current_model(self):
		print("Saving current model")
		filename = self.config.model_folder / Path(f"latest.pth")
		torch.save({
			'model_states': self.model.state_dict()
		}, filename)
		print('Finished saving model!')

	def tokenize(self, source):
		max_tokens = self.config.train_config["max_sentence_tokens"]
		s_token = torch.tensor([self.tokenizer_source.token_to_id("<S>")], dtype=torch.int64)
		e_token = torch.tensor([self.tokenizer_source.token_to_id("<E>")], dtype=torch.int64)
		p_token = torch.tensor([self.tokenizer_source.token_to_id("<P>")], dtype=torch.int64)

		encoded = self.tokenizer_source.encode(source).ids

		num_padding = max_tokens - len(encoded) - 2

		if num_padding < 0: raise OverflowError(f"Too many words, maximum is set to {self.max_tokens}! If more words are desired, change the limit in the configuration and retrain.")

		to_encoder = torch.cat([
			s_token,
			torch.tensor(encoded, dtype=torch.int64),
			e_token,
			torch.tensor([p_token] * num_padding, dtype=torch.int64)
		])
		mask_encoder = (to_encoder != p_token).unsqueeze(0).unsqueeze(0).int()
		return to_encoder, mask_encoder


	def decode(self, source):
		s_token = self.tokenizer_target.token_to_id("<S>")
		e_token = self.tokenizer_target.token_to_id("<E>")

		with torch.no_grad():
			self.model.eval() # set to evaluation mode


			# prepare for handing to transformer => needs unsqueeze as it's trained on batches
			to_encoder, mask_encoder = self.tokenize(source)
			to_encoder = torch.unsqueeze(to_encoder, 0).to(self.config.device)
			mask_encoder = torch.unsqueeze(mask_encoder, 0).to(self.config.device)

			if to_encoder.size(0) > 1: raise ValueError("For evaluation dimension must be 1!")

			# decode
			encoded = self.model.encode(to_encoder, mask_encoder)
			to_decoder = torch.empty(1,1).fill_(s_token).type_as(to_encoder).to(self.config.device)

			for _ in range(0, self.config.train_config["max_sentence_tokens"]):
				mask_decoder = TranslationDataset.triangular_mask(to_decoder.size(1)).type_as(mask_encoder).to(self.config.device)
				
				# get output
				output = self.model.decode(encoded, mask_encoder, to_decoder, mask_decoder)

				p = self.model.project(output[:, -1])
				_, most_likely = torch.max(p, dim=1)

				if most_likely == e_token: break # we reached the end
				
				# next run with old content to decode + most likely token
				to_decoder = torch.cat(
					[
						to_decoder, #last input
						torch.empty(1,1).type_as(to_encoder).fill_(most_likely.item()).to(self.config.device)
					], dim=1
				)
			
			# get the sentences back out from the tokens
			estimated = self.tokenizer_target.decode(to_decoder.squeeze(0).detach().cpu().numpy())

			return estimated

	def load_tokenizer(self, language):
		file = Path(self.config.tokenize_folder) / f"{language}.json"
		print(f"Looking for tokenizer file: {file.resolve()}")
		if Path.exists(file): 
			print(f"Loading existing tokenizer file for language {language}...")
			return Tokenizer.from_file(str(file)) #tokenizer already exists, just load
		else: return None