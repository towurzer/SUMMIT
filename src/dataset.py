import torch
import torch.nn
from torch.utils.data import Dataset
from config import Config

# dataset for translation task
class TranslationDataset(Dataset):
	
	def __init__(self, dataset, config:Config, tokenizer_source, tokenizer_target):
		super()
		self.dataset = dataset
		self.config = config
		self.tokenizer_source = tokenizer_source
		self.tokenizer_target = tokenizer_target

		self.language_source = config.data_config['lang_source']
		self.language_target = config.data_config['lang_target']
		self.max_tokens = config.train_config['max_sentence_tokens']

		self.s_token = torch.tensor([tokenizer_target.token_to_id("<S>")], dtype=torch.int64)
		self.e_token = torch.tensor([tokenizer_target.token_to_id("<E>")], dtype=torch.int64)
		self.p_token = torch.tensor([tokenizer_target.token_to_id("<P>")], dtype=torch.int64)
		self.u_token = torch.tensor([tokenizer_target.token_to_id("<U>")], dtype=torch.int64)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		# here we want to return one item of the dataset, however, encoded. So we make <S><encoded words><E><P>

		entry = self.dataset[index]
		#print(entry)
		# the data come as json, with an 'id' and 'translation' object on root, the latter containing the source and target texts.
		# e.g. {'id': '15590', 'translation': {'de': '»Die arme, liebe Frau!', 'en': '"Poor little woman!'}}

		# getting the individual texts
		text_source = entry['translation'][self.language_source]
		text_target = entry['translation'][self.language_target]

		# encoding them using the dictionary
		encoded_source = self.tokenizer_source.encode(text_source).ids
		encoded_target = self.tokenizer_target.encode(text_target).ids

		# encoder => <S><source text tokens><E><P><P>...<P> (until total length is reached)
		# decoder => <S><target text tokens><P><P>..<P> (until total length is reached)

		# check lengths first and throw error if too long
		if len(encoded_source) - 2 > self.max_tokens or len(encoded_target) - 1> self.max_tokens:
			raise OverflowError(f"Too many words, maximum is set to {self.max_tokens}! If more words are desired, change the limit in the configuration and retrain.")

		# calculate number of padding tokens required
		num_padding_encoder = self.max_tokens - len(encoded_source) - 2
		num_padding_decoder = self.max_tokens - len(encoded_target) - 1

		to_encoder = torch.cat([
			self.s_token, 
			torch.tensor(encoded_source, dtype=torch.int64), 
			self.e_token, 
			torch.tensor([self.p_token] * num_padding_encoder, dtype=torch.int64)
		])
		to_decoder = torch.cat([
			self.s_token,
			torch.tensor(encoded_target, dtype=torch.int64),
			torch.tensor([self.p_token] * num_padding_decoder, dtype=torch.int64)
		])
		label = torch.cat([
			torch.tensor(encoded_target, dtype=torch.int64),
			self.e_token,
			torch.tensor([self.p_token] * num_padding_decoder, dtype=torch.int64)
		])

		# check lengths
		assert to_encoder.size(0) == self.max_tokens
		assert to_decoder.size(0) == self.max_tokens
		assert label.size(0) == self.max_tokens

		mask = TranslationDataset.triangular_mask(to_decoder.size(0))

		return {
			'to_encoder': to_encoder,
			'to_decoder': to_decoder,
			'label': label,
			'text_source': text_source,
			'text_target': text_target,
			'mask_encoder': (to_encoder != self.p_token).unsqueeze(0).unsqueeze(0).int(),
			'mask_decoder': (to_decoder != self.p_token).unsqueeze(0).int() & mask,
		}
	
	# creates a square matrix of given size that has True on the main diagonal and lower triangle and False on the others
	def triangular_mask(size):
		mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
		return mask == 0
