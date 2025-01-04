import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset

from config import get_config

from pathlib import Path

# tokenizers (using package)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# dataset
from dataset import TranslationDataset

# model
from transformer import TransformerBuilder

class DataLoader():

	@staticmethod
	def get_dataset(config):
	#split='train' is chosen because of errors occuring or the splits not being loaded when choosing another.
	#('train' is 1m lines while test and val are 2k each) https://huggingface.co/datasets/Helsinki-NLP/opus-100/viewer/de-en/train
		dataset_raw = load_dataset(f"{config['datasource']}", f"{config['lang_source']}-{config['lang_target']}", split='train')

	# tokenize
		tokenizer_source = DataLoader.__load_tokenizer(config, dataset_raw, config['lang_source'])
		tokenizer_target = DataLoader.__load_tokenizer(config, dataset_raw, config['lang_target'])

	# last line has to be written like this because the lengths otherwise do not match exactly and cause an error (splits do not overlap by function)
		train_ds_size = int(config['TRAIN_SIZE'] * len(dataset_raw))  
		validation_ds_size = int(config['VALIDATION_SIZE'] * len(dataset_raw))  
		test_ds_size = len(dataset_raw) - train_ds_size - validation_ds_size  

	# splitting into train/val/test datasets
		train_ds_raw, validation_ds_raw, test_ds_raw = random_split(dataset_raw, [train_ds_size, validation_ds_size, test_ds_size])

	# create dataset instances and return
		train_ds = TranslationDataset(train_ds_raw, config, tokenizer_source, tokenizer_target)
		validation_ds = TranslationDataset(validation_ds_raw, config, tokenizer_source, tokenizer_target)
		test_ds = TranslationDataset(test_ds_raw, config, tokenizer_source, tokenizer_target)
		
		return train_ds, validation_ds, test_ds, tokenizer_source, tokenizer_target
	
	
	def get_sentences(dataset, language):
		for item in dataset: yield item['translation'][language]

	# loads tokenizer from file, if none is found, it makes a new one
	
	def __load_tokenizer(config, dataset, language):
		folder = Path(config['TOKENIZER_DIRECTORY'])
		if not Path.exists(folder): 
			folder.mkdir(parents = True)
		file = folder / f"{language}.json"
		print(f"Looking for tokenizer files in: {file.resolve()}")
		if Path.exists(file): 
			print("found existing tokenizer files!")
			return Tokenizer.from_file(str(file)) #tokenizer already exists, just load
		else:
			print("Tokenizing...")
			tokenizer = Tokenizer(WordLevel(unk_token='<U>'))
			tokenizer.pre_tokenizer = Whitespace()
			trainer = WordLevelTrainer(special_tokens = ['<U>', '<S>', '<E>', '<P>']) # <U> = unknown words, <S> = start of sentence, <E> = end of sentence, <P> = padding to fill spaces
			tokenizer.train_from_iterator(DataLoader.get_sentences(dataset, language), trainer = trainer)
			tokenizer.save(path=str(file), pretty=True)
			return tokenizer

class Training():

	def __init__(self, config):
		# print some nice looking message
		print("=== SUMMIT Training Process ===\n")

		self.config = config
		self.max_tokens = int(config['MAX_SUPPORTED_SENTENCE_TOKEN_LENGTH'])
		self.learning_rate = float(config['LEARNING_RATE'])
		self.eps = float(config['EPS'])
		self.seed = int(config['SEED'])

		# get device
		print("Checking devices...")
		device_str = "cpu"
		if torch.cuda.is_available(): device_str = "cuda"
		self.device = torch.device(device_str)

		print(f"Device for training: {self.device}")

		# fix seed
		print(f"Random seed: {self.seed}")
		torch.manual_seed(self.seed)

		# get dataset
		print("Loading dataset...")
		train_ds, validation_ds, test_ds, tokenizer_source, tokenizer_target = DataLoader.get_dataset(get_config())

		# data points printed are the amount of sentence pairs
		print(f"Train dataset size: {len(train_ds)}")
		print(f"Validation dataset size: {len(validation_ds)}")
		print(f"Test dataset size: {len(test_ds)}\n")

		# print random example
		print(f"Example data entry: {train_ds[621]}")

		# todo: make use of different configurations ?????
		# self.model = TransformerBuilder.build_transformer(tokenizer_source.get_vocab_size(), tokenizer_target.get_vocab_size(), self.max_tokens, self.max_tokens, False, True)

		# self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, eps = self.eps)

		# print(self.model)


	def training_loop(self, start_epoch = 0):
		print(f"Starting training at epoch {start_epoch}")
		if torch.cuda.is_available(): torch.cuda.empty_cache()
		
		epoch = start_epoch
		global_step = 0

		# call train iteration here
		

		# save iteration
		# torch.save({
		# 	'epoch': epoch,
		# 	'global_step': global_step
		# 	'model_states': self.model.state_dict(),
		# 	'optimizer_state': self.optimizer.state_dict(),
		# }, )

		pass



trainer = Training(get_config())
