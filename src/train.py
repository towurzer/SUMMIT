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

class DataSetLoader():

	@staticmethod
	def get_dataset(config):
	#split='train' is chosen because of errors occuring or the splits not being loaded when choosing another.
	#('train' is 1m lines while test and val are 2k each) https://huggingface.co/datasets/Helsinki-NLP/opus-100/viewer/de-en/train
		dataset_raw = load_dataset(f"{config['datasource']}", f"{config['lang_source']}-{config['lang_target']}", split='train')

	# tokenize
		tokenizer_source = DataSetLoader.__load_tokenizer(config, dataset_raw, config['lang_source'])
		tokenizer_target = DataSetLoader.__load_tokenizer(config, dataset_raw, config['lang_target'])

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
		folder = Path(config['TRAIN_DIRECTORY']) / Path(config['datasource']) / Path(config['TOKENIZER_DIRECTORY'])
		if not Path.exists(folder): 
			folder.mkdir(parents = True)
		file = folder / f"{language}.json"
		print(f"Looking for tokenizer file: {file.resolve()}")
		if Path.exists(file): 
			print("Found existing tokenizer file, reusing it!")
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
		self.batch_size = int(config['BATCH_SIZE'])
		self.epochs = int(config["EPOCHS"])

		# folders
		self.dataset_folder = Path(self.config["TRAIN_DIRECTORY"]) / Path(self.config["datasource"])
		if not Path.exists(self.dataset_folder): 
			self.dataset_folder.mkdir(parents = True)
		print(f"Base directory for model-related data: {str(self.dataset_folder)}")
		self.checkpoint_folder = self.dataset_folder / Path(self.config["CHECKPOINT_DIRECTORY"])
		if not Path.exists(self.checkpoint_folder): 
			self.checkpoint_folder.mkdir(parents = True)
		print(f"Checkpoint directory: {str(self.checkpoint_folder)}")

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
		self.train_ds, self.validation_ds, self.test_ds, self.tokenizer_source, self.tokenizer_target = DataSetLoader.get_dataset(get_config())

		# data points printed are the amount of sentence pairs
		print(f"Train dataset size: {len(self.train_ds)}")
		print(f"Validation dataset size: {len(self.validation_ds)}")
		print(f"Test dataset size: {len(self.test_ds)}\n")

		# print random example
		print(f"Example data entry: {self.train_ds[621]}\n")

		# dataloader
		print("Creating dataloaders...")
		self.train_dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
		self.validation_dataloader = DataLoader(self.validation_ds, batch_size=1, shuffle=True)
		self.test_dataloader = DataLoader(self.test_ds, batch_size=1, shuffle=True)

		print("Loading model")
		# TODO: make use of different configurations ?????
		self.model = TransformerBuilder.build_transformer(self.tokenizer_source.get_vocab_size(), self.tokenizer_target.get_vocab_size(), self.max_tokens, self.max_tokens, False, True).to(self.device)

		self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, eps = self.eps)

		print(self.model)


	def train_model(self, start_epoch = 0):
		self.epoch = 0
		self.global_step = 0

		self.loss_function = nn.CrossEntropyLoss(ignore_index=self.tokenizer_source.token_to_id('<P>'), label_smoothing=0.1).to(self.device)

		if start_epoch > 0:
			# TODO load existing state if epoch > 0
			pass
		
		print(f"Starting training at epoch {self.epoch}")

		while self.epoch < self.epochs:
			self.training_loop()
			self.epoch += 1


	def training_loop(self):
		# clear CUDA cache
		if torch.cuda.is_available(): torch.cuda.empty_cache()

		# set model to training mode
		self.model.train()

		# load batch
		for batch in self.train_dataloader:
			#print(batch)
			to_encoder = batch['to_encoder'].to(self.device)
			to_decoder = batch['to_decoder'].to(self.device)
			mask_encoder = batch['mask_encoder'].to(self.device)
			mask_decoder = batch['mask_decoder'].to(self.device)
			label = batch['label'].to(self.device)

			# for debug
			text_source = batch['text_source']
			text_target = batch['text_target']

			# encoder, decoder, projection
			encoded = self.model.encode(to_encoder, mask_encoder)
			decoded = self.model.decode(encoded, mask_encoder, to_decoder, mask_decoder)
			projected = self.model.project(decoded)

			# loss
			loss = self.loss_function(projected.view(-1, self.tokenizer_target.get_vocab_size()), label.view(-1))
			print(loss)

			# backpropagation
			loss.backward()

			# update weights
			self.optimizer.step()
			self.optimizer.zero_grad()

			self.global_step += 1

		# save iteration
		filename = self.checkpoint_folder / Path(f"{self.epoch:02d}")
		torch.save({
			'epoch': self.epoch,
			'global_step': self.global_step,
			'model_states': self.model.state_dict(),
			'optimizer_state': self.optimizer.state_dict(),
		}, filename)


trainer = Training(get_config())
trainer.train_model(0)
