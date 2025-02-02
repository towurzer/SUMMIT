import torch
import pandas as pd

from dataset import TranslationDataset
from transformer import InputEmbeddings
from config import get_config

from pathlib import Path


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset

from config import get_config

from pathlib import Path

from tqdm import tqdm

from transformer import TransformerBuilder

# tokenizers (using package)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


torch.mean(torch.zeros(4))
print(torch.zeros(4))
print(torch.mean(torch.zeros(4)))
print(type(pd.DataFrame()))

print(f'has cuda? {torch.cuda.is_available()}')


def translate(self):
		s_token = self.tokenizer_target.token_to_id("<S>")
		e_token = self.tokenizer_target.token_to_id("<E>")

		with torch.no_grad():
			self.model.eval()
			repeats = 3
			counter = 0
			texts_source_lang = []
			texts_target_lang = []
			texts_predictions = []

			for batch in self.validation_dataloader:
				if counter >= repeats: break
				counter += 1

				to_encoder = batch['to_encoder'].to(self.device)
				#to_decoder = batch['to_decoder'].to(self.device)
				mask_encoder = batch['mask_encoder'].to(self.device)
				#mask_decoder = batch['mask_decoder'].to(self.device)
				label = batch['label'].to(self.device)

				text_source = batch['text_source']

				if to_encoder.size(0) > 1: raise ValueError("For evaluation dimension must be 1!")

				# decode
				encoded = self.model.encode(to_encoder, mask_encoder)
				to_decoder = torch.empty(1,1).fill_(s_token).type_as(to_encoder).to(self.device)

				for iteration in range(0, self.max_tokens):
					mask_decoder = TranslationDataset.triangular_mask(to_decoder.size(1)).type_as(mask_encoder).to(self.device)
					
					# get output
					output = self.model.decode(encoded, mask_encoder, to_decoder, mask_decoder)

					p = self.model.project(output[:, -1])
					_, most_likely = torch.max(p, dim=1)

					if most_likely == e_token: break # we reached the end
					
					# next run with old content to decode + most likely token
					to_decoder = torch.cat(
						[
							to_decoder, #last input
							torch.empty(1,1).type_as(to_encoder).fill_(most_likely.item()).to(self.device)
						], dim=1
					)
				
				# get the sentences back out from the tokens
				estimated = self.tokenizer_target.decode(to_decoder.squeeze(0).detach().cpu().numpy())

				# add to lists
				texts_source_lang.append(text_source)
				texts_predictions.append(estimated)

				# print for debug
				print(f"Source: {text_source}")
				print(f"Predict: {estimated}")
			#raise ValueError("AAAAA")


class DataSetLoader():

	@staticmethod
	def get_dataset(config):
		#split='train' is chosen because of errors occuring or the splits not being loaded when choosing another.
		#('train' is 1m lines while test and val are 2k each) https://huggingface.co/datasets/Helsinki-NLP/opus-100/viewer/de-en/train
		print("Loading raw dataset...")
		dataset_raw = load_dataset(f"{config['datasource']}", f"{config['lang_source']}-{config['lang_target']}", split='train')

		# tokenize
		print("Creating tokenizers...")
		tokenizer_source = DataSetLoader.__load_tokenizer(config, dataset_raw, config['lang_source'])
		tokenizer_target = DataSetLoader.__load_tokenizer(config, dataset_raw, config['lang_target'])

		# get maximum token count in sentence
		print("Finding longest items...")
		longest_source = 0
		longest_target = 0

		Threshold = config['MAX_SUPPORTED_SENTENCE_TOKEN_LENGTH']-2
		exclude = []

		for entry in dataset_raw:
			encoded_source = tokenizer_source.encode(entry['translation'][config['lang_source']])
			encoded_target = tokenizer_target.encode(entry['translation'][config['lang_target']])
			
			if(len(encoded_source.ids) >= Threshold or len(encoded_target.ids) >= Threshold):
				exclude.append(entry)

			longest_source = max(longest_source, len(encoded_source.ids))
			longest_target = max(longest_target, len(encoded_target.ids))	
			

		print(f"Longest items found: {config['lang_source']}: {longest_source}, {config['lang_target']}: {longest_target}")
		print(f"number of rows in raw dataset: {dataset_raw.num_rows}")
		print(f"number of items above a certain number): {len(exclude)}" )

		dataset_raw2 = dataset_raw.filter(lambda example: example not in exclude)
		print(f"number of rows in raw dataset2: {dataset_raw2.num_rows}")

		longest_source = 0
		longest_target = 0

		for entry in dataset_raw2:
			encoded_source = tokenizer_source.encode(entry['translation'][config['lang_source']])
			encoded_target = tokenizer_target.encode(entry['translation'][config['lang_target']])

			longest_source = max(longest_source, len(encoded_source.ids))
			longest_target = max(longest_target, len(encoded_target.ids))	

		print(f"New longest items found: {config['lang_source']}: {longest_source}, {config['lang_target']}: {longest_target}")

		print(f"Dataset reduced by {dataset_raw.num_rows/dataset_raw2.num_rows*100}%")

		dataset_raw = dataset_raw2
		
		# last line has to be written like this because the lengths otherwise do not match exactly and cause an error (splits do not overlap by function)
		print("Splitting dataset...")
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
	
	def get_model(config, vocab_source_len, vocab_target_len):
		model = build_transformer(vocab_source_len, vocab_target_len, config["MAX_SUPPORTED_SENTENCE_TOKEN_LENGTH"], config
		['MAX_SUPPORTED_SENTENCE_TOKEN_LENGTH'], d_model=config['MODEL_DIMENSIONS'])
		return model

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
			tokenizer.train_from_iterator(DataSetLoader.get_sentences(dataset, language), trainer = trainer)
			tokenizer.save(path=str(file), pretty=True)
			return tokenizer


class Translate():

	def __init__(self, config):

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
		self.train_ds, self.validation_ds, self.test_ds, self.tokenizer_source, self.tokenizer_target = DataSetLoader.get_dataset(self.config)

		print(f"Maximum token length found: {self.max_tokens}")

		# data points printed are the amount of sentence pairs
		print(f"Train dataset size: {len(self.train_ds)}")
		print(f"Validation dataset size: {len(self.validation_ds)}")
		print(f"Test dataset size: {len(self.test_ds)}\n")

		# print random example
		print(f"Example data entry: {self.train_ds[621]}\n")

		# dataloader
		print("Creating dataloaders...")
		self.train_dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)


		print("Loading model")
		# TODO: make use of different configurations ?????
		self.model = TransformerBuilder.build_transformer(self.tokenizer_source.get_vocab_size(), self.tokenizer_target.get_vocab_size(), self.max_tokens, self.max_tokens, False, True, self.config["MODEL_DIMENSIONS"], self.config["NUM_ENCODER_BLOCKS"], self.config["NUM_HEADS"], self.config["DROPOUT"]).to(self.device)

		self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, eps = self.eps)

		print(self.model)

	def train_model(self):
		self.epoch = 0
		self.global_step = 0

		self.loss_function = nn.CrossEntropyLoss(ignore_index=self.tokenizer_source.token_to_id('<P>'), label_smoothing=0.1).to(self.device)

		# check for existing training state (finished epochs)
		print("Looking for old training states...")
		old_train_files = list(Path(self.checkpoint_folder).glob('*'))
		if len(old_train_files) > 0:
			old_train_files.sort(reverse=True)
			old_train_filename = old_train_files[0]
			print(f"Found latest model at: {old_train_filename}")
		
			state = torch.load(old_train_filename)
			self.model.load_state_dict(state['model_states'])
			self.optimizer.load_state_dict(state['optimizer_state'])
			self.global_step = state['global_step']

			print(f"Successfully loaded existing model")
		else :
			print(f"There is no model.")


		# starting training
		print(f"Starting training at epoch {self.epoch}")
		while self.epoch < self.epochs:
			print(f"--- Epoch {self.epoch} ---")
			self.training_loop()
			self.validation()
			self.epoch += 1


	def training_loop(self):
		# clear CUDA cache
		if torch.cuda.is_available(): torch.cuda.empty_cache()

		# set model to training mode
		self.model.train()
		

		# iterator with tqdm progress bar
		batch_iterator = tqdm(self.train_dataloader, desc=f"Processing Epoch {self.epoch:02d}")

		# load batch
		for batch in batch_iterator:
			#print(batch)
			to_encoder = batch['to_encoder'].to(self.device)
			to_decoder = batch['to_decoder'].to(self.device)
			mask_encoder = batch['mask_encoder'].to(self.device)
			mask_decoder = batch['mask_decoder'].to(self.device)


			# encoder, decoder, projection
			encoded = self.model.encode(to_encoder, mask_encoder)
			decoded = self.model.decode(encoded, mask_encoder, to_decoder, mask_decoder)
			projected = self.model.project(decoded)

			# update weights
			self.optimizer.step()
			self.optimizer.zero_grad()




	def validation(self):
		s_token = self.tokenizer_target.token_to_id("<S>")
		e_token = self.tokenizer_target.token_to_id("<E>")

		with torch.no_grad():
			self.model.eval()
			repeats = 3
			counter = 0
			texts_source_lang = []
			texts_target_lang = []
			texts_predictions = []

			for batch in self.validation_dataloader:
				if counter >= repeats: break
				counter += 1

				to_encoder = batch['to_encoder'].to(self.device)
				#to_decoder = batch['to_decoder'].to(self.device)
				mask_encoder = batch['mask_encoder'].to(self.device)
				#mask_decoder = batch['mask_decoder'].to(self.device)
				label = batch['label'].to(self.device)

				text_source = batch['text_source']
				text_target = batch['text_target']

				if to_encoder.size(0) > 1: raise ValueError("For evaluation dimension must be 1!")

				# decode
				encoded = self.model.encode(to_encoder, mask_encoder)
				to_decoder = torch.empty(1,1).fill_(s_token).type_as(to_encoder).to(self.device)

				for iteration in range(0, self.max_tokens):
					mask_decoder = TranslationDataset.triangular_mask(to_decoder.size(1)).type_as(mask_encoder).to(self.device)
					
					# get output
					output = self.model.decode(encoded, mask_encoder, to_decoder, mask_decoder)

					p = self.model.project(output[:, -1])
					_, most_likely = torch.max(p, dim=1)

					if most_likely == e_token: break # we reached the end
					
					# next run with old content to decode + most likely token
					to_decoder = torch.cat(
						[
							to_decoder, #last input
							torch.empty(1,1).type_as(to_encoder).fill_(most_likely.item()).to(self.device)
						], dim=1
					)
				
				# get the sentences back out from the tokens
				estimated = self.tokenizer_target.decode(to_decoder.squeeze(0).detach().cpu().numpy())

				# add to lists
				texts_source_lang.append(text_source)
				texts_target_lang.append(text_target)
				texts_predictions.append(estimated)

				# print for debug
				print(f"Source: {text_source}")
				print(f"Target: {text_target}")
				print(f"Predict: {estimated}")


val = input("Enter your value: ")
print(val)
translate(val)





translater = Translate(get_config())
translater.train_model()
