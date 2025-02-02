from pathlib import Path
import json

# #More things to be added here later for the model and training 
# @DeprecationWarning("DO NOT USE ANYMORE!")
# def get_config():
# 	return {
# 		"datasource": 'opus_books',
# 		"lang_source": "de",
# 		"lang_target": "en",
# 		"TRAIN_SIZE": 0.8,
# 		"VALIDATION_SIZE": 0.1,
# 		"MAX_SUPPORTED_SENTENCE_TOKEN_LENGTH": 500,
# 		"MODEL_DIMENSIONS": 512,
# 		"NUM_ENCODER_BLOCKS": 6,
# 		"NUM_HEADS": 8,
# 		"DROPOUT": 0.1,
# 		"TRAIN_DIRECTORY": "train",
# 		"TOKENIZER_DIRECTORY": "tokenize", # directory for tokenizer caches
# 		"CHECKPOINT_DIRECTORY": "checkpoints", # directory for checkpoints
# 		"SEED": 69420, # seed for reproducible random results
# 		"BATCH_SIZE": 8, # how many items are part of one batch
# 		"LEARNING_RATE": 0.0001,
# 		"EPS": 1e-9,
# 		"EPOCHS" : 5
# 	}

class Config:
	def __init__(self, config_path):
		self.config_path = config_path

		self.__load_from_json()

		# TODO: Possible JSON Schema validation to config file
		
		# splitting config
		self.path_config = self.raw["paths"]
		self.data_config = self.raw["data"]
		self.train_config = self.raw["training"]
		self.app_config = self.raw["app"]

		self.check_or_create_folders()

	def __load_from_json(self):
		if not Path.exists(self.config_path):
			raise FileNotFoundError(f"Config file '{self.config_path.resolve()}' not found!")
		else:
			with open(self.config_path) as f:
					self.raw = json.load(f)


	# checking and creating folders
	def check_or_create_folders(self):
		self.dataset_folder = Path(self.path_config["root_directory"]) / Path(self.data_config["datasource"])
		if not Path.exists(self.dataset_folder): 
			self.dataset_folder.mkdir(parents = True)
		print(f"Base directory for model/training data: {str(self.dataset_folder)}") # e.g. train/opus_books

		self.tokenize_folder = self.dataset_folder / Path(self.path_config['sub_directory_tokenizer'])
		if not Path.exists(self.tokenize_folder): 
			self.tokenize_folder.mkdir(parents = True)
		print(f"Tokenize directory: {str(self.tokenize_folder)}") # e.g. train/opus_books/tokenize

		self.checkpoint_folder = self.dataset_folder / Path(self.path_config["sub_directory_checkpoints"])
		if not Path.exists(self.checkpoint_folder): 
			self.checkpoint_folder.mkdir(parents = True)
		print(f"Model/Checkpoint directory: {str(self.checkpoint_folder)}") #e.g. train/opus_books/checkpoints

	def get_latest_model_path(self):
		found_model_files = list(Path(self.checkpoint_folder).glob('*'))
		if len(found_model_files) > 0:
			found_model_files.sort(reverse=True)
			latest_model = found_model_files[0]
			print(f"Found latest model at: {latest_model}")
			return latest_model
		return None
	
