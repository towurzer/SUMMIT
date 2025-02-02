from pathlib import Path
import torch
import json

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

		# get device
		print("Checking devices...")
		device_str = "cpu"
		if torch.cuda.is_available(): device_str = "cuda"
		self.device = torch.device(device_str)
		print(f"... found {device_str}")

		# set seed
		print(f"Seed: {self.train_config['seed']}")
		torch.manual_seed(self.train_config["seed"])

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
		print(f"Checkpoint directory: {str(self.checkpoint_folder)}") #e.g. train/opus_books/checkpoints

		self.model_folder = self.dataset_folder / Path(self.path_config["sub_directory_model"])
		if not Path.exists(self.model_folder): 
			self.model_folder.mkdir(parents = True)
		print(f"Model directory: {str(self.model_folder)}") #e.g. train/opus_books/model

	def get_latest_model_path(self):
		found_model_files = list(Path(self.model_folder).glob('*'))
		if len(found_model_files) > 0:
			found_model_files.sort(reverse=True)
			latest_model = found_model_files[0]
			print(f"Found latest model at: {latest_model}")
			return latest_model
		return None
	
