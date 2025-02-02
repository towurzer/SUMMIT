#More things to be added here later for the model and training 
from pathlib import Path

def get_config():
	return {
		"datasource": 'opus_books',
		"lang_source": "de",
		"lang_target": "en",
		"TRAIN_SIZE": 0.8,
		"VALIDATION_SIZE": 0.1,
		"MAX_SUPPORTED_SENTENCE_TOKEN_LENGTH": 50,
		"MODEL_DIMENSIONS": 512,
		"NUM_ENCODER_BLOCKS": 6,
		"NUM_HEADS": 8,
		"DROPOUT": 0.1,
		"model_name": "00",
		"TRAIN_DIRECTORY": "train",
		"TOKENIZER_DIRECTORY": "tokenize", # directory for tokenizer caches
		"CHECKPOINT_DIRECTORY": "checkpoints", # directory for checkpoints
		"SEED": 69420, # seed for reproducible random results
		"BATCH_SIZE": 8, # how many items are part of one batch
		"LEARNING_RATE": 0.0001,
		"EPS": 1e-9,
		"EPOCHS" : 5
	}

def get_model_path(config, epoch: str):
	model_folder = f"{config['datasource']}\{config['CHECKPOINT_DIRECTORY']}"
	return str(Path('.') / model_folder / "model_name")

print(get_model_path(get_config(), 5))

