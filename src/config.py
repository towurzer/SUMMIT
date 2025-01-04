#More things to be added here later for the model and training 
def get_config():
	return {
		"datasource": 'opus_books',
		"lang_source": "de",
		"lang_target": "en",
		"TRAIN_SIZE": 0.8,
		"VALIDATION_SIZE": 0.1,
		"MAX_SUPPORTED_SENTENCE_TOKEN_LENGTH": 500,
		"TOKENIZER_DIRECTORY": "train/tokenize", # directory for tokenizer caches
		"CHECKPOINT_DIRECTORY": "train/checkpoints", # directory for checkpoints
		"SEED": 69420, # seed for reproducible random results
		"BATCH_SIZE": 10, # how many items are part of one batch
		"LEARNING_RATE": 10e-4,
		"EPS": 1e-9,
	}
	