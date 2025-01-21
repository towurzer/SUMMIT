#More things to be added here later for the model and training 
def get_config():
	return {
		"datasource": 'opus_books',
		"lang_source": "de",
		"lang_target": "en",
		"TRAIN_SIZE": 0.8,
		"VALIDATION_SIZE": 0.1,
		"MAX_SUPPORTED_SENTENCE_TOKEN_LENGTH": 500,
		"MODEL_DIMENSIONS": 512,
		"NUM_ENCODER_BLOCKS": 6,
		"NUM_HEADS": 8,
		"DROPOUT": 0.1,
		"TRAIN_DIRECTORY": "train",
		"TOKENIZER_DIRECTORY": "tokenize", # directory for tokenizer caches
		"CHECKPOINT_DIRECTORY": "checkpoints", # directory for checkpoints
		"SEED": 69420, # seed for reproducible random results
		"BATCH_SIZE": 8, # how many items are part of one batch
		"LEARNING_RATE": 0.0001,
		"EPS": 1e-9,
		"EPOCHS" : 5
	}
	