#More things to be added here later for the model and training 
def get_config():
    return {
        "datasource": 'opus_books',
        "lang_source": "de",
        "lang_target": "en",
        "TRAIN_SIZE": 0.8,
        "VALIDATION_SIZE": 0.1,
    }
    