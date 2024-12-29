

#split='train' is chosen because of no split being loaded when no split is specified in this case due to something 
#train is 1m lines while test and val are 2k each) https://huggingface.co/datasets/Helsinki-NLP/opus-100/viewer/de-en/train
ds_raw = load_dataset('opus_books', 'de-en', split='train')

# Last line has to be written like this because the lengths otherwise do not match exactly
train_ds_size = int(0.8 * len(ds_raw))  
validation_ds_size = int(0.1 * len(ds_raw))  
test_ds_size = len(ds_raw) - train_ds_size - validation_ds_size  

train_ds_raw, validation_ds_raw, test_ds_raw = random_split(ds_raw, [train_ds_size, validation_ds_size, test_ds_size])

#Data points printed are the amount of sentence pairs
print(f"Train dataset size: {len(train_ds_raw)}")
print(f"Validation dataset size: {len(validation_ds_raw)}")
print(f"Test dataset size: {len(test_ds_raw)}")


