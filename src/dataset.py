import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset 
from config import get_config
   
def get_dataset(config):
#split='train' is chosen because of errors occuring or the splits not being loaded when choosing another
#('train' is 1m lines while test and val are 2k each) https://huggingface.co/datasets/Helsinki-NLP/opus-100/viewer/de-en/train
    dataset = load_dataset(f"{config['datasource']}", f"{config['lang_source']}-{config['lang_target']}", split='train')

# Last line has to be written like this because the lengths otherwise do not match exactly and cause an error (splits do not overlap by function)
    train_ds_size = int(0.8 * len(dataset))  
    validation_ds_size = int(0.1 * len(dataset))  
    test_ds_size = len(dataset) - train_ds_size - validation_ds_size  

#Splitting into train/val/test datasets
    train_ds, validation_ds, test_ds = random_split(dataset, [train_ds_size, validation_ds_size, test_ds_size])

    return train_ds, validation_ds, test_ds


train_ds, validation_ds, test_ds = get_dataset(get_config())

#Data points printed are the amount of sentence pairs
print(f"Train dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(validation_ds)}")
print(f"Test dataset size: {len(test_ds)}\n")

#Checking it is really loaded
for i in range(3):
    print(f"{i+1}: {train_ds[i]}")


