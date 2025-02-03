import torch
import torch.nn as nn
import altair as alt
import pandas as pd
import numpy as np
import warnings
import tokenizer
import tokenizers
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
warnings.filterwarnings("ignore")

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.cuda.empty_cache()

from train import DataSetLoader, Training
from config import get_config
from transformer import Transformer
from transformer import TransformerBuilder


def load_model(config):
    print("=== SUMMIT Training Process ===\n")

    max_tokens = int(config['MAX_SUPPORTED_SENTENCE_TOKEN_LENGTH'])
    learning_rate = float(config['LEARNING_RATE'])
    eps = float(config['EPS'])
    seed = int(config['SEED'])
    batch_size = int(config['BATCH_SIZE'])
    epochs = int(config["EPOCHS"])

    # Folders
    dataset_folder = Path(config["TRAIN_DIRECTORY"]) / Path(config["datasource"])
    if not dataset_folder.exists():
        dataset_folder.mkdir(parents=True)
    print(f"Base directory for model-related data: {str(dataset_folder)}")
    
    checkpoint_folder = dataset_folder / Path(config["CHECKPOINT_DIRECTORY"])
    if not checkpoint_folder.exists():
        checkpoint_folder.mkdir(parents=True)
    print(f"Checkpoint directory: {str(checkpoint_folder)}")

    print("Checking devices...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device for training: {device}")
    
    print(f"Random seed: {seed}")
    torch.manual_seed(seed)

    print("Loading dataset...")
    train_ds, validation_ds, test_ds, tokenizer_source, tokenizer_target = DataSetLoader.get_dataset(config)
    print(f"Maximum token length found: {max_tokens}")
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(validation_ds)}")
    print(f"Test dataset size: {len(test_ds)}\n")
    print(f"Example data entry: {train_ds[621]}\n")

    print("Creating dataloaders...")
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_ds, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)

    print("Loading model")
    model = TransformerBuilder.build_transformer(
        tokenizer_source.get_vocab_size(), tokenizer_target.get_vocab_size(), max_tokens, max_tokens, False, True,
        config["MODEL_DIMENSIONS"], config["NUM_ENCODER_BLOCKS"], config["NUM_HEADS"], config["DROPOUT"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, eps=eps)
    old_train_files = list(checkpoint_folder.glob('*'))
    if old_train_files:
        old_train_files.sort(reverse=True)
        old_train_filename = old_train_files[0]
        print(f"Found latest model at: {old_train_filename}")
        state = torch.load(old_train_filename)
        model.load_state_dict(state['model_states'])
        optimizer.load_state_dict(state['optimizer_state'])
        epoch = state['epoch']
        print(f"Successfully loaded existing state, at epoch {epoch}")
    
    return model

config = get_config()
model = load_model(config)
trainer = Training(config)

def load_batch():
    batch = next(iter(trainer.validation_dataloader))
    encoder_input = batch["to_encoder"]
    decoder_input = batch["to_decoder"]
    encoder_mask = batch["mask_encoder"]
    decoder_mask = batch["mask_decoder"]

    encoder_input_tokens = [tokenizer_source.get_vocab_size().id_to_token(index) for index in encoder_input[0].cpu().numpy()]
    decoder_input_tokens = [tokenizer_target.get_vocab_size().id_to_token(index) for index in decoder_input[0].cpu().numpy()]

    assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
    return batch, encoder_input_tokens, decoder_input_tokens


def get_attn_map(attn_type: str, layer: int, head: int):
    if attn_type == "encoder":
        attn = model.encoder.encoder_module_list._modules['0'].self_attention_layer.attention_scores
    elif attn_type == "decoder":
        attn = model.decoder.decoder_module_list._modules['0'].self_attention_layer.attention_scores
    elif attn_type == "encoder-decoder":
        attn = model.decoder.decoder_module_list._modules['0'].cross_attention_layer.attention_scores
    return attn[0, head].data


def attn_map(attn_type, layer, head, row_tokens, col_tokens, max_sentence_len):
    df = mtx2df(get_attn_map(attn_type, layer, head), max_sentence_len, max_sentence_len, row_tokens, col_tokens)
    return (
        alt.Chart(data=df).mark_rect().encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400, title=f"Layer {layer} Head {head}")
        .interactive()
    )


def get_all_attention_maps(attn_type: str, layers: list, heads: list, row_tokens: list, col_tokens, max_sentence_len: int):
    charts = []
    for layer in layers:
        rowCharts = [attn_map(attn_type, layer, head, row_tokens, col_tokens, max_sentence_len) for head in heads]
        charts.append(alt.hconcat(*rowCharts))
    return alt.vconcat(*charts)

batch, encoder_input_tokens, decoder_input_tokens = load_batch()
print(f'Source: {batch["text_source"][0]}')
print(f'Target: {batch["text_target"][0]}')
sentence_len = encoder_input_tokens.index("[P]")

layers = [0, 1, 2]
heads = [0, 1, 2, 3, 4, 5, 6, 7]

get_all_attention_maps("encoder", layers, heads, encoder_input_tokens, encoder_input_tokens, min(20, sentence_len))
get_all_attention_maps("decoder", layers, heads, decoder_input_tokens, decoder_input_tokens, min(20, sentence_len))
get_all_attention_maps("encoder-decoder", layers, heads, encoder_input_tokens, decoder_input_tokens, min(20, sentence_len))
