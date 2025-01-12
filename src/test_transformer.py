import pytest
import torch
from torch.utils.data import DataLoader

from dataset import TranslationDataset
from config import get_config
from transformer import TransformerBuilder
from train import DataSetLoader, Training
from datasets import load_dataset


@pytest.fixture
def config():
    """Fixture to load the configuration for the tests."""
    return get_config()


@pytest.fixture
def dataset_loader(config):
    """Fixture for loading datasets."""
    return DataSetLoader.get_dataset(config)


@pytest.fixture
def training_instance(config):
    """Fixture for creating a Training instance."""
    return Training(config)


def test_dataset_loading(dataset_loader):
    """Test dataset loading and splitting."""
    train_ds, val_ds, test_ds, tokenizer_source, tokenizer_target = dataset_loader

    assert len(train_ds) > 0, "Train dataset should not be empty."
    assert len(val_ds) > 0, "Validation dataset should not be empty."
    assert len(test_ds) > 0, "Test dataset should not be empty."

    # Check tokenizers
    assert tokenizer_source.get_vocab_size() > 0, "Source tokenizer should have a non-empty vocabulary."
    assert tokenizer_target.get_vocab_size() > 0, "Target tokenizer should have a non-empty vocabulary."


def test_tokenizer_creation(config):
    """Test tokenizer creation for both source and target languages."""
    dataset_raw = DataSetLoader.get_sentences(load_dataset(config['datasource'], 
                                                           f"{config['lang_source']}-{config['lang_target']}", split="train"), 
                                              config['lang_source'])
    tokenizer = DataSetLoader._DataSetLoader__load_tokenizer(config, dataset_raw, config['lang_source'])

    assert tokenizer.get_vocab_size() > 0, "Tokenizer should be trained and have a non-empty vocabulary."


def test_training_pipeline(training_instance):
    """Test the training process initialization."""
    assert training_instance.train_ds is not None, "Training dataset should be initialized."
    assert isinstance(training_instance.train_dataloader, DataLoader), "Train dataloader should be a valid DataLoader instance."
    assert isinstance(training_instance.validation_dataloader, DataLoader), "Validation dataloader should be a valid DataLoader instance."
    assert isinstance(training_instance.test_dataloader, DataLoader), "Test dataloader should be a valid DataLoader instance."


def test_model_forward_pass(training_instance):
    """Test if the model can perform a forward pass."""
    model = training_instance.model
    train_loader = training_instance.train_dataloader

    # Get a batch from the DataLoader
    batch = next(iter(train_loader))
    to_encoder = batch['to_encoder'].to(training_instance.device)
    mask_encoder = batch['mask_encoder'].to(training_instance.device)

    # Forward pass through encoder
    encoded = model.encode(to_encoder, mask_encoder)

    assert encoded is not None, "Encoded representation should not be None."
    assert encoded.size(0) == to_encoder.size(0), "Encoded batch size should match input batch size."


def test_validation_step(training_instance):
    """Test validation step for predictions."""
    training_instance.validation()
    # This test mainly ensures no errors occur in validation logic.


def test_save_and_load_model(training_instance):
    """Test saving and loading of the model."""
    model_path = training_instance.checkpoint_folder / "test_checkpoint.pt"

    # Save state
    torch.save({
        'epoch': 0,
        'global_step': 0,
        'model_states': training_instance.model.state_dict(),
        'optimizer_state': training_instance.optimizer.state_dict(),
    }, model_path)

    # Load state
    loaded_state = torch.load(model_path)
    training_instance.model.load_state_dict(loaded_state['model_states'])
    training_instance.optimizer.load_state_dict(loaded_state['optimizer_state'])

    assert model_path.exists(), "Checkpoint file should exist after saving."


def test_loss_function(training_instance):
    """Test if the loss function handles a batch correctly."""
    train_loader = training_instance.train_dataloader
    batch = next(iter(train_loader))

    to_encoder = batch['to_encoder'].to(training_instance.device)
    mask_encoder = batch['mask_encoder'].to(training_instance.device)
    to_decoder = batch['to_decoder'].to(training_instance.device)
    mask_decoder = batch['mask_decoder'].to(training_instance.device)
    label = batch['label'].to(training_instance.device)

    # Forward pass
    encoded = training_instance.model.encode(to_encoder, mask_encoder)
    decoded = training_instance.model.decode(encoded, mask_encoder, to_decoder, mask_decoder)
    projected = training_instance.model.project(decoded)

    # Compute loss
    loss = training_instance.loss_function(projected.view(-1, training_instance.tokenizer_target.get_vocab_size()), label.view(-1))
    assert loss.item() > 0, "Loss should be a positive value."
