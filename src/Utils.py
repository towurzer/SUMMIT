from config import get_config
from pathlib import Path

def get_model_path(config, epoch: str):
	model_folder = f"{config['datasource']}{config['CHECKPOINT_DIRECTORY']}"
	return str(Path('.') / model_folder / "model_name")

print(Path('.').resolve())
