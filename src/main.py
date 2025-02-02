import sys
from pathlib import Path
import json
from config import Config

import torch
import pandas as pd

from model import Model

from app import create_app

print("=== SUMMIT ===")
print(
r""" 
                     ____  _     _      _      _  _____                     
_____ _____ _____   / ___\/ \ /\/ \__/|/ \__/|/ \/__ __\  _____ _____ _____ 
\____\\____\\____\  |    \| | ||| |\/||| |\/||| |  / \    \____\\____\\____\
_____ _____ _____   \___ || \_/|| |  ||| |  ||| |  | |    _____ _____ _____ 
\____\\____\\____\  \____/\____/\_/  \|\_/  \|\_/  \_/    \____\\____\\____\
                                                                            
""") # https://patorjk.com/software/taag

print("Launch settings: ", sys.argv)

config_file_path = Path('config.json')
config = Config(config_file_path)

# get device
print("Checking devices...")
device_str = "cpu"
if torch.cuda.is_available(): device_str = "cuda"
config.device = torch.device(device_str)
print(f"... found {device_str}")

print("Configuration: ", config.raw)

# check for training mode
if len(sys.argv) > 1 and sys.argv[1].strip() == "train":
	print("Launching training mode...")

else:
	print("Launching Web API...")

	# load latest state of the model
	print("Looking for old training states...")
	model = Model(config)
	model.load_latest_model(config)

	app = create_app(config, model)
	app.run(debug=config.app_config["debug"], port=config.app_config["port"])
