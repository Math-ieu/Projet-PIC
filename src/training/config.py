import yaml
import os

def load_training_config(config_path="configs/training_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_model_config(config_path="configs/model_configs.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
