import yaml
import torch
import copy

def load_config(config_path):
    """ Load configuration from a YAML file

    Args:
        config_path (str): Path to the config YAML file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
