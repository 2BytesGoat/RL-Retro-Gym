import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        data = json.load(f)
    return data