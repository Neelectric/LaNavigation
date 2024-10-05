
import yaml

# Function to load the YAML configuration file
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_key(key, file='Neel_config.yaml'):
    config = load_config(file)

    # Access the API keys
    key_val = config['api_keys'][key]
    return key_val