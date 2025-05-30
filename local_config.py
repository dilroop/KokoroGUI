import os
import json
import constants

# Hardcoded default speaker and language
DEFAULT_SPEAKER_KEY = "am_liam" # Use the key
DEFAULT_LANGUAGE_DISPLAY = "English" # Use the display name
CONFIG_FILE = os.path.join(constants.MODELS_DIR, "gui_config.json") # File for persistent settings

# --- Configuration Persistence Functions ---
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)