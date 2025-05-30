
import numpy as np
import torch
import os
import logging
import requests
from tqdm import tqdm
import io
from datetime import datetime
import constants

# --- Configuration and Download Functions ---
logger = logging.getLogger(__name__)

def download_file():
    # Directory: kokoro_models
    modelsDir = constants.MODELS_DIR
    # Models Path: kokoro_models/kokoro.onnx
    modelPath = constants.MODEL_PATH
    # URL for the model
    modelUrl = constants.MODEL_URL
    
    if not os.path.exists(modelsDir):
        os.makedirs(modelsDir)
 
    if os.path.exists(modelPath):
        print(f"'{modelPath}' already exists. Skipping download.")
        return
    
    print(f"Downloading {modelPath} from {modelUrl}...")
    with requests.get(modelUrl, stream=True, allow_redirects=True) as response:
        response.raise_for_status() # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        block_size = 4096  # 4KB blocks
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=modelPath)
        with open(modelPath, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    print(f"Downloaded at {modelPath}")

def download_voices_data():
    # Check if the voices file already exists
    if os.path.exists(constants.VOICES_PATH):
        print(f"'{constants.VOICES_PATH}' already exists. Skipping download.")
        return

    names = constants.SUPPORTED_VOICES
    pattern = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{name}.pt"
    voices = {}

    for name in tqdm(names, desc="Downloading voices"):
        url = pattern.format(name=name)
        try:
            r = requests.get(url)
            r.raise_for_status()  # Ensure the request was successful
            content = io.BytesIO(r.content)
            # Use map_location='cpu' to load to CPU, preventing potential CUDA errors
            # if a GPU isn't available or configured for torch.
            data: np.ndarray = torch.load(content, map_location='cpu').numpy()
            voices.update({name: data})
        except Exception as e:
            logger.warning(f"Failed to download voice '{name}' from {url}: {e}")
            continue

    if not voices:
        raise RuntimeError("No voices were successfully downloaded. Cannot create voices file.")

    with open(constants.VOICES_PATH, "wb") as f:
        np.savez(f, **voices)
    print(f"Created {constants.VOICES_PATH}")

def ensure_kokoro_assets_exist():
    download_file()
    download_voices_data()


def generate_timestamp_filename(prefix: str = "synthesized_audio", extension: str = ".wav") -> str:
    # %I for 12-hour clock, %p for AM/PM, %d for day, %B for full month name
    timestamp = datetime.now().strftime("%I%p %dth %B").lower() # .lower() to make "pm" lowercase
    # Remove leading zero for hours if present and handle 'st', 'nd', 'rd', 'th' for day
    if timestamp[0] == '0':
        timestamp = timestamp[1:]

    day_str = datetime.now().day
    if 10 < day_str < 20: # Handles 11th, 12th, 13th, etc.
        day_suffix = "th"
    else:
        day_suffix = {1: "st", 2: "nd", 3: "rd"}.get(day_str % 10, "th")

    # Replace the default 'th' with the correct suffix
    timestamp = timestamp.replace("th", day_suffix, 1) # Only replace the first 'th'

    return f"{prefix} - {timestamp}{extension}"