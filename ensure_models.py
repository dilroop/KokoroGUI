
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

MODEL_FILENAME = "kokoro_v1.onnx"
VOICES_FILENAME = "voices_v1.bin"

def download_file(url, file_name, path):
    if not os.path.exists(path):
        os.makedirs(path)
    full_file_path = os.path.join(path, file_name)
    if os.path.exists(full_file_path):
        print(f"'{file_name}' already exists. Skipping download.")
        return
    print(f"Downloading {file_name} from {url}...")
    with requests.get(url, stream=True, allow_redirects=True) as response:
        response.raise_for_status() # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        block_size = 4096  # 4KB blocks
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name)
        with open(full_file_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    print(f"Downloaded '{file_name}' to {full_file_path}")

def download_voices_data(path):
    """Downloads individual voice files and saves them into a single .bin file."""
    file_path = os.path.join(path, VOICES_FILENAME)
    if os.path.exists(file_path):
        print(f"'{VOICES_FILENAME}' already exists. Skipping download.")
        return

    names = constants.supported_voices
    pattern = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{name}.pt"
    voices = {}

    print(f"Downloading individual voice files and compiling into '{VOICES_FILENAME}'...")
    for name in tqdm(names, desc="Downloading voices"):
        url = pattern.format(name=name)
        try:
            r = requests.get(url)
            r.raise_for_status()  # Ensure the request was successful
            content = io.BytesIO(r.content)
            # Use map_location='cpu' to load to CPU, preventing potential CUDA errors
            # if a GPU isn't available or configured for torch.
            data: np.ndarray = torch.load(content, map_location='cpu').numpy()
            voices.update({name: data}) # Use update for dictionary merging
        except Exception as e:
            logger.warning(f"Failed to download voice '{name}' from {url}: {e}")
            continue # Continue to the next voice if one fails

    if not voices:
        raise RuntimeError("No voices were successfully downloaded. Cannot create voices file.")

    with open(file_path, "wb") as f:
        np.savez(f, **voices)
    print(f"Created {file_path}")

def ensure_kokoro_assets_exist():
    """Ensures the Kokoro model and voices are downloaded."""
    download_file(constants.MODEL_URL, MODEL_FILENAME, constants.MODEL_DIR)
    download_voices_data(constants.MODEL_DIR)


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