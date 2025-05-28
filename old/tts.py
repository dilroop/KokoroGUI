import numpy as np
import torch
from kokoro_onnx import Kokoro # Assuming kokoro_onnx is installed and accessible
import logging
import os
import requests
from tqdm import tqdm
import random
import string
import io
import soundfile as sf # For saving audio to a file
import sounddevice as sd # For playing audio
from datetime import datetime # Import datetime for timestamped filenames


# --- Configuration and Download Functions (from your provided code) ---
logger = logging.getLogger(__name__)

# Set a directory for models and voices, e.g., in the same directory as your script
# You can change this path if you prefer to store models elsewhere
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kokoro_models")
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure the directory exists

MODEL_URL = "https://github.com/taylorchu/kokoro-onnx/releases/download/v0.2.0/kokoro.onnx"
MODEL_FILENAME = "kokoro_v1.onnx"
VOICES_FILENAME = "voices_v1.bin"

# Hardcoded default speaker and language as requested
DEFAULT_SPEAKER = "am_liam"
DEFAULT_LANGUAGE = "English"
DEFAULT_SPEED = 1.5

supported_languages_display = ["English", "English (British)","French", "Japanese", "Hindi", "Mandarin Chinese", "Spanish", "Brazilian Portuguese", "Italian"]

supported_languages = {
    supported_languages_display[0]: "en-us",
    supported_languages_display[1]: "en-gb",
    supported_languages_display[2]: "fr-fr",
    supported_languages_display[3]: "ja",
    supported_languages_display[4]: "hi",
    supported_languages_display[5]: "cmn",
    supported_languages_display[6]: "es",
    supported_languages_display[7]: "pt-br",
    supported_languages_display[8]: "it",
}

supported_voices =[
    # American Female
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    #American Male
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
    # British Female
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    # British Male
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    # Japanese Female
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
    # Japanese Male
    "jm_kumo",

    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    "ef_dora", "em_alex", "em_santa", "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "pf_dora", "pm_alex", "pm_santa",
]

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

# Ensure models are present

def download_file(url, file_name, path):
    if not os.path.exists(path):
        os.makedirs(path)
    full_file_path = os.path.join(path, file_name)
    if os.path.exists(full_file_path):
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
    file_path = os.path.join(path, VOICES_FILENAME)
    if os.path.exists(file_path):
        return

    names = supported_voices
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
            voices[name] = data
        except Exception as e:
            logger.warning(f"Failed to download voice '{name}' from {url}: {e}")
            continue # Continue to the next voice if one fails

    if not voices:
        raise RuntimeError("No voices were successfully downloaded. Cannot create voices file.")

    with open(file_path, "wb") as f:
        np.savez(f, **voices)
    print(f"Created {file_path}")

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def ensure_kokoro_assets_exist(model_dir):
    download_file(MODEL_URL, MODEL_FILENAME, model_dir)
    download_voices_data(model_dir)

# --- Main Program Logic ---
class KokoroTTS:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, MODEL_FILENAME)
        self.voices_path = os.path.join(self.model_dir, VOICES_FILENAME)
        self.kokoro_instance = None
        self._load_kokoro_model()

    def _load_kokoro_model(self):
        ensure_kokoro_assets_exist(self.model_dir)
        try:
            self.kokoro_instance = Kokoro(model_path=self.model_path, voices_path=self.voices_path)
            print("Kokoro model initialized.")
        except Exception as e:
            logger.error(f"ERROR: Could not load kokoro-onnx: {e}")
            raise

    def get_speaker_embedding(self, speaker_name: str) -> dict:
        if self.kokoro_instance is None:
            self._load_kokoro_model()
        speaker_embedding: np.ndarray = self.kokoro_instance.get_voice_style(speaker_name)
        return {"speaker": speaker_embedding} # Match the dict format from ComfyUI node

    def generate_audio(self, text: str, speaker_data: dict, speed: float = 1.0, lang_display: str = "English") -> tuple[np.ndarray, int]:
        if self.kokoro_instance is None:
            self._load_kokoro_model()

        lang_code = supported_languages.get(lang_display, "en-us") # Default to en-us if not found

        try:
            print(f"Generating audio for text: '{text[:50]}...' with speaker '{speaker_data.get('name', 'unknown')}' and language '{lang_display}'...")
            audio_array, sample_rate = self.kokoro_instance.create(text, voice=speaker_data["speaker"], speed=speed, lang=lang_code)
            print("Audio generation complete.")
            if audio_array is None:
                raise ValueError("Kokoro returned no audio.")
            return audio_array, sample_rate
        except Exception as e:
            logger.error(f"Error during audio generation: {e}")
            raise

def play_audio(audio_array: np.ndarray, sample_rate: int):
    print("Playing audio...")
    try:
        sd.play(audio_array, samplerate=sample_rate)
        sd.wait() # Wait until playback is finished
        print("Audio playback finished.")
    except Exception as e:
        print(f"Error playing audio. ")

def save_audio_to_wav(audio_array: np.ndarray, sample_rate: int, output_filename: str = "output_audio.wav"):
    try:
        sf.write(output_filename, audio_array, sample_rate)
        print(f"Audio saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving audio to WAV: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        kokoro_tts = KokoroTTS(MODEL_DIR)
    except Exception as e:
        print(f"Failed to initialize Kokoro TTS. Exiting: {e}")
        exit()

    while True:
        print("Enter text to synthesize:")
        input_text = ""
        while True:
            line = input()
            if not line:
                break
            input_text += line + "\n"
        input_text = input_text.strip()

        if input_text.lower() == 'quit':
            break
        if not input_text:
            print("No text entered. Please try again.")
            continue

        # Use hardcoded default speaker and language
        chosen_voice_name = DEFAULT_SPEAKER
        chosen_lang_display = DEFAULT_LANGUAGE
        chosen_speed = DEFAULT_SPEED

        try:
            speaker_embedding = kokoro_tts.get_speaker_embedding(chosen_voice_name)
            speaker_embedding["name"] = chosen_voice_name

            # 2. Generate audio
            audio_data, sample_rate = kokoro_tts.generate_audio(input_text, speaker_embedding, chosen_speed, chosen_lang_display)

            # 3. Play audio
            play_audio(audio_data, sample_rate)

            # 4. Save audio for FFmpeg
            # output_wav_filename = "synthesized_audio.wav"
            output_wav_filename = generate_timestamp_filename()

            save_audio_to_wav(audio_data, sample_rate, output_wav_filename)

        except Exception as e:
            print(f"An error occurred during processing: {e}")

    print("Exiting Kokoro TTS program. Goodbye!")





