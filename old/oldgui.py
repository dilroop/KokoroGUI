import numpy as np
import torch
from kokoro_onnx import Kokoro # Assuming kokoro_onnx is installed and accessible
import logging
import os
import requests
from tqdm import tqdm
import io
import soundfile as sf # For saving audio to a file
import sounddevice as sd # For playing audio
from datetime import datetime # Import datetime for timestamped filenames
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, ttk # Import ttk for themed widgets
import json # For saving and loading configuration

# --- Configuration and Download Functions ---
logger = logging.getLogger(__name__)

# Set a directory for models and voices, e.g., in the same directory as your script
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kokoro_models")
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure the directory exists

MODEL_URL = "https://github.com/taylorchu/kokoro-onnx/releases/download/v0.2.0/kokoro.onnx"
MODEL_FILENAME = "kokoro_v1.onnx"
VOICES_FILENAME = "voices_v1.bin"

# Hardcoded default speaker and language
DEFAULT_SPEAKER_KEY = "am_liam" # Use the key
DEFAULT_LANGUAGE_DISPLAY = "English" # Use the display name
CONFIG_FILE = "tts_config.json" # File for persistent settings

supported_languages_display = ["English", "English (British)","French", "Japanese", "Hindi", "Mandarin Chinese", "Spanish", "Brazilian Portuguese", "Italian"]

supported_languages = {
    "English": "en-us",
    "English (British)": "en-gb",
    "French": "fr-fr",
    "Japanese": "ja",
    "Hindi": "hi",
    "Mandarin Chinese": "cmn",
    "Spanish": "es",
    "Brazilian Portuguese": "pt-br",
    "Italian": "it",
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
            voices.update({name: data}) # Use update for dictionary merging
        except Exception as e:
            logger.warning(f"Failed to download voice '{name}' from {url}: {e}")
            continue # Continue to the next voice if one fails

    if not voices:
        raise RuntimeError("No voices were successfully downloaded. Cannot create voices file.")

    with open(file_path, "wb") as f:
        np.savez(f, **voices)
    print(f"Created {file_path}")

def ensure_kokoro_assets_exist(model_dir):
    """Ensures the Kokoro model and voices are downloaded."""
    download_file(MODEL_URL, MODEL_FILENAME, model_dir)
    download_voices_data(model_dir)

# --- Configuration Persistence Functions ---
def load_config():
    """Loads configuration from a JSON file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {} # Return empty if file is corrupted
    return {}

def save_config(config):
    """Saves configuration to a JSON file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# --- Main Program Logic ---

class KokoroTTS:
    """A wrapper class to manage Kokoro model loading and text-to-speech generation."""
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, MODEL_FILENAME)
        self.voices_path = os.path.join(self.model_dir, VOICES_FILENAME)
        self.kokoro_instance = None
        self._load_kokoro_model()

    def _load_kokoro_model(self):
        """Loads the Kokoro model and voices."""
        ensure_kokoro_assets_exist(self.model_dir)
        try:
            print("Initializing Kokoro model...")
            self.kokoro_instance = Kokoro(model_path=self.model_path, voices_path=self.voices_path)
            print("Kokoro model initialized.")
        except Exception as e:
            logger.error(f"ERROR: Could not load kokoro-onnx: {e}")
            raise

    def get_speaker_embedding(self, speaker_name: str) -> dict:
        """Retrieves the speaker embedding for a given speaker name."""
        if self.kokoro_instance is None:
            self._load_kokoro_model()
        speaker_embedding: np.ndarray = self.kokoro_instance.get_voice_style(speaker_name)
        return {"speaker": speaker_embedding} # Match the dict format from ComfyUI node

    def generate_audio(self, text: str, speaker_data: dict, speed: float = 1.0, lang_display: str = "English") -> tuple:
        """Generates audio from text using the specified speaker, speed, and language."""
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
    """Plays the given audio array."""
    print("Playing audio...")
    try:
        sd.play(audio_array, samplerate=sample_rate)
        
        print("Audio playback finished.")
    except Exception as e:
        print(f"Error playing audio. Make sure you have an audio device and 'sounddevice' is configured: {e}")
        print("You can try installing sounddevice with 'pip install sounddevice'.")

def stop_audio():
    print("Stop audio...")
    try:
        sd.stop()
        print("Audio playback stopped.")
    except Exception as e:
        print(f"Error stopping audio playback: {e}")
        print("You can try installing sounddevice with 'pip install sounddevice'.")

def save_audio_to_wav(audio_array: np.ndarray, sample_rate: int, output_filename: str):
    """Saves the audio array to a WAV file."""
    try:
        sf.write(output_filename, audio_array, sample_rate)
        print(f"Audio saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving audio to WAV: {e}")

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

# --- GUI Application ---
class TTSApp:
    def __init__(self, master):
        self.master = master
        master.title("Kokoro TTS Generator")
        master.geometry("650x600") # Adjust window size

        # Use the 'clam' theme for better button contrast (you can try others like 'alt', 'default', 'classic')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', padding=10, font=('Arial', 12))
        style.configure('TLabel', font=('Arial', 10))
        style.configure('TLabelframe.Label', font=('Arial', 10, 'bold'))

        # Configure logging for GUI
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        self.kokoro_tts = None
        self.last_audio_data = None
        self.last_sample_rate = None
        self.last_output_filename = None # Store the filename for FFmpeg comments

        # Load persistent configuration
        self.config = load_config()
        self.initial_speed = self.config.get('speed', 1.0) # Default to 1.0 if not found
        self.initial_language = self.config.get('language', DEFAULT_LANGUAGE_DISPLAY)
        self.initial_speaker = self.config.get('speaker', DEFAULT_SPEAKER_KEY)

        self.selected_language = tk.StringVar(master)
        self.selected_language.set(self.initial_language) # Set default language
        self.selected_speaker = tk.StringVar(master)
        self.selected_speaker.set(self.initial_speaker) # Set default speaker

        # Initialize Kokoro TTS
        try:
            self.kokoro_tts = KokoroTTS(MODEL_DIR)
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize Kokoro TTS: {e}\n"
                                                        "Please check your internet connection and ensure dependencies are installed.")
            master.destroy()
            return

        self._create_widgets()
        master.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle window close event

    def _create_widgets(self):
        # Text Input Frame
        text_frame = ttk.Labelframe(self.master, text="Text to Synthesize", padding=10)
        text_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.text_input = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, width=60, height=10, font=("Arial", 10))
        self.text_input.pack(fill="both", expand=True)
        self.text_input.insert(tk.END, "Hello, this is a test of the Kokoro text-to-speech system.")

        # Settings Frame
        settings_frame = ttk.Labelframe(self.master, text="Settings", padding=10)
        settings_frame.pack(padx=10, pady=10, fill="x")

        # Language Selection
        language_label = ttk.Label(settings_frame, text="Language:")
        language_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.language_dropdown = ttk.Combobox(settings_frame, textvariable=self.selected_language, values=supported_languages_display, state="readonly")
        self.language_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Speaker Selection
        speaker_label = ttk.Label(settings_frame, text="Speaker:")
        speaker_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.speaker_dropdown = ttk.Combobox(settings_frame, textvariable=self.selected_speaker, values=supported_voices, state="readonly")
        self.speaker_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Set initial values for dropdowns
        self.language_dropdown.set(self.initial_language)
        self.speaker_dropdown.set(self.initial_speaker)

        # Speed Input
        speed_label = ttk.Label(settings_frame, text="Speech Speed (0.1-4.0):")
        speed_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.speed_entry = ttk.Entry(settings_frame, width=10, font=("Arial", 10))
        self.speed_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.speed_entry.insert(0, str(self.initial_speed)) # Set initial value from config

        # Buttons Frame
        button_frame = tk.Frame(self.master, padx=10, pady=10)
        button_frame.pack(padx=10, pady=10, fill="x")

        self.generate_button = ttk.Button(button_frame, text="Generate & Play Audio", command=self.generate_and_play_audio)
        self.generate_button.pack(side=tk.LEFT, padx=5, expand=True, fill="x")

        self.play_button = ttk.Button(button_frame, text="Play Last Generated Audio", command=self.play_last_audio, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=5, expand=True, fill="x")
        
        self.stop_button = ttk.Button(button_frame, text="Stop Playing", command=self.stop_last_audio, state=tk.ACTIVE)
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def generate_and_play_audio(self):
        input_text = self.text_input.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Input Error", "Please enter some text to synthesize.")
            return

        try:
            speech_speed = float(self.speed_entry.get())
            if not (0.1 <= speech_speed <= 4.0):
                messagebox.showwarning("Input Error", "Speed must be between 0.1 and 4.0.")
                return
        except ValueError:
            messagebox.showwarning("Input Error", "Invalid speed. Please enter a number.")
            return

        selected_lang_display = self.selected_language.get()
        selected_speaker_key = self.selected_speaker.get()

        # Save current settings to config
        self.config['speed'] = speech_speed
        self.config['language'] = selected_lang_display
        self.config['speaker'] = selected_speaker_key
        save_config(self.config)

        # Disable buttons during generation
        self.generate_button.config(state=tk.DISABLED, text="Generating...")
        self.play_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.master.update_idletasks() # Update GUI immediately

        try:
            # 1. Get speaker embedding
            speaker_embedding = self.kokoro_tts.get_speaker_embedding(selected_speaker_key)
            speaker_embedding["name"] = selected_speaker_key # Add name for logging clarity

            # 2. Generate audio
            audio_data, sample_rate = self.kokoro_tts.generate_audio(input_text, speaker_embedding, speech_speed, selected_lang_display)

            self.last_audio_data = audio_data
            self.last_sample_rate = sample_rate

            # 3. Play audio
            play_audio(audio_data, sample_rate)
            self.stop_button.config(state=tk.NORMAL) # Enable stop button

            # 4. Save audio for FFmpeg using the new timestamp function
            self.last_output_filename = generate_timestamp_filename()
            save_audio_to_wav(audio_data, sample_rate, self.last_output_filename)

            # Enable play button
            self.play_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Generation Error", f"An error occurred during audio generation: {e}")
            logger.error(f"Error in generate_and_play_audio: {e}")
        finally:
            self.generate_button.config(state=tk.NORMAL, text="Generate & Play Audio") # Re-enable button

    def play_last_audio(self):
        if self.last_audio_data is not None and self.last_sample_rate is not None:
            try:
                play_audio(self.last_audio_data, self.last_sample_rate)
            except Exception as e:
                messagebox.showerror("Playback Error", f"An error occurred during playback: {e}")
                logger.error(f"Error in play_last_audio: {e}")
        else:
            messagebox.showinfo("No Audio", "No audio has been generated yet. Please generate audio first.")

    def stop_last_audio(self):
        stop_audio()

    def on_closing(self):
        """Saves configuration before closing the window."""
        try:
            self.config['speed'] = float(self.speed_entry.get()) # Ensure latest speed is saved
            self.config['language'] = self.selected_language.get()
            self.config['speaker'] = self.selected_speaker.get()
            save_config(self.config)
        except ValueError:
            messagebox.showerror("Configuration Error", "Could not save configuration. Invalid speed value.")
        self.master.destroy()

# --- Main Execution Block ---
if __name__ == "__main__":
    root = tk.Tk()
    app = TTSApp(root)
    root.mainloop()