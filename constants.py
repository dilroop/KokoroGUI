import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(PROJECT_DIR, "kokoro_models")
MODEL_URL = "https://github.com/taylorchu/kokoro-onnx/releases/download/v0.2.0/kokoro.onnx"
_MODEL_FILENAME = "kokoro_v1.onnx"
_VOICES_FILENAME = "voices_v1.bin"

MODEL_PATH = os.path.join(MODELS_DIR, _MODEL_FILENAME)
VOICES_PATH = os.path.join(MODELS_DIR, _VOICES_FILENAME)

SUPPORTED_LANGUAGES_DISPLAY = ["English", "English (British)","French", "Japanese", "Hindi", "Mandarin Chinese", "Spanish", "Brazilian Portuguese", "Italian"]

SUPPORTED_LANGUAGES = {
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

SUPPORTED_VOICES =[
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