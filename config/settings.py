import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_CACHE_DIR = Path("D:/FalconAI_Models")

DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "inputs"
OUTPUT_DIR = DATA_DIR / "outputs"
TEMP_DIR = DATA_DIR / "temp"

for _dir in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "logs").mkdir(exist_ok=True)

DEVICE = "cpu" 
TORCH_DTYPE = "fp32"

LLM_CONFIG = {
    "model_name": "microsoft/phi-2", 
    "model_cache_dir": str(MODELS_CACHE_DIR / "llm"),
    "max_new_tokens": 1200,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "language": "English",
    "num_scenes": 5,
}

DIFFUSION_CONFIG = {
    "model_name": "runwayml/stable-diffusion-v1-5",
    "model_cache_dir": str(MODELS_CACHE_DIR / "diffusion"),
    
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "seed": None,
    
    "style_prompt": (
        "children's book illustration style, pixar animation, vibrant colors, "
        "soft lighting, friendly characters, high quality, detailed, storybook art"
    ),
    "negative_prompt": (
        "ugly, blurry, dark, scary, violent, adult content, photo, realistic, "
        "bad anatomy, deformed, low quality, watermark, text, signature"
    )
}

AUDIO_CONFIG = {
    "tts_model": "tts_models/eng/cv/vits",
    "tts_speed": 1.0,
    "tts_cache_dir": str(MODELS_CACHE_DIR / "tts"),
    "music_volume": 0.3,
    "voice_volume": 1.0
}

PIPELINE_CONFIG = {
    "steps": [
        "story_generator",
        "image_generator",
        "audio_generator"
    ],
    "cleanup_temp": True,
    "save_locally": True
}

LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s | %(levelname)s | %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_to_file": True,
    "log_file": str(BASE_DIR / "logs" / "falconai.log")
}

INPUT_VALIDATION = {
    "max_name_length": 50,
    "max_interests_length": 200,
    "min_age_years": 1,
    "max_age_years": 16
}
