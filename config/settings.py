import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

if Path("/content").exists():
    MODELS_CACHE_DIR = Path("/content/FalconAI_Models")
else:
    MODELS_CACHE_DIR = BASE_DIR / "models_cache"

DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "inputs"
OUTPUT_DIR = DATA_DIR / "outputs"
TEMP_DIR = DATA_DIR / "temp"

for _dir in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "logs").mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = "fp16" if DEVICE == "cuda" else "fp32"

FACE_CONFIG = {
    "model_name": "buffalo_l",
    "det_thresh": 0.5,
    "min_face_size": 40,
    "ctx_id": 0 if DEVICE == "cuda" else -1,
    "detection_size": (640, 640),
}

LLM_CONFIG = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "model_cache_dir": str(MODELS_CACHE_DIR / "llm"),
    "max_new_tokens": 850,
    "temperature": 0.75,
    "top_p": 0.92,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "language": "Albanian",
    "num_scenes": 3,
}

DIFFUSION_CONFIG = {
    "model_name": "runwayml/stable-diffusion-v1-5",
    "model_cache_dir": str(MODELS_CACHE_DIR / "diffusion"),

    "ip_adapter_model": "h94/IP-Adapter",
    "ip_adapter_scale": 0.6,

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
    ),
}

ANIMATOR_CONFIG = {
    "model_name": "guoyww/animatediff-motion-adapter-v1-5-2",
    "model_cache_dir": str(MODELS_CACHE_DIR / "animatediff"),
    "num_frames": 16,
}

UPSCALER_CONFIG = {
    "model_name": "RealESRGAN_x4plus_anime_6B",
    "model_cache_dir": str(MODELS_CACHE_DIR / "esrgan"),
    "scale": 4,
}

AUDIO_CONFIG = {
    "tts_model": "tts_models/eng/cv/vits",
    "tts_speed": 1.0,
    "tts_cache_dir": str(MODELS_CACHE_DIR / "tts"),
    "music_volume": 0.3,
    "voice_volume": 1.0,
}

VIDEO_CONFIG = {
    "fps": 24,
    "resolution": (DIFFUSION_CONFIG["width"], DIFFUSION_CONFIG["height"]),
    "codec": "libx264",
    "audio_codec": "aac",
    "quality": 23,
}

PIPELINE_CONFIG = {
    "steps": [
        "face_processor",
        "story_generator",
        "audio_generator",
        "frame_generator",
        "video_assembler",
    ],
    "cleanup_temp": True,
    "save_locally": True,
}

LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s | %(levelname)s | %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_to_file": True,
    "log_file": str(BASE_DIR / "logs" / "falconai.log"),
}

INPUT_VALIDATION = {
    "max_name_length": 50,
    "max_interests_length": 200,
    "min_age_years": 1,
    "max_age_years": 16,
    "allowed_image_formats": [".jpg", ".jpeg", ".png", ".webp"],
    "max_image_size_mb": 10,
    "min_image_resolution": (256, 256),
}
