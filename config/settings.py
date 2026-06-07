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

DEVICE = "cpu" 
TORCH_DTYPE = "fp32"

FACE_CONFIG = {
    "model_name": "buffalo_l",
    "detection_size": (640, 640),
    "det_thresh": 0.5,
    "min_face_size": 50,
    "ctx_id": -1
}

LLM_CONFIG = {
    "model_name": "microsoft/phi-2", 
    "model_cache_dir": str(MODELS_CACHE_DIR / "llm"),

    "max_new_tokens": 1200,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True,

    "language": "Albanian",

    "num_scenes": 5,
    "scene_duration_sec": 4
}

DIFFUSION_CONFIG = {
    "model_name": "runwayml/stable-diffusion-v1-5",
    "model_cache_dir": str(MODELS_CACHE_DIR / "diffusion"),

    "ip_adapter_model": "h94/IP-Adapter",
    "ip_adapter_scale": 0.7,

    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "seed": None,

    "style_prompt": (
        "children's movie style, pixar animation, vibrant colors, "
        "soft lighting, friendly characters, high quality, detailed"
    ),
    "negative_prompt": (
        "ugly, blurry, dark, scary, violent, adult content, "
        "bad anatomy, deformed, low quality, watermark"
    )
}

ANIMATOR_CONFIG = {
    "model_name": "guoyww/animatediff-motion-adapter-v1-5-2",
    "model_cache_dir": str(MODELS_CACHE_DIR / "animator"),
    "num_frames": 16,
    "fps": 8
}

AUDIO_CONFIG = {
    "tts_model": "tts_models/eng/cv/vits",
    "tts_speed": 1.0,
    "tts_cache_dir": str(MODELS_CACHE_DIR / "tts"),

    "music_volume": 0.3,
    "voice_volume": 1.0
}

VIDEO_CONFIG = {
    "fps": 24,
    "output_format": "mp4",
    "codec": "libx264",
    "audio_codec": "aac",
    "quality": 23,
    "resolution": (1280, 720)
}

UPSCALER_CONFIG = {
    "model_name": "RealESRGAN_x4plus_anime_6B",
    "scale": 2,
    "model_cache_dir": str(MODELS_CACHE_DIR / "upscaler"),
    "enabled": True
}

LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s | %(levelname)s | %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_to_file": True,
    "log_file": str(BASE_DIR / "logs" / "falconai.log")
}

(BASE_DIR / "logs").mkdir(exist_ok=True)

PIPELINE_CONFIG = {
    "steps": [
        "face_processor",
        "story_generator",
        "frame_generator",
        "audio_generator",
        "video_assembler",
        "upscaler"
    ],

    "cleanup_temp": True,
    "save_locally": True
}

INPUT_VALIDATION = {
    "allowed_image_formats": [".jpg", ".jpeg", ".png", ".webp"],
    "max_image_size_mb": 10,
    "min_image_resolution": (100, 100),
    "max_name_length": 50,
    "min_age_years": 1,
    "max_age_years": 16
}