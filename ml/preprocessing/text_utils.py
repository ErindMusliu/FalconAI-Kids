import re
import unicodedata
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = _remove_emoji(text)

    text = unicodedata.normalize("NFC", text)

    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)].rstrip() + suffix

def split_into_sentences(text: str) -> list[str]:
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]

def build_sd_prompt(
    scene_description: str,
    style_prefix: str,
    character_desc: str,
    mood: str,
    negative_prompt: Optional[str] = None,
) -> tuple[str, str]:
    scene = clean_text(scene_description)[:200]
    char  = clean_text(character_desc)[:100]

    positive = (
        f"{style_prefix}, "
        f"{scene}, "
        f"{char}, "
        f"mood: {mood}, "
        f"masterpiece, best quality, highly detailed"
    )

    positive = re.sub(r',\s*,', ',', positive)
    positive = re.sub(r'\s+', ' ', positive).strip(', ')

    neg = negative_prompt or (
        "ugly, blurry, deformed, scary, violent, adult content, "
        "watermark, text, logo, bad anatomy, low quality"
    )

    return positive, neg

def extract_json_from_text(text: str) -> Optional[str]:
    import json

    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = re.sub(r'```\s*$', '', cleaned).strip()
    try:
        json.loads(cleaned)
        return cleaned
    except (json.JSONDecodeError, ValueError):
        pass

    return None

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def sanitize_filename(name: str) -> str:
    name = name.lower().strip()
    name = unicodedata.normalize("NFD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[\s-]+', '_', name)
    return name[:50]

def _remove_emoji(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)