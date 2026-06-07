from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)


def load_image_cv2(path: Union[str, Path]) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        img = cv2.imdecode(
            np.fromfile(str(path), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
    if img is None:
        raise ValueError(f"Imazhi nuk u ngarkua: {path}")
    return img


def load_image_pil(path: Union[str, Path]) -> Image.Image:
    return Image.open(path).convert("RGB")

def resize_image(
    image: Union[np.ndarray, Image.Image],
    width: int,
    height: int,
    keep_aspect: bool = False,
) -> Union[np.ndarray, Image.Image]:
    if isinstance(image, np.ndarray):
        if keep_aspect:
            image = _resize_with_padding_cv2(image, width, height)
        else:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    else:
        if keep_aspect:
            image = _resize_with_padding_pil(image, width, height)
        else:
            image = image.resize((width, height), Image.LANCZOS)
    return image

def normalize_image(image: np.ndarray, mean=(0.5,), std=(0.5,)) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    return img

def denormalize_image(image: np.ndarray) -> np.ndarray:
    img = (image * 0.5 + 0.5) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))

def numpy_to_pil(image: np.ndarray) -> Image.Image:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)

def save_image(image: Union[np.ndarray, Image.Image], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        cv2.imwrite(str(path), image)
    else:
        image.save(str(path))


def get_image_size(path: Union[str, Path]) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def enhance_sharpness(image: Image.Image, factor: float = 1.5) -> Image.Image:
    from PIL import ImageEnhance
    return ImageEnhance.Sharpness(image).enhance(factor)

def create_blank_image(
    width: int, height: int,
    color: tuple = (0, 0, 0)
) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color
    return img

def blend_images(
    img_a: np.ndarray,
    img_b: np.ndarray,
    alpha: float,
) -> np.ndarray:
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)
    blended = a * (1 - alpha) + b * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)

def _resize_with_padding_cv2(
    image: np.ndarray, width: int, height: int
) -> np.ndarray:
    h, w = image.shape[:2]
    scale   = min(width / w, height / h)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    pad_top  = (height - new_h)
    pad_left = (width  - new_w)

    result = np.zeros((height, width, 3), dtype=np.uint8)
    result[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    return result

def _resize_with_padding_pil(
    image: Image.Image, width: int, height: int
) -> Image.Image:
    image.thumbnail((width, height), Image.LANCZOS)
    result = Image.new("RGB", (width, height), (0, 0, 0))
    offset = ((width - image.width) // 2, (height - image.height) // 2)
    result.paste(image, offset)
    return result