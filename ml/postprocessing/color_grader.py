import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

from utils.logger import get_logger

logger = get_logger(__name__)

MOOD_GRADES = {
    "happy"     : {"saturation": 1.3, "brightness": 1.1, "contrast": 1.05, "warmth": 10},
    "adventure" : {"saturation": 1.2, "brightness": 1.0, "contrast": 1.15, "warmth": 5},
    "magical"   : {"saturation": 1.4, "brightness": 1.15,"contrast": 1.0,  "warmth": 15},
    "mysterious": {"saturation": 0.9, "brightness": 0.9, "contrast": 1.2,  "warmth": -5},
    "heroic"    : {"saturation": 1.2, "brightness": 1.05,"contrast": 1.2,  "warmth": 8},
    "exciting"  : {"saturation": 1.35,"brightness": 1.0, "contrast": 1.25, "warmth": 3},
}


class ColorGrader:
    def grade_image(self, image: Image.Image, mood: str = "happy") -> Image.Image:
        params = MOOD_GRADES.get(mood, MOOD_GRADES["happy"])

        img = ImageEnhance.Color(image).enhance(params["saturation"])

        img = ImageEnhance.Brightness(img).enhance(params["brightness"])

        img = ImageEnhance.Contrast(img).enhance(params["contrast"])

        warmth = params["warmth"]
        if warmth != 0:
            img = self._apply_warmth(img, warmth)

        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=50, threshold=3))

        return img

    def grade_frame_dir(self, frames_dir: Path, mood: str = "happy") -> None:
        frames = sorted(frames_dir.glob("*.png"))
        for fp in frames:
            try:
                img    = Image.open(fp).convert("RGB")
                graded = self.grade_image(img, mood)
                graded.save(fp)
            except Exception as e:
                logger.debug(f"Color grade deshtoi për {fp.name}: {e}")

        logger.debug(f"Color grading aplikuar: {len(frames)} frames, mood={mood}")

    def _apply_warmth(self, image: Image.Image, warmth: int) -> Image.Image:
        arr = np.array(image, dtype=np.int16)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + warmth, 0, 255)      # R +
        arr[:, :, 2] = np.clip(arr[:, :, 2] - warmth // 2, 0, 255) # B -
        return Image.fromarray(arr.astype(np.uint8))