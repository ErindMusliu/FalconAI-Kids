from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from config.settings import COMPOSITOR_CONFIG
from utils.logger import get_logger
from utils.exceptions import AnimationCompositingError

logger = get_logger(__name__)


class Compositor:
    """
    Merges a SadTalker talking-head clip (already extracted to PNG frames by
    TalkingHeadGenerator) onto the AnimateDiff/SD-generated background frames
    for a scene, for scenes where the child is speaking ("child" or "both").

    This is the piece that answers "where does the real photo-based head
    actually go inside the cartoon scene". Since Stable Diffusion doesn't
    guarantee where the main character lands in frame (there's no reliable
    detector for "the child" inside an illustrated cartoon scene, same
    limitation discussed for MouthAnimator), placement uses a fixed,
    configurable relative region (COMPOSITOR_CONFIG["head_region"]) rather
    than true scene understanding. This is a deliberate, documented
    approximation — not real subject detection.

    Background removal on the SadTalker head frames prefers `rembg` (a real
    segmentation model) when installed, and falls back to a naive
    corner-color-distance matte otherwise so the feature still works without
    that extra dependency, just with rougher edges.

    IMPORTANT frame-count design decision: this class outputs exactly as many
    frames as the *background* sequence has (frame_generator's raw AnimateDiff
    output, typically ANIMATOR_CONFIG["num_frames"] frames — NOT yet
    duration-stretched). The talking-head frames (which run at a different,
    audio-duration-driven frame count) are resampled to match. This keeps
    video_assembler.py's existing duration-based frame stretching
    (_normalize_frames) working unmodified downstream, since it only cares
    about per-scene frame counts, not where those frames came from.
    """

    def __init__(self):
        self.head_region = COMPOSITOR_CONFIG.get("head_region", (0.30, 0.06, 0.40, 0.48))
        self.feather_px = COMPOSITOR_CONFIG.get("feather_px", 6)
        self.bg_removal_tolerance = COMPOSITOR_CONFIG.get("bg_removal_tolerance", 30)

        self._rembg_session = self._try_load_rembg()

        logger.success(
            f"Compositor initialized | background removal: "
            f"{'rembg (model-based)' if self._rembg_session else 'naive corner-color fallback'}"
        )

    def _try_load_rembg(self):
        try:
            from rembg import new_session
            session = new_session("u2net")
            logger.success("rembg (u2net) background removal model loaded.")
            return session
        except ImportError:
            logger.warning(
                "rembg is not installed; falling back to a naive corner-color "
                "background matte for compositing the talking head (rougher edges). "
                "Install it with: pip install rembg"
            )
            return None
        except Exception as e:
            logger.warning(f"rembg failed to initialize ({e}); falling back to naive background removal.")
            return None

    def composite(
        self,
        background_frames_dir: Path,
        head_frames_dir: Path,
        output_dir: Path,
        scene_index: Optional[int] = None,
    ) -> Path:
        """
        Composites the talking-head frame sequence onto the background frame
        sequence for one scene, writing the result to output_dir using the
        background sequence's own filenames (so downstream code sees no
        difference in naming convention).
        """
        background_frames_dir = Path(background_frames_dir)
        head_frames_dir = Path(head_frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        bg_frames = self._collect_frames(background_frames_dir)
        head_frames = self._collect_frames(head_frames_dir)

        if not bg_frames:
            raise AnimationCompositingError(
                f"No background frames found in '{background_frames_dir}'.",
                scene_index=scene_index,
            )
        if not head_frames:
            raise AnimationCompositingError(
                f"No talking-head frames found in '{head_frames_dir}'.",
                scene_index=scene_index,
            )

        aligned_head_frames = self._align_frame_count(head_frames, target_count=len(bg_frames))

        try:
            for i, (bg_path, head_path) in enumerate(zip(bg_frames, aligned_head_frames), start=1):
                composited = self._composite_single_frame(bg_path, head_path)
                out_path = output_dir / bg_path.name
                composited.save(out_path, format="PNG", optimize=False)
        except Exception as e:
            raise AnimationCompositingError(
                f"Failed while compositing frame {i}/{len(bg_frames)}: {e}",
                scene_index=scene_index,
            )

        logger.success(f"[scene {scene_index}] Composited talking head onto {len(bg_frames)} background frame(s).")
        return output_dir

    def _collect_frames(self, frames_dir: Path) -> List[Path]:
        frames = sorted(frames_dir.glob("frame_*.png"))
        if not frames:
            frames = sorted(frames_dir.glob("*.png"))
        return frames

    def _align_frame_count(self, frames: List[Path], target_count: int) -> List[Path]:
        """Resamples `frames` to exactly `target_count` entries by nearest-index
        selection — mirrors the same np.linspace approach video_assembler.py
        already uses for its own duration-based frame stretching, so the
        temporal behavior stays consistent across the codebase."""
        current_count = len(frames)
        if current_count == target_count:
            return frames

        indices = np.linspace(0, current_count - 1, target_count).round().astype(int)
        indices = np.clip(indices, 0, current_count - 1)
        return [frames[i] for i in indices]

    def _composite_single_frame(self, bg_path: Path, head_path: Path) -> Image.Image:
        with Image.open(bg_path) as bg_img, Image.open(head_path) as head_img:
            bg_img = bg_img.convert("RGBA")
            head_img = head_img.convert("RGBA")

            head_rgba = self._extract_head_foreground(head_img)

            bg_w, bg_h = bg_img.size
            rel_x, rel_y, rel_w, rel_h = self.head_region
            target_w = max(1, int(rel_w * bg_w))
            target_h = max(1, int(rel_h * bg_h))
            target_x = int(rel_x * bg_w)
            target_y = int(rel_y * bg_h)

            head_resized = self._resize_preserving_aspect(head_rgba, target_w, target_h)

            paste_x = target_x + (target_w - head_resized.width) // 2
            paste_y = target_y + (target_h - head_resized.height) // 2

            result = bg_img.copy()
            result.alpha_composite(head_resized, dest=(paste_x, paste_y))
            return result.convert("RGB")

    def _resize_preserving_aspect(self, img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        src_w, src_h = img.size
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        return img.resize((new_w, new_h), Image.LANCZOS)

    def _extract_head_foreground(self, head_img: Image.Image) -> Image.Image:
        """Returns an RGBA image where the background of the SadTalker head
        crop has been made transparent, so only the head/shoulders composite
        onto the scene. Prefers rembg (real segmentation model); falls back
        to a naive corner-color matte with feathered edges otherwise."""
        if self._rembg_session is not None:
            try:
                from rembg import remove
                result = remove(head_img, session=self._rembg_session)
                return result.convert("RGBA")
            except Exception as e:
                logger.debug(f"rembg segmentation failed at runtime ({e}); using naive fallback for this frame.")

        return self._naive_background_removal(head_img)

    def _naive_background_removal(self, img: Image.Image) -> Image.Image:
        """Crude background matte for when rembg isn't available: samples the
        four corner pixels (assumed to be background, since SadTalker crops
        are tightly centered on the face), then makes any pixel close in
        color to that corner average transparent. Feathers the resulting
        alpha edge with a Gaussian blur so the cutout isn't razor-sharp.
        This will not handle busy/non-uniform backgrounds well — it's a
        best-effort fallback, not a real segmentation model."""
        rgba = img.convert("RGBA")
        arr = np.array(rgba).astype(np.float32)

        w, h = img.size
        corner_pixels = np.array([
            arr[0, 0, :3], arr[0, w - 1, :3],
            arr[h - 1, 0, :3], arr[h - 1, w - 1, :3],
        ])
        bg_color = corner_pixels.mean(axis=0)

        diff = np.sqrt(np.sum((arr[:, :, :3] - bg_color) ** 2, axis=-1))
        alpha = np.clip((diff - self.bg_removal_tolerance) * 4, 0, 255).astype(np.uint8)

        rgba_arr = arr.astype(np.uint8)
        rgba_arr[:, :, 3] = alpha

        result = Image.fromarray(rgba_arr, mode="RGBA")

        if self.feather_px > 0:
            alpha_channel = result.getchannel("A")
            alpha_channel = alpha_channel.filter(ImageFilter.GaussianBlur(self.feather_px))
            result.putalpha(alpha_channel)

        return result
