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
    CPU-only compositor for FalconAI Kids.

    Combines talking-head frames with generated background frames.

    Design goals:
        - No GPU required.
        - No PyTorch.
        - No rembg/u2net dependency.
        - Low RAM usage.
        - Graceful fallback when a head frame cannot be processed.
        - Output frame count always matches the background frame count.

    The talking-head image is placed into a configurable relative region
    defined by:

        COMPOSITOR_CONFIG["head_region"]

    Example:

        (0.30, 0.06, 0.40, 0.48)

    means:

        x = 30% from left
        y = 6% from top
        width = 40% of background
        height = 48% of background

    This is intentionally a lightweight compositor rather than a
    computer-vision subject detector.
    """

    def __init__(self):
        self.head_region = self._get_head_region()

        self.feather_px = max(
            0,
            int(COMPOSITOR_CONFIG.get("feather_px", 4))
        )

        self.bg_removal_tolerance = max(
            0,
            float(COMPOSITOR_CONFIG.get("bg_removal_tolerance", 35))
        )

        self.enable_background_removal = bool(
            COMPOSITOR_CONFIG.get(
                "enable_background_removal",
                True
            )
        )

        self.head_opacity = float(
            COMPOSITOR_CONFIG.get(
                "head_opacity",
                1.0
            )
        )

        self.head_opacity = np.clip(
            self.head_opacity,
            0.0,
            1.0
        )

        logger.success(
            "CPU Compositor initialized | "
            f"head_region={self.head_region} | "
            f"background_removal={self.enable_background_removal}"
        )

    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------

    def _get_head_region(self) -> Tuple[float, float, float, float]:
        """
        Safely reads the configured head region.

        Returns:
            (x, y, width, height)
        """

        default_region = (
            0.30,
            0.06,
            0.40,
            0.48,
        )

        try:
            region = COMPOSITOR_CONFIG.get(
                "head_region",
                default_region
            )

            if len(region) != 4:
                return default_region

            x, y, w, h = map(float, region)

            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            w = np.clip(w, 0.01, 1.0)
            h = np.clip(h, 0.01, 1.0)

            return (
                float(x),
                float(y),
                float(w),
                float(h),
            )

        except Exception:
            logger.warning(
                "Invalid COMPOSITOR_CONFIG['head_region']; "
                "using default region."
            )

            return default_region

    # ------------------------------------------------------------------
    # MAIN API
    # ------------------------------------------------------------------

    def composite(
        self,
        background_frames_dir: Path,
        head_frames_dir: Path,
        output_dir: Path,
        scene_index: Optional[int] = None,
    ) -> Path:
        """
        Composite talking-head frames onto background frames.

        The output always contains exactly the same number of frames as
        the background sequence.
        """

        background_frames_dir = Path(background_frames_dir)
        head_frames_dir = Path(head_frames_dir)
        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        bg_frames = self._collect_frames(
            background_frames_dir
        )

        head_frames = self._collect_frames(
            head_frames_dir
        )

        if not bg_frames:
            raise AnimationCompositingError(
                f"No background frames found in "
                f"'{background_frames_dir}'.",
                scene_index=scene_index,
            )

        if not head_frames:
            raise AnimationCompositingError(
                f"No talking-head frames found in "
                f"'{head_frames_dir}'.",
                scene_index=scene_index,
            )

        aligned_head_frames = self._align_frame_count(
            head_frames,
            len(bg_frames)
        )

        logger.debug(
            f"[scene {scene_index}] "
            f"Background frames: {len(bg_frames)} | "
            f"Head frames: {len(head_frames)} | "
            f"Aligned: {len(aligned_head_frames)}"
        )

        processed = 0

        for i, (bg_path, head_path) in enumerate(
            zip(bg_frames, aligned_head_frames),
            start=1,
        ):
            try:
                composited = self._composite_single_frame(
                    bg_path,
                    head_path,
                )

                output_path = output_dir / bg_path.name

                composited.save(
                    output_path,
                    format="PNG",
                    optimize=False,
                )

                processed += 1

            except Exception as e:
                logger.warning(
                    f"[scene {scene_index}] "
                    f"Frame {i}/{len(bg_frames)} compositing failed: {e}. "
                    f"Using background frame unchanged."
                )

                self._copy_background_frame(
                    bg_path,
                    output_dir / bg_path.name
                )

        if processed == 0:
            raise AnimationCompositingError(
                f"No frames could be composited for scene "
                f"{scene_index}.",
                scene_index=scene_index,
            )

        logger.success(
            f"[scene {scene_index}] "
            f"CPU compositor processed {len(bg_frames)} frame(s)."
        )

        return output_dir

    # ------------------------------------------------------------------
    # FRAME DISCOVERY
    # ------------------------------------------------------------------

    def _collect_frames(
        self,
        frames_dir: Path,
    ) -> List[Path]:
        """
        Collect PNG frames in deterministic order.
        """

        if not frames_dir.exists():
            return []

        frames = sorted(
            frames_dir.glob("frame_*.png")
        )

        if frames:
            return frames

        frames = sorted(
            frames_dir.glob("*.png")
        )

        return frames

    def _align_frame_count(
        self,
        frames: List[Path],
        target_count: int,
    ) -> List[Path]:
        """
        Resample frame sequence to target_count.

        Uses nearest-index selection instead of creating additional
        image data, keeping memory usage very low.
        """

        if not frames:
            return []

        if target_count <= 0:
            return []

        current_count = len(frames)

        if current_count == target_count:
            return frames

        if current_count == 1:
            return [frames[0]] * target_count

        indices = np.linspace(
            0,
            current_count - 1,
            target_count,
        )

        indices = np.round(indices).astype(np.int32)

        indices = np.clip(
            indices,
            0,
            current_count - 1,
        )

        return [
            frames[int(index)]
            for index in indices
        ]

    # ------------------------------------------------------------------
    # SINGLE FRAME
    # ------------------------------------------------------------------

    def _composite_single_frame(
        self,
        bg_path: Path,
        head_path: Path,
    ) -> Image.Image:
        """
        Composite one head frame onto one background frame.
        """

        with Image.open(bg_path) as bg_source:
            background = bg_source.convert("RGBA")

        with Image.open(head_path) as head_source:
            head = head_source.convert("RGBA")

        # Generate transparency mask.
        if self.enable_background_removal:
            head = self._extract_head_foreground(head)
        else:
            head = self._apply_global_opacity(head)

        bg_w, bg_h = background.size

        rel_x, rel_y, rel_w, rel_h = self.head_region

        target_w = max(
            1,
            int(rel_w * bg_w)
        )

        target_h = max(
            1,
            int(rel_h * bg_h)
        )

        target_x = int(
            rel_x * bg_w
        )

        target_y = int(
            rel_y * bg_h
        )

        head_resized = self._resize_preserving_aspect(
            head,
            target_w,
            target_h,
        )

        paste_x = (
            target_x
            + (target_w - head_resized.width) // 2
        )

        paste_y = (
            target_y
            + (target_h - head_resized.height) // 2
        )

        # Keep image inside the canvas.
        paste_x = max(
            0,
            min(
                paste_x,
                bg_w - head_resized.width,
            ),
        )

        paste_y = max(
            0,
            min(
                paste_y,
                bg_h - head_resized.height,
            ),
        )

        result = background.copy()

        result.alpha_composite(
            head_resized,
            dest=(
                paste_x,
                paste_y,
            ),
        )

        return result.convert("RGB")

    # ------------------------------------------------------------------
    # RESIZE
    # ------------------------------------------------------------------

    def _resize_preserving_aspect(
        self,
        image: Image.Image,
        target_w: int,
        target_h: int,
    ) -> Image.Image:
        """
        Resize image while preserving its aspect ratio.
        """

        src_w, src_h = image.size

        if src_w <= 0 or src_h <= 0:
            raise ValueError(
                "Invalid source image dimensions."
            )

        scale = min(
            target_w / src_w,
            target_h / src_h,
        )

        new_w = max(
            1,
            int(src_w * scale),
        )

        new_h = max(
            1,
            int(src_h * scale),
        )

        return image.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS,
        )

    # ------------------------------------------------------------------
    # BACKGROUND REMOVAL
    # ------------------------------------------------------------------

    def _extract_head_foreground(
        self,
        head_img: Image.Image,
    ) -> Image.Image:
        """
        Lightweight CPU-only background removal.

        The four corners are assumed to represent the background.

        Pixels close to the estimated background color become transparent.

        This is intentionally simple because the goal is to keep the
        pipeline GPU/model-free.
        """

        return self._naive_background_removal(
            head_img
        )

    def _naive_background_removal(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        Creates a lightweight alpha matte using corner color distance.

        This is not semantic segmentation. It is a CPU-friendly
        approximation intended for simple talking-head images.
        """

        rgba = image.convert("RGBA")

        # Convert to NumPy only for the current frame.
        arr = np.asarray(
            rgba,
            dtype=np.float32,
        ).copy()

        height, width = arr.shape[:2]

        if width == 0 or height == 0:
            return rgba

        # --------------------------------------------------------------
        # Sample corners
        # --------------------------------------------------------------

        corner_pixels = np.array(
            [
                arr[0, 0, :3],
                arr[0, width - 1, :3],
                arr[height - 1, 0, :3],
                arr[height - 1, width - 1, :3],
            ],
            dtype=np.float32,
        )

        background_color = np.median(
            corner_pixels,
            axis=0,
        )

        # --------------------------------------------------------------
        # Calculate color distance
        # --------------------------------------------------------------

        rgb = arr[:, :, :3]

        difference = rgb - background_color

        distance = np.sqrt(
            np.sum(
                difference * difference,
                axis=2,
            )
        )

        tolerance = self.bg_removal_tolerance

        # --------------------------------------------------------------
        # Build alpha mask
        # --------------------------------------------------------------

        # Pixels clearly matching background -> transparent.
        # Pixels clearly different -> opaque.
        alpha = np.clip(
            (distance - tolerance) * 5.0,
            0.0,
            255.0,
        )

        alpha = alpha.astype(
            np.uint8
        )

        # --------------------------------------------------------------
        # Protect center area
        # --------------------------------------------------------------
        #
        # A pure corner-color algorithm can accidentally remove parts
        # of a face if the face contains similar colors. We therefore
        # avoid aggressive transparency in the central region.

        center_x1 = int(width * 0.20)
        center_x2 = int(width * 0.80)

        center_y1 = int(height * 0.12)
        center_y2 = int(height * 0.90)

        center_alpha = alpha[
            center_y1:center_y2,
            center_x1:center_x2,
        ]

        center_alpha = np.maximum(
            center_alpha,
            80,
        )

        alpha[
            center_y1:center_y2,
            center_x1:center_x2,
        ] = center_alpha

        # --------------------------------------------------------------
        # Apply original alpha
        # --------------------------------------------------------------

        original_alpha = arr[:, :, 3]

        alpha = np.minimum(
            alpha.astype(np.float32),
            original_alpha,
        )

        alpha = alpha.astype(
            np.uint8
        )

        # --------------------------------------------------------------
        # Feather edges
        # --------------------------------------------------------------

        result_array = arr.astype(
            np.uint8
        )

        result_array[:, :, 3] = alpha

        result = Image.fromarray(
            result_array,
            mode="RGBA",
        )

        if self.feather_px > 0:
            alpha_channel = result.getchannel("A")

            alpha_channel = alpha_channel.filter(
                ImageFilter.GaussianBlur(
                    self.feather_px
                )
            )

            result.putalpha(
                alpha_channel
            )

        return self._apply_global_opacity(
            result
        )

    # ------------------------------------------------------------------
    # OPACITY
    # ------------------------------------------------------------------

    def _apply_global_opacity(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        Applies configured global opacity without modifying RGB values.
        """

        if self.head_opacity >= 0.999:
            return image

        rgba = image.convert("RGBA")

        alpha = rgba.getchannel("A")

        alpha = alpha.point(
            lambda value: int(
                value * self.head_opacity
            )
        )

        rgba.putalpha(alpha)

        return rgba

    # ------------------------------------------------------------------
    # FALLBACK
    # ------------------------------------------------------------------

    def _copy_background_frame(
        self,
        source: Path,
        destination: Path,
    ) -> None:
        """
        Copies the original background frame when compositing fails.
        """

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with Image.open(source) as image:
            image.convert("RGB").save(
                destination,
                format="PNG",
                optimize=False,
            )
