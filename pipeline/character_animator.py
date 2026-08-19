import shutil
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

from utils.logger import get_logger
from utils.exceptions import CharacterAnimationError

logger = get_logger(__name__)


class CharacterAnimator:
    """
    CPU-safe character animation orchestrator.

    Responsibilities:
        child:
            Try talking-head animation if available.
            Otherwise keep original frames.

        creature:
            Try procedural mouth animation if available.
            Otherwise keep original frames.

        both:
            Try child animation first, then creature animation.
            Any failed stage falls back gracefully.

        narrator_only:
            Copy frames unchanged.

    IMPORTANT:
        This module intentionally does not require GPU-only dependencies
        during import. Optional animation engines are loaded lazily.

    This allows FalconAI Kids to run in:
        - CPU-only environments
        - GPU environments
        - environments where SadTalker is unavailable
        - environments where creature animation is unavailable
    """

    def __init__(self):
        self.talking_head_generator = None
        self.mouth_animator = None
        self.compositor = None

        self._load_optional_components()

    # ------------------------------------------------------------------
    # OPTIONAL COMPONENT LOADING
    # ------------------------------------------------------------------

    def _load_optional_components(self) -> None:
        """
        Load optional animation components without allowing import failures
        to crash the entire pipeline.
        """

        # Talking Head
        try:
            from pipeline.talking_head_generator import TalkingHeadGenerator

            self.talking_head_generator = TalkingHeadGenerator()
            logger.success("Talking-head animation component initialized.")

        except Exception as e:
            self.talking_head_generator = None
            logger.warning(
                f"Talking-head component unavailable: {e}. "
                "Child character animation will use static fallback."
            )

        # Mouth Animator
        try:
            from pipeline.mouth_animator import MouthAnimator

            self.mouth_animator = MouthAnimator()
            logger.success("Mouth animation component initialized.")

        except Exception as e:
            self.mouth_animator = None
            logger.warning(
                f"Mouth animation component unavailable: {e}. "
                "Creature animation will use static fallback."
            )

        # Compositor
        try:
            from pipeline.compositor import Compositor

            self.compositor = Compositor()
            logger.success("Frame compositor initialized.")

        except Exception as e:
            self.compositor = None
            logger.warning(
                f"Compositor unavailable: {e}. "
                "Character compositing will use static fallback."
            )

    # ------------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------------

    def animate(
        self,
        scenes: List[Dict[str, Any]],
        frames_dir: Path,
        face_image_path: Optional[Path],
        output_dir: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Path:

        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_root = output_dir.parent / "character_animator_temp"

        try:
            temp_root.mkdir(
                parents=True,
                exist_ok=True,
            )
        except Exception as e:
            logger.warning(
                f"Could not create temporary animation directory: {e}"
            )

        total_scenes = len(scenes)

        if total_scenes == 0:
            logger.warning(
                "CharacterAnimator received an empty scene list."
            )
            return output_dir

        logger.info(
            f"Starting character animation stage for {total_scenes} scene(s)."
        )

        try:

            for idx, scene in enumerate(scenes):

                scene_num = self._get_scene_number(
                    scene,
                    idx,
                )

                scene_label = f"scene_{scene_num:02d}"

                input_scene_dir = frames_dir / scene_label
                output_scene_dir = output_dir / scene_label

                speaker = str(
                    scene.get(
                        "speaker",
                        "narrator_only",
                    )
                ).strip().lower()

                logger.step(
                    f"Character animation [{scene_label}] "
                    f"speaker='{speaker}'"
                )

                # ------------------------------------------------------
                # Missing scene directory
                # ------------------------------------------------------

                if not input_scene_dir.exists():

                    logger.warning(
                        f"[{scene_label}] Input frame directory does not "
                        f"exist: {input_scene_dir}"
                    )

                    if progress_callback:
                        progress_callback(
                            idx + 1,
                            total_scenes,
                            f"{scene_label}: skipped",
                        )

                    continue

                # ------------------------------------------------------
                # Process scene
                # ------------------------------------------------------

                try:

                    self._animate_scene(
                        scene=scene,
                        scene_label=scene_label,
                        input_scene_dir=input_scene_dir,
                        output_scene_dir=output_scene_dir,
                        face_image_path=face_image_path,
                        temp_root=temp_root,
                        scene_index=scene_num,
                    )

                except CharacterAnimationError as e:

                    logger.warning(
                        f"[{scene_label}] Character animation failed: {e}. "
                        "Using original frames."
                    )

                    self._copy_frames_unchanged(
                        input_scene_dir,
                        output_scene_dir,
                    )

                except Exception as e:

                    logger.warning(
                        f"[{scene_label}] Unexpected animation error: {e}. "
                        "Using original frames."
                    )

                    self._copy_frames_unchanged(
                        input_scene_dir,
                        output_scene_dir,
                    )

                # ------------------------------------------------------
                # Progress
                # ------------------------------------------------------

                if progress_callback:

                    progress_callback(
                        idx + 1,
                        total_scenes,
                        f"{scene_label}: character animation complete",
                    )

        finally:

            try:
                shutil.rmtree(
                    temp_root,
                    ignore_errors=True,
                )
            except Exception:
                pass

        logger.success(
            f"Character animation stage completed for "
            f"{total_scenes} scene(s) -> {output_dir}"
        )

        return output_dir

    # ------------------------------------------------------------------
    # SCENE ROUTER
    # ------------------------------------------------------------------

    def _animate_scene(
        self,
        scene: Dict[str, Any],
        scene_label: str,
        input_scene_dir: Path,
        output_scene_dir: Path,
        face_image_path: Optional[Path],
        temp_root: Path,
        scene_index: int,
    ) -> None:

        speaker = str(
            scene.get(
                "speaker",
                "narrator_only",
            )
        ).strip().lower()

        audio_path = scene.get(
            "narration_audio_path"
        )

        # --------------------------------------------------------------
        # Narrator only
        # --------------------------------------------------------------

        if speaker == "narrator_only":

            logger.debug(
                f"[{scene_label}] narrator_only -> "
                "frames copied unchanged."
            )

            self._copy_frames_unchanged(
                input_scene_dir,
                output_scene_dir,
            )

            return

        # --------------------------------------------------------------
        # No narration audio
        # --------------------------------------------------------------

        if not audio_path:

            logger.debug(
                f"[{scene_label}] No narration audio available. "
                "Character animation skipped."
            )

            self._copy_frames_unchanged(
                input_scene_dir,
                output_scene_dir,
            )

            return

        audio_path = Path(audio_path)

        if not audio_path.exists():

            logger.warning(
                f"[{scene_label}] Narration audio does not exist: "
                f"{audio_path}"
            )

            self._copy_frames_unchanged(
                input_scene_dir,
                output_scene_dir,
            )

            return

        # --------------------------------------------------------------
        # Child
        # --------------------------------------------------------------

        if speaker == "child":

            self._animate_child_only(
                input_scene_dir=input_scene_dir,
                output_scene_dir=output_scene_dir,
                face_image_path=face_image_path,
                audio_path=audio_path,
                temp_root=temp_root,
                scene_label=scene_label,
                scene_index=scene_index,
            )

            return

        # --------------------------------------------------------------
        # Creature
        # --------------------------------------------------------------

        if speaker == "creature":

            self._animate_creature_only(
                input_scene_dir=input_scene_dir,
                output_scene_dir=output_scene_dir,
                audio_path=audio_path,
                scene_index=scene_index,
            )

            return

        # --------------------------------------------------------------
        # Both
        # --------------------------------------------------------------

        if speaker == "both":

            self._animate_both(
                input_scene_dir=input_scene_dir,
                output_scene_dir=output_scene_dir,
                face_image_path=face_image_path,
                audio_path=audio_path,
                temp_root=temp_root,
                scene_label=scene_label,
                scene_index=scene_index,
            )

            return

        # --------------------------------------------------------------
        # Unknown speaker
        # --------------------------------------------------------------

        logger.warning(
            f"[{scene_label}] Unknown speaker '{speaker}'. "
            "Using narrator_only fallback."
        )

        self._copy_frames_unchanged(
            input_scene_dir,
            output_scene_dir,
        )

    # ------------------------------------------------------------------
    # CHILD ANIMATION
    # ------------------------------------------------------------------

    def _animate_child_only(
        self,
        input_scene_dir: Path,
        output_scene_dir: Path,
        face_image_path: Optional[Path],
        audio_path: Path,
        temp_root: Path,
        scene_label: str,
        scene_index: int,
    ) -> None:

        # No talking-head engine
        if self.talking_head_generator is None:

            logger.debug(
                f"[{scene_label}] Talking-head engine unavailable. "
                "Using original frames."
            )

            self._copy_frames_unchanged(
                input_scene_dir,
                output_scene_dir,
            )

            return

        # No face image
        if face_image_path is None:

            logger.debug(
                f"[{scene_label}] No face image supplied. "
                "Talking-head generation skipped."
            )

            self._copy_frames_unchanged(
                input_scene_dir,
                output_scene_dir,
            )

            return

        face_image_path = Path(face_image_path)

        if not face_image_path.exists():

            logger.debug(
                f"[{scene_label}] Face image not found: "
                f"{face_image_path}"
            )

            self._copy_frames_unchanged(
                input_scene_dir,
                output_scene_dir,
            )

            return

        head_temp_dir = (
            temp_root /
            f"{scene_label}_head"
        )

        try:

            head_frames_dir = (
                self.talking_head_generator.generate(
                    face_image_path=face_image_path,
                    audio_path=str(audio_path),
                    output_dir=head_temp_dir,
                    scene_index=scene_index,
                )
            )

        except Exception as e:

            logger.warning(
                f"[{scene_label}] Talking-head generation failed: {e}"
            )

            self._copy_frames_unchanged(
                input_scene_dir,
                output_scene_dir,
            )

            return

        if not head_frames_dir:

            logger.debug(
                f"[{scene_label}] Talking-head generator returned no "
                "frames. Using original background."
            )

            self._copy_frames_unchanged(
                input_scene_dir,
                output_scene_dir,
            )

            return

        # No compositor
        if self.compositor is None:

            logger.warning(
                f"[{scene_label}] Compositor unavailable. "
                "Cannot place talking head over background."
            )

            self._copy_frames_unchanged(
                input_scene_dir,
                output_scene_dir,
            )

            return

        try:

            self.compositor.composite(
                background_frames_dir=input_scene_dir,
                head_frames_dir=Path(head_frames_dir),
                output_dir=output_scene_dir,
                scene_index=scene_index,
            )

        except Exception as e:

            logger.warning(
                f"[{scene_label}] Character compositing failed: {e}"
            )

            self._copy_frames_unchanged(
                input_scene_dir,
                output_scene_dir,
            )

    # ------------------------------------------------------------------
    # CREATURE ANIMATION
    # ------------------------------------------------------------------

    def _animate_creature_only(
        self,
        input_scene_dir: Path,
        output_scene_dir: Path,
        audio_path: Path,
        scene_index: int,
    ) -> None:

        if self.mouth_animator is None:

            logger.debug(
                "Creature mouth animator unavailable. "
                "Using original frames."
            )

            self._copy_frames_unchanged(
                input_scene_dir,
                output_scene_dir,
            )

            return

        try:

            result_dir = self.mouth_animator.generate(
                frames_dir=input_scene_dir,
                audio_path=str(audio_path),
                output_dir=output_scene_dir,
                scene_index=scene_index,
            )

        except Exception as e:

            logger.warning(
                f"Creature mouth animation failed: {e}"
            )

            self._copy_frames_unchanged(
                input_scene_dir,
                output_scene_dir,
            )

            return

        if not result_dir:

            self._copy_frames_unchanged(
                input_scene_dir,
                output_scene_dir,
            )

    # ------------------------------------------------------------------
    # BOTH CHARACTERS
    # ------------------------------------------------------------------

    def _animate_both(
        self,
        input_scene_dir: Path,
        output_scene_dir: Path,
        face_image_path: Optional[Path],
        audio_path: Path,
        temp_root: Path,
        scene_label: str,
        scene_index: int,
    ) -> None:

        # --------------------------------------------------------------
        # Start with original background
        # --------------------------------------------------------------

        base_dir_for_mouth = input_scene_dir

        # --------------------------------------------------------------
        # Generate child talking head
        # --------------------------------------------------------------

        if (
            self.talking_head_generator is not None
            and self.compositor is not None
            and face_image_path is not None
            and Path(face_image_path).exists()
        ):

            head_temp_dir = (
                temp_root /
                f"{scene_label}_head"
            )

            composited_temp_dir = (
                temp_root /
                f"{scene_label}_composited"
            )

            try:

                head_frames_dir = (
                    self.talking_head_generator.generate(
                        face_image_path=Path(face_image_path),
                        audio_path=str(audio_path),
                        output_dir=head_temp_dir,
                        scene_index=scene_index,
                    )
                )

                if head_frames_dir:

                    self.compositor.composite(
                        background_frames_dir=input_scene_dir,
                        head_frames_dir=Path(head_frames_dir),
                        output_dir=composited_temp_dir,
                        scene_index=scene_index,
                    )

                    if self._has_frames(composited_temp_dir):

                        base_dir_for_mouth = composited_temp_dir

                        logger.debug(
                            f"[{scene_label}] Child talking-head "
                            "successfully composited."
                        )

            except Exception as e:

                logger.warning(
                    f"[{scene_label}] Child animation/compositing "
                    f"failed in 'both' mode: {e}. "
                    "Continuing with original background."
                )

        else:

            logger.debug(
                f"[{scene_label}] Child animation unavailable "
                "in 'both' mode."
            )

        # --------------------------------------------------------------
        # Apply creature mouth animation
        # --------------------------------------------------------------

        if self.mouth_animator is None:

            self._copy_frames_unchanged(
                base_dir_for_mouth,
                output_scene_dir,
            )

            return

        try:

            result_dir = self.mouth_animator.generate(
                frames_dir=base_dir_for_mouth,
                audio_path=str(audio_path),
                output_dir=output_scene_dir,
                scene_index=scene_index,
            )

            if not result_dir:

                self._copy_frames_unchanged(
                    base_dir_for_mouth,
                    output_scene_dir,
                )

        except Exception as e:

            logger.warning(
                f"[{scene_label}] Creature animation failed in "
                f"'both' mode: {e}. "
                "Using child-composited/original frames."
            )

            self._copy_frames_unchanged(
                base_dir_for_mouth,
                output_scene_dir,
            )

    # ------------------------------------------------------------------
    # FRAME UTILITIES
    # ------------------------------------------------------------------

    def _copy_frames_unchanged(
        self,
        src_dir: Path,
        dst_dir: Path,
    ) -> None:

        src_dir = Path(src_dir)
        dst_dir = Path(dst_dir)

        dst_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        frames = self._find_frames(src_dir)

        if not frames:

            logger.warning(
                f"No PNG frames found in: {src_dir}"
            )

            return

        for frame_path in frames:

            destination = (
                dst_dir /
                frame_path.name
            )

            try:

                shutil.copy2(
                    str(frame_path),
                    str(destination),
                )

            except Exception as e:

                logger.warning(
                    f"Failed to copy frame "
                    f"{frame_path.name}: {e}"
                )

    def _find_frames(
        self,
        directory: Path,
    ) -> List[Path]:

        if not directory.exists():
            return []

        frames = sorted(
            directory.glob("frame_*.png")
        )

        if not frames:

            frames = sorted(
                directory.glob("*.png")
            )

        return frames

    def _has_frames(
        self,
        directory: Path,
    ) -> bool:

        return len(
            self._find_frames(directory)
        ) > 0

    # ------------------------------------------------------------------
    # SCENE NUMBER
    # ------------------------------------------------------------------

    def _get_scene_number(
        self,
        scene: Dict[str, Any],
        fallback_index: int,
    ) -> int:

        possible_values = [
            scene.get("scene_number"),
            scene.get("index"),
            scene.get("scene_index"),
        ]

        for value in possible_values:

            try:

                if value is not None:

                    number = int(value)

                    if number > 0:
                        return number

            except (
                TypeError,
                ValueError,
            ):
                continue

        return fallback_index + 1
