import shutil
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

from utils.logger import get_logger
from utils.exceptions import CharacterAnimationError

from pipeline.talking_head_generator import TalkingHeadGenerator
from pipeline.mouth_animator import MouthAnimator
from pipeline.compositor import Compositor

logger = get_logger(__name__)


class CharacterAnimator:
    """
    Orchestrates the three animation components (SadTalker talking-head
    generation, procedural creature mouth-flap, and background compositing)
    according to each scene's `speaker` field, produced by story_generator.py:

        "child"          -> SadTalker head, composited onto the background
        "creature"       -> procedural mouth-flap applied directly to the
                             background frames
        "both"           -> SadTalker head composited in first, THEN
                             mouth-flap applied on top of the result. Both are
                             driven by the SAME single narration audio track —
                             this pipeline generates one narration voice per
                             scene, not separate isolated tracks per
                             character, so "both talking" means both mouths
                             move to that one track rather than each getting
                             their own independent line.
        "narrator_only"  -> frames pass through completely unchanged

    Failures degrade gracefully per scene: if lip-sync/motion generation
    fails or isn't available for a given scene, that scene's original
    (pre-animation) frames are carried through unchanged into the output
    directory rather than aborting the whole video — the same philosophy
    frame_generator.py already applies when AnimateDiff itself fails and it
    falls back to static frames.

    Output directory structure mirrors the input exactly
    (scene_XX/frame_NNNN.png), so video_assembler.py requires no changes to
    consume it.
    """

    def __init__(self):
        self.talking_head_generator = TalkingHeadGenerator()
        self.mouth_animator = MouthAnimator()
        self.compositor = Compositor()

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
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_root = output_dir.parent / "character_animator_temp"
        temp_root.mkdir(parents=True, exist_ok=True)

        total_scenes = len(scenes)

        try:
            for idx, scene in enumerate(scenes):
                scene_num = scene.get("scene_number", idx + 1)
                scene_label = f"scene_{scene_num:02d}"
                input_scene_dir = frames_dir / scene_label
                output_scene_dir = output_dir / scene_label

                logger.step(f"Character animation [{scene_label}] speaker='{scene.get('speaker', 'narrator_only')}'")

                if not input_scene_dir.exists():
                    logger.warning(f"[{scene_label}] Input frame directory missing ('{input_scene_dir}'); skipping scene.")
                    if progress_callback:
                        progress_callback(idx + 1, total_scenes, f"{scene_label}: skipped (no input frames)")
                    continue

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
                        f"[{scene_label}] Character animation failed ({e}); "
                        f"falling back to the original animated-but-silent frames for this scene."
                    )
                    self._copy_frames_unchanged(input_scene_dir, output_scene_dir)
                except Exception as e:
                    logger.warning(
                        f"[{scene_label}] Unexpected error during character animation ({e}); "
                        f"falling back to the original animated-but-silent frames for this scene."
                    )
                    self._copy_frames_unchanged(input_scene_dir, output_scene_dir)

                if progress_callback:
                    progress_callback(idx + 1, total_scenes, f"{scene_label}: character animation complete")

        finally:
            try:
                shutil.rmtree(temp_root, ignore_errors=True)
            except Exception:
                pass

        logger.success(f"Character animation stage completed for {total_scenes} scene(s) -> {output_dir}")
        return output_dir

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
        speaker = str(scene.get("speaker", "narrator_only")).strip().lower()
        audio_path = scene.get("narration_audio_path")

        if speaker == "narrator_only" or not audio_path:
            self._copy_frames_unchanged(input_scene_dir, output_scene_dir)
            return

        if speaker == "child":
            self._animate_child_only(
                input_scene_dir, output_scene_dir, face_image_path, audio_path,
                temp_root, scene_label, scene_index,
            )

        elif speaker == "creature":
            self._animate_creature_only(
                input_scene_dir, output_scene_dir, audio_path, scene_index,
            )

        elif speaker == "both":
            self._animate_both(
                input_scene_dir, output_scene_dir, face_image_path, audio_path,
                temp_root, scene_label, scene_index,
            )

        else:
            logger.warning(f"[{scene_label}] Unrecognized speaker value '{speaker}'; treating scene as narrator_only.")
            self._copy_frames_unchanged(input_scene_dir, output_scene_dir)

    def _animate_child_only(
        self,
        input_scene_dir: Path,
        output_scene_dir: Path,
        face_image_path: Optional[Path],
        audio_path: str,
        temp_root: Path,
        scene_label: str,
        scene_index: int,
    ) -> None:
        head_temp_dir = temp_root / f"{scene_label}_head"

        head_frames_dir = self.talking_head_generator.generate(
            face_image_path=face_image_path,
            audio_path=audio_path,
            output_dir=head_temp_dir,
            scene_index=scene_index,
        )

        if head_frames_dir is None:
            logger.debug(f"[{scene_label}] Talking-head generation unavailable/skipped; keeping original background frames.")
            self._copy_frames_unchanged(input_scene_dir, output_scene_dir)
            return

        self.compositor.composite(
            background_frames_dir=input_scene_dir,
            head_frames_dir=head_frames_dir,
            output_dir=output_scene_dir,
            scene_index=scene_index,
        )

    def _animate_creature_only(
        self,
        input_scene_dir: Path,
        output_scene_dir: Path,
        audio_path: str,
        scene_index: int,
    ) -> None:
        result_dir = self.mouth_animator.generate(
            frames_dir=input_scene_dir,
            audio_path=audio_path,
            output_dir=output_scene_dir,
            scene_index=scene_index,
        )

        if result_dir is None:
            self._copy_frames_unchanged(input_scene_dir, output_scene_dir)

    def _animate_both(
        self,
        input_scene_dir: Path,
        output_scene_dir: Path,
        face_image_path: Optional[Path],
        audio_path: str,
        temp_root: Path,
        scene_label: str,
        scene_index: int,
    ) -> None:
        head_temp_dir = temp_root / f"{scene_label}_head"
        composited_temp_dir = temp_root / f"{scene_label}_composited"

        head_frames_dir = self.talking_head_generator.generate(
            face_image_path=face_image_path,
            audio_path=audio_path,
            output_dir=head_temp_dir,
            scene_index=scene_index,
        )

        if head_frames_dir is not None:
            self.compositor.composite(
                background_frames_dir=input_scene_dir,
                head_frames_dir=head_frames_dir,
                output_dir=composited_temp_dir,
                scene_index=scene_index,
            )
            base_dir_for_mouth = composited_temp_dir
        else:
            logger.debug(
                f"[{scene_label}] Talking-head generation unavailable/skipped for a 'both' scene; "
                f"using the original background as the base instead."
            )
            base_dir_for_mouth = input_scene_dir

        result_dir = self.mouth_animator.generate(
            frames_dir=base_dir_for_mouth,
            audio_path=audio_path,
            output_dir=output_scene_dir,
            scene_index=scene_index,
        )

        if result_dir is None:
            self._copy_frames_unchanged(base_dir_for_mouth, output_scene_dir)

    def _copy_frames_unchanged(self, src_dir: Path, dst_dir: Path) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        frames = sorted(src_dir.glob("frame_*.png"))
        if not frames:
            frames = sorted(src_dir.glob("*.png"))
        for frame_path in frames:
            shutil.copy2(str(frame_path), str(dst_dir / frame_path.name))
