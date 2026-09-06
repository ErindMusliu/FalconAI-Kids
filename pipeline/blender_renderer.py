from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import psutil

from config.settings import BLENDER_CONFIG, OUTPUT_DIR
from utils.exceptions import PipelineError
from utils.logger import get_logger

logger = get_logger(__name__)


class BlenderRendererError(PipelineError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, step="blender_renderer")
        self.details = details or {}

class BlenderRenderer:
    FRAME_PATTERN = re.compile(r"Saved:\s*['\"]?(.*frame_(\d+)\.png)['\"]?")
    PROGRESS_PATTERN = re.compile(r"Rendered\s+(\d+)/(\d+)\s+Tiles|Sample\s+(\d+)/(\d+)")
    TIME_PATTERN = re.compile(r"Time:\s*(\d{2}:\d{2}\.\d{2})")

    def __init__(self, context: Optional[Dict[str, Any]] = None) -> None:
        self.context = context or {}
        self.config = BLENDER_CONFIG if isinstance(BLENDER_CONFIG, dict) else {}

        self.blender_executable = self._resolve_blender_executable()
        self.script_path = self._resolve_internal_script()
        self.output_dir = Path(
            self.context.get("output_dir", OUTPUT_DIR)
        ).resolve()
        self.frames_dir = self.output_dir / "rendered_frames"
        self.temp_dir = self.output_dir / "temp_payloads"
        
        self.max_retries = int(self.config.get("max_retries", 2))
        self.use_gpu = bool(self.config.get("use_gpu", True))

    def _resolve_blender_executable(self) -> str:
        configured_path = self.config.get("executable_path") or os.getenv("BLENDER_PATH")
        
        if configured_path:
            path = Path(configured_path).expanduser().resolve()
            if path.exists() and path.is_file():
                return str(path)

        for binary in ["blender", "blender.exe"]:
            for env_path in os.environ.get("PATH", "").split(os.pathsep):
                possible_path = Path(env_path) / binary
                if possible_path.exists() and possible_path.is_file():
                    return str(possible_path)

        raise BlenderRendererError(
            "Blender executable not found. Ensure Blender is installed and "
            "BLENDER_PATH is correctly configured in your settings environment."
        )

    def _resolve_internal_script(self) -> Path:
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "scripts" / "blender_script.py"

        if not script_path.exists():
            raise BlenderRendererError(
                f"Internal Blender script missing at expected path: '{script_path}'."
            )

        return script_path

    def _get_existing_frames(self) -> Dict[int, Path]:
        if not self.frames_dir.exists():
            return {}

        existing: Dict[int, Path] = {}
        for file in self.frames_dir.glob("frame_*.png"):
            if file.stat().st_size > 0:
                try:
                    frame_idx = int(file.stem.split("_")[1])
                    existing[frame_idx] = file
                except (IndexError, ValueError):
                    continue
        return existing

    def _prepare_render_payload(self, force_cpu: bool = False) -> Path:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        payload_file = self.temp_dir / f"blender_payload_{int(time.time())}.json"

        existing_frames = self._get_existing_frames()

        payload: Dict[str, Any] = {
            "character_name": self.context.get("name", "Hero"),
            "age": self.context.get("age", 5),
            "theme": self.context.get("preferences", {}).get("theme", "magical adventure"),
            "favorite_animal": self.context.get("preferences", {}).get("favorite_animal", "friendly creature"),
            "trait": self.context.get("preferences", {}).get("trait", "brave"),
            "face_texture_path": self.context.get("face_texture_path"),
            "story_scenes": self.context.get("story_scenes", []),
            "resume_skip_frames": list(existing_frames.keys()),
            "render_settings": {
                "resolution_width": self.config.get("resolution_width", 1920),
                "resolution_height": self.config.get("resolution_height", 1080),
                "fps": self.config.get("fps", 24),
                "engine": self.config.get("engine", "EEVEE"),
                "samples": self.config.get("samples", 64),
                "use_gpu": self.use_gpu and not force_cpu,
                "device_type": "CPU" if force_cpu else self.config.get("device_type", "CUDA"),
                "output_dir": str(self.frames_dir),
            },
        }

        try:
            with open(payload_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            raise BlenderRendererError(
                f"Failed to write Blender execution payload JSON: {exc}"
            ) from exc

        return payload_file

    def _monitor_system_resources(self) -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_gb": round((vm.total - vm.available) / (1024 ** 3), 2),
            "ram_free_gb": round(vm.available / (1024 ** 3), 2),
        }

    def render(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Path]:
        logger.info("Initializing Advanced Headless 3D Blender Engine...")
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        attempt = 0
        force_cpu = False
        success = False

        while attempt <= self.max_retries and not success:
            attempt += 1
            payload_file = self._prepare_render_payload(force_cpu=force_cpu)

            cmd = [
                self.blender_executable,
                "--background",
                "--python",
                str(self.script_path),
                "--",
                "--config",
                str(payload_file),
            ]

            logger.info(
                "Starting Blender 3D render worker (Attempt %d/%d) [Device: %s]...",
                attempt,
                self.max_retries + 1,
                "CPU" if force_cpu else "GPU/Default",
            )
            logger.debug("Executing Blender command: %s", " ".join(cmd))

            start_time = time.perf_counter()
            total_expected_frames = 0
            rendered_count = len(self._get_existing_frames())

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                if process.stdout:
                    for line in iter(process.stdout.readline, ""):
                        line_str = line.strip()
                        if not line_str:
                            continue

                        if "[TOTAL_FRAMES]" in line_str:
                            try:
                                total_expected_frames = int(line_str.split(":")[1])
                            except ValueError:
                                pass

                        frame_match = self.FRAME_PATTERN.search(line_str)
                        if frame_match or "[FRAME_RENDERED]" in line_str:
                            rendered_count += 1
                            stats = self._monitor_system_resources()
                            
                            label = (
                                f"Rendering 3D [Frame {rendered_count}/{total_expected_frames or '?'}] "
                                f"| RAM: {stats['ram_used_gb']}GB"
                            )

                            if progress_callback and total_expected_frames > 0:
                                progress_callback(
                                    rendered_count,
                                    total_expected_frames,
                                    label,
                                )

                        elif "Out of memory" in line_str or "CUDA error" in line_str or "optix" in line_str.lower():
                            logger.warning("Memory/GPU error detected in Blender stream: %s", line_str)
                            force_cpu = True

                        else:
                            logger.debug("[Blender Stream] %s", line_str)

                process.wait()

                if process.returncode == 0:
                    success = True
                    logger.success(
                        "Blender worker completed successfully in %.1f seconds.",
                        time.perf_counter() - start_time,
                    )
                else:
                    logger.error(
                        "Blender process crashed or exited with status code: %d",
                        process.returncode,
                    )
                    force_cpu = True

            except (OSError, subprocess.SubprocessError) as exc:
                logger.error("Subprocess execution exception: %s", exc)
                force_cpu = True

            finally:
                if payload_file.exists():
                    try:
                        payload_file.unlink()
                    except OSError:
                        pass

        rendered_files = sorted(list(self.frames_dir.glob("frame_*.png")))

        if not rendered_files:
            raise BlenderRendererError(
                "Headless 3D rendering failed completely. No frame files were generated."
            )

        logger.success(
            "Finalized 3D rendering pipeline: Total %d frame(s) available.",
            len(rendered_files),
        )

        return rendered_files
