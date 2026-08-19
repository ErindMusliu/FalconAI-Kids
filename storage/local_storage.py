import json
import re
import shutil
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from config.settings import OUTPUT_DIR, TEMP_DIR
from utils.logger import get_logger
from utils.exceptions import StorageError, DiskSpaceError

logger = get_logger(__name__)


class LocalStorage:
    """
    Reliable local filesystem storage layer for FalconAI Kids.

    Responsibilities:
    - Persist generated videos safely.
    - Maintain optional JSON sidecar metadata.
    - Enumerate stored videos.
    - Delete videos and their metadata.
    - Clean stale temporary artifacts.
    - Report storage statistics.
    """

    VIDEO_EXTENSIONS = frozenset({
        ".mp4",
        ".mkv",
        ".mov",
        ".avi",
        ".webm",
    })

    DEFAULT_REQUIRED_FREE_GB = 1.0
    MIN_FREE_BYTES_AFTER_WRITE = 100 * 1024 * 1024  # 100 MB
    MAX_FILENAME_LENGTH = 80

    def __init__(self, base_dir: Union[str, Path] = OUTPUT_DIR) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()

        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)

            if not self.base_dir.is_dir():
                raise StorageError(
                    f"Storage target is not a directory: {self.base_dir}"
                )

        except StorageError:
            raise
        except Exception as init_err:
            raise StorageError(
                f"Failed to initialize local storage directory: {self.base_dir}"
            ) from init_err

        logger.debug(
            f"Local storage initialized successfully: {self.base_dir}"
        )

    # ------------------------------------------------------------------
    # VIDEO PERSISTENCE
    # ------------------------------------------------------------------

    def save_video(
        self,
        video_path: Union[str, Path],
        name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """
        Safely copies a generated video into the persistent output directory.

        The file is first copied into a temporary file inside the same
        filesystem and then atomically renamed into its final destination.
        This prevents partially-written output files from appearing as valid
        videos if the process is interrupted.
        """

        source = Path(video_path).expanduser()

        if not source.exists():
            raise FileNotFoundError(
                f"Source video file not found: {source}"
            )

        if not source.is_file():
            raise StorageError(
                f"Source video path is not a regular file: {source}"
            )

        try:
            source_size = source.stat().st_size
        except OSError as stat_err:
            raise StorageError(
                f"Unable to inspect source video: {source}"
            ) from stat_err

        if source_size <= 0:
            raise StorageError(
                f"Source video is empty: {source}"
            )

        suffix = source.suffix.lower()

        if suffix not in self.VIDEO_EXTENSIONS:
            logger.warning(
                f"Saving video with non-standard extension: {suffix or '<none>'}"
            )

        self._check_disk_space_for_bytes(source_size)

        safe_name = _sanitize(name)

        if not safe_name:
            safe_name = "falconai_kids_video"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        destination_name = (
            f"{safe_name}_{timestamp}{suffix}"
        )

        destination = self._safe_output_path(destination_name)

        temp_path: Optional[Path] = None

        try:
            temp_path = self._create_temp_path(
                suffix=suffix or ".tmp"
            )

            logger.debug(
                f"Persisting video atomically: "
                f"{source.name} -> {destination.name}"
            )

            shutil.copy2(source, temp_path)

            copied_size = temp_path.stat().st_size

            if copied_size != source_size:
                raise StorageError(
                    "Video transfer verification failed: "
                    f"source={source_size} bytes, "
                    f"copied={copied_size} bytes"
                )

            if copied_size <= 0:
                raise StorageError(
                    "Video transfer produced an empty destination file."
                )

            temp_path.replace(destination)
            temp_path = None

            logger.debug(
                f"Video persisted successfully: {destination}"
            )

        except StorageError:
            raise

        except Exception as write_err:
            raise StorageError(
                f"Failed to persist video to local storage: "
                f"{write_err}"
            ) from write_err

        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        if metadata is not None:
            self._write_metadata(
                video_path=destination,
                metadata=metadata,
                timestamp=timestamp,
            )

        return destination

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------

    def _write_metadata(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        timestamp: str,
    ) -> Optional[Path]:
        meta_path = video_path.with_suffix(".json")

        payload = dict(metadata)
        payload.setdefault("saved_at", timestamp)
        payload.setdefault("file", video_path.name)

        payload.setdefault(
            "file_size_bytes",
            video_path.stat().st_size,
        )

        payload.setdefault(
            "storage_path",
            str(video_path),
        )

        temp_path: Optional[Path] = None

        try:
            temp_path = self._create_temp_path(
                suffix=".json"
            )

            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

            temp_path.write_text(
                serialized,
                encoding="utf-8",
            )

            temp_path.replace(meta_path)
            temp_path = None

            logger.debug(
                f"Video metadata persisted successfully: {meta_path.name}"
            )

            return meta_path

        except Exception as meta_err:
            logger.warning(
                f"Video was saved, but metadata could not be written "
                f"for {video_path.name}: {meta_err}"
            )
            return None

        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # VIDEO INDEX
    # ------------------------------------------------------------------

    def list_videos(self) -> list[dict[str, Any]]:
        """
        Returns stored videos ordered from newest to oldest.
        """

        if not self.base_dir.exists():
            return []

        videos: list[dict[str, Any]] = []

        try:
            entries = list(self.base_dir.iterdir())
        except Exception as read_err:
            logger.error(
                f"Unable to enumerate local storage directory: {read_err}"
            )
            return videos

        for file_item in entries:
            try:
                if not file_item.is_file():
                    continue

                if file_item.suffix.lower() not in self.VIDEO_EXTENSIONS:
                    continue

                stats = file_item.stat()

                metadata = self._read_metadata(file_item)

                videos.append({
                    "path": file_item,
                    "name": file_item.name,
                    "size_mb": stats.st_size / (1024 ** 2),
                    "size_bytes": stats.st_size,
                    "created": datetime.fromtimestamp(
                        stats.st_mtime
                    ),
                    "modified": datetime.fromtimestamp(
                        stats.st_mtime
                    ),
                    "metadata": metadata,
                })

            except FileNotFoundError:
                # File may have been removed while indexing.
                continue

            except Exception as stat_err:
                logger.warning(
                    f"Unable to index stored video "
                    f"{file_item.name}: {stat_err}"
                )

        videos.sort(
            key=lambda item: item["created"],
            reverse=True,
        )

        return videos

    def _read_metadata(
        self,
        video_path: Path,
    ) -> dict[str, Any]:
        meta_path = video_path.with_suffix(".json")

        if not meta_path.exists() or not meta_path.is_file():
            return {}

        try:
            content = meta_path.read_text(
                encoding="utf-8"
            ).strip()

            if not content:
                return {}

            payload = json.loads(content)

            if not isinstance(payload, dict):
                logger.warning(
                    f"Metadata file does not contain a JSON object: "
                    f"{meta_path.name}"
                )
                return {}

            return payload

        except json.JSONDecodeError as parse_err:
            logger.warning(
                f"Corrupted metadata ignored for "
                f"{video_path.name}: {parse_err}"
            )
            return {}

        except Exception as read_err:
            logger.warning(
                f"Unable to read metadata for "
                f"{video_path.name}: {read_err}"
            )
            return {}

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def delete_video(
        self,
        video_path: Union[str, Path],
    ) -> None:
        """
        Deletes a video and its matching sidecar metadata.

        The target must resolve inside the configured storage directory.
        This prevents accidental deletion of unrelated filesystem files.
        """

        target = Path(video_path).expanduser()

        if not target.is_absolute():
            target = self.base_dir / target

        try:
            target = target.resolve()
        except Exception as resolve_err:
            raise StorageError(
                f"Unable to resolve deletion target: {target}"
            ) from resolve_err

        if not self._is_inside_storage(target):
            raise StorageError(
                f"Refusing to delete file outside storage directory: "
                f"{target}"
            )

        try:
            if target.exists():
                if not target.is_file():
                    raise StorageError(
                        f"Deletion target is not a regular file: {target}"
                    )

                target.unlink()

                logger.debug(
                    f"Video deleted successfully: {target.name}"
                )

            metadata_path = target.with_suffix(".json")

            if metadata_path.exists():
                if metadata_path.is_file():
                    metadata_path.unlink()

                    logger.debug(
                        f"Associated metadata deleted: "
                        f"{metadata_path.name}"
                    )

        except StorageError:
            raise

        except Exception as delete_err:
            raise StorageError(
                f"Failed to delete stored video safely: "
                f"{delete_err}"
            ) from delete_err

    # ------------------------------------------------------------------
    # TEMPORARY STORAGE CLEANUP
    # ------------------------------------------------------------------

    def cleanup_temp(
        self,
        older_than_hours: int = 24,
    ) -> int:
        """
        Removes temporary files/directories older than the specified age.

        Only direct children of TEMP_DIR are removed. The method never
        recursively targets the entire configured project tree.
        """

        if older_than_hours < 0:
            raise ValueError(
                "older_than_hours must be >= 0"
            )

        target_temp_dir = Path(TEMP_DIR).expanduser()

        if not target_temp_dir.exists():
            logger.debug(
                f"Temporary directory does not exist: "
                f"{target_temp_dir}"
            )
            return 0

        if not target_temp_dir.is_dir():
            logger.warning(
                f"Temporary storage target is not a directory: "
                f"{target_temp_dir}"
            )
            return 0

        cutoff = (
            datetime.now().timestamp()
            - (older_than_hours * 3600)
        )

        deleted_count = 0

        try:
            items = list(target_temp_dir.iterdir())
        except Exception as read_err:
            logger.warning(
                f"Unable to inspect temporary storage: {read_err}"
            )
            return 0

        for item in items:
            try:
                if item.stat().st_mtime >= cutoff:
                    continue

                if item.is_dir():
                    shutil.rmtree(item)
                elif item.is_file():
                    item.unlink()
                else:
                    continue

                deleted_count += 1

            except FileNotFoundError:
                continue

            except Exception as clear_err:
                logger.debug(
                    f"Unable to remove stale temporary object "
                    f"{item}: {clear_err}"
                )

        if deleted_count:
            logger.info(
                f"Temporary storage cleanup completed: "
                f"{deleted_count} stale item(s) removed."
            )

        return deleted_count

    # ------------------------------------------------------------------
    # STORAGE INFORMATION
    # ------------------------------------------------------------------

    def get_storage_info(self) -> dict[str, Any]:
        try:
            total, used, free = shutil.disk_usage(
                self.base_dir
            )

            video_files: list[Path] = []

            for item in self.base_dir.iterdir():
                try:
                    if (
                        item.is_file()
                        and item.suffix.lower()
                        in self.VIDEO_EXTENSIONS
                    ):
                        video_files.append(item)
                except OSError:
                    continue

            total_video_bytes = 0

            for video in video_files:
                try:
                    total_video_bytes += video.stat().st_size
                except OSError:
                    continue

            return {
                "total_gb": total / (1024 ** 3),
                "used_gb": used / (1024 ** 3),
                "free_gb": free / (1024 ** 3),
                "video_count": len(video_files),
                "video_size_gb": total_video_bytes / (1024 ** 3),
                "video_size_mb": total_video_bytes / (1024 ** 2),
                "output_dir": str(self.base_dir),
            }

        except Exception as storage_err:
            raise StorageError(
                f"Unable to collect storage information: "
                f"{storage_err}"
            ) from storage_err

    # ------------------------------------------------------------------
    # DISK SPACE
    # ------------------------------------------------------------------

    def _check_disk_space(
        self,
        required_gb: float,
    ) -> None:
        if required_gb < 0:
            raise ValueError(
                "required_gb cannot be negative."
            )

        required_bytes = int(
            required_gb * (1024 ** 3)
        )

        self._check_disk_space_for_bytes(
            required_bytes
        )

    def _check_disk_space_for_bytes(
        self,
        required_bytes: int,
    ) -> None:
        if required_bytes < 0:
            raise ValueError(
                "required_bytes cannot be negative."
            )

        try:
            _, _, free_bytes = shutil.disk_usage(
                self.base_dir
            )
        except Exception as disk_err:
            raise DiskSpaceError(
                0,
                0,
            ) from disk_err

        required_with_buffer = (
            required_bytes
            + self.MIN_FREE_BYTES_AFTER_WRITE
        )

        if free_bytes < required_with_buffer:
            available_gb = free_bytes / (1024 ** 3)
            required_gb = required_with_buffer / (1024 ** 3)

            raise DiskSpaceError(
                required_gb,
                available_gb,
            )

    # ------------------------------------------------------------------
    # PATH SAFETY
    # ------------------------------------------------------------------

    def _safe_output_path(
        self,
        filename: str,
    ) -> Path:
        destination = (
            self.base_dir / filename
        ).resolve()

        if not self._is_inside_storage(destination):
            raise StorageError(
                "Generated destination escaped the configured "
                "storage directory."
            )

        return destination

    def _is_inside_storage(
        self,
        path: Path,
    ) -> bool:
        try:
            path.resolve().relative_to(
                self.base_dir
            )
            return True
        except ValueError:
            return False

    def _create_temp_path(
        self,
        suffix: str = ".tmp",
    ) -> Path:
        """
        Creates a temporary filename inside the actual storage directory.

        Keeping the temporary file on the same filesystem allows `replace()`
        to perform an atomic rename.
        """

        fd, raw_path = tempfile.mkstemp(
            prefix=".falconai_",
            suffix=suffix,
            dir=str(self.base_dir),
        )

        Path(raw_path).unlink(missing_ok=True)

        # Close descriptor created by mkstemp.
        import os
        os.close(fd)

        return Path(raw_path)


# ----------------------------------------------------------------------
# NAME SANITIZATION
# ----------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """
    Converts arbitrary user/content names into safe filesystem names.

    Examples:
        "Alex Musliu!" -> "alex_musliu"
        "../../movie" -> "movie"
        "  My Story  " -> "my_story"
    """

    if name is None:
        return ""

    name = str(name).strip()

    if not name:
        return ""

    # Unicode normalization.
    name = unicodedata.normalize(
        "NFKD",
        name,
    )

    # Remove combining marks / accents.
    name = "".join(
        char
        for char in name
        if not unicodedata.combining(char)
    )

    # ASCII fallback keeps filesystem naming predictable.
    name = name.encode(
        "ascii",
        "ignore",
    ).decode("ascii")

    # Normalize whitespace.
    name = re.sub(
        r"\s+",
        "_",
        name,
    )

    # Only allow predictable filename characters.
    name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        name,
    )

    # Prevent path traversal.
    name = name.replace(
        "..",
        "_",
    )

    # Remove leading/trailing unsafe separators.
    name = name.strip(
        "._-"
    )

    # Collapse repeated separators.
    name = re.sub(
        r"_+",
        "_",
        name,
    )

    # Windows reserved device names.
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }

    if name.upper() in reserved:
        name = f"file_{name.lower()}"

    return name[:80].strip("._-")
