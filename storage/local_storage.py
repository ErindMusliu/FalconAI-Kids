import json
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from config.settings import OUTPUT_DIR, TEMP_DIR
from utils.logger import get_logger
from utils.exceptions import StorageError, DiskSpaceError

logger = get_logger(__name__)


class LocalStorage:
    def __init__(self, base_dir: Path = OUTPUT_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_video(
        self,
        video_path: Path,
        name: str,
        metadata: Optional[dict] = None,
    ) -> Path:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Source video file not found at path: {video_path}")

        self._check_disk_space(required_gb=1.0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = _sanitize(name)
        dest_name = f"{safe_name}_{timestamp}{video_path.suffix}"
        dest_path = self.base_dir / dest_name

        try:
            shutil.copy2(str(video_path), str(dest_path))

            if dest_path.stat().st_size != video_path.stat().st_size:
                raise StorageError("Video transfer verification failed due to mismatched byte stream footprints.")
                
            logger.debug(f"Video file persisted cleanly to target tracking path: {dest_path}")
        except Exception as write_err:
            if dest_path.exists():
                dest_path.unlink()
            raise StorageError(f"Failed to copy video binary block to local storage: {write_err}") from write_err

        if metadata:
            meta_path = dest_path.with_suffix(".json")
            try:
                meta_payload = {**metadata, "saved_at": timestamp, "file": dest_name}
                meta_path.write_text(
                    json.dumps(meta_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
            except Exception as meta_err:
                logger.error(f"Failed to write video sidecar metadata payload: {meta_err}")

        return dest_path

    def list_videos(self) -> list[dict]:
        videos = []
        valid_extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

        if not self.base_dir.exists():
            return videos

        all_files = sorted(
            self.base_dir.iterdir(), 
            key=lambda item: item.stat().st_mtime if item.is_file() else 0, 
            reverse=True
        )

        for file_item in all_files:
            if file_item.is_file() and file_item.suffix.lower() in valid_extensions:
                meta_path = file_item.with_suffix(".json")
                meta_payload = {}
                
                if meta_path.exists():
                    try:
                        meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception as parse_err:
                        logger.warning(f"Could not parse corrupted sidecar file metadata for {file_item.name}: {parse_err}")

                try:
                    file_stats = file_item.stat()
                    videos.append({
                        "path": file_item,
                        "name": file_item.name,
                        "size_mb": file_stats.st_size / (1024**2),
                        "created": datetime.fromtimestamp(file_stats.st_mtime),
                        "metadata": meta_payload,
                    })
                except Exception as stat_err:
                    logger.error(f"Bypassed tracking metrics index step for {file_item.name}: {stat_err}")

        return videos

    def delete_video(self, video_path: Path) -> None:
        video_path = Path(video_path)
        
        try:
            if video_path.exists():
                video_path.unlink()
                logger.debug(f"Target video file deleted from path references: {video_path.name}")
            
            meta_path = video_path.with_suffix(".json")
            if meta_path.exists():
                meta_path.unlink()
                logger.debug(f"Associated metadata asset unlinked cleanly: {meta_path.name}")
        except Exception as delete_err:
            raise StorageError(f"Operation aborted; failed to clean up files safely: {delete_err}") from delete_err

    def cleanup_temp(self, older_than_hours: int = 24) -> int:
        deleted_count = 0

        target_temp_dir = Path(TEMP_DIR)
        if not target_temp_dir.exists() or not target_temp_dir.is_dir():
            logger.debug(f"Temporary workspace cleanup pass bypassed; path does not exist: {target_temp_dir}")
            return deleted_count

        cutoff_timestamp = datetime.now().timestamp() - (older_than_hours * 3600)

        for item in target_temp_dir.iterdir():
            try:
                if item.stat().st_mtime < cutoff_timestamp:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    deleted_count += 1
            except Exception as clear_err:
                logger.debug(f"Could not clean up temporary file or directory workspace object {item}: {clear_err}")

        if deleted_count > 0:
            logger.debug(f"Temporary staging area maintenance completed. Cleared {deleted_count} stale cache items.")
        
        return deleted_count

    def get_storage_info(self) -> dict:
        total, used, free = shutil.disk_usage(self.base_dir)
        
        valid_extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
        video_files = [
            f for f in self.base_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        
        total_video_bytes = sum(f.stat().st_size for f in video_files)

        return {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "video_count": len(video_files),
            "video_size_gb": total_video_bytes / (1024**3),
            "output_dir": str(self.base_dir),
        }

    def _check_disk_space(self, required_gb: float) -> None:
        _, _, free_bytes = shutil.disk_usage(self.base_dir)
        available_free_gb = free_bytes / (1024**3)
        if available_free_gb < required_gb:
            raise DiskSpaceError(required_gb, available_free_gb)


def _sanitize(name: str) -> str:
    name = unicodedata.normalize("NFD", name.lower())
    name = name.encode("ascii", "ignore").decode("ascii")

    name = re.sub(r'[^\w\-]', '_', name)
    
    return name[:30].strip('_')
