import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from config.settings import OUTPUT_DIR, TEMP_DIR
from utils.logger import get_logger
from utils.exceptions import StorageError, DiskSpaceError

logger = get_logger(__name__)

class LocalStorage:
    def __init__(self, base_dir: Path = OUTPUT_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_video(
        self,
        video_path: Path,
        name: str,
        metadata: Optional[dict] = None,
    ) -> Path:
        self._check_disk_space(required_gb=1.0)

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name  = _sanitize(name)
        dest_name  = f"{safe_name}_{timestamp}{video_path.suffix}"
        dest_path  = self.base_dir / dest_name

        shutil.copy2(str(video_path), str(dest_path))
        logger.debug(f"Video u ruajt: {dest_path}")

        if metadata:
            meta_path = dest_path.with_suffix(".json")
            meta_path.write_text(
                json.dumps({**metadata, "saved_at": timestamp, "file": dest_name},
                           ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        return dest_path

    def list_videos(self) -> list[dict]:
        videos = []
        for f in sorted(self.base_dir.glob("*.mp4"), reverse=True):
            meta_path = f.with_suffix(".json")
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            videos.append({
                "path"     : f,
                "name"     : f.name,
                "size_mb"  : f.stat().st_size / (1024**2),
                "created"  : datetime.fromtimestamp(f.stat().st_mtime),
                "metadata" : meta,
            })
        return videos

    def delete_video(self, video_path: Path) -> None:
        video_path = Path(video_path)
        if video_path.exists():
            video_path.unlink()
            meta = video_path.with_suffix(".json")
            if meta.exists():
                meta.unlink()
            logger.debug(f"Video u fshi: {video_path.name}")

    def cleanup_temp(self, older_than_hours: int = 24) -> int:
        count = 0
        cutoff = datetime.now().timestamp() - (older_than_hours * 3600)

        for item in TEMP_DIR.iterdir():
            try:
                if item.stat().st_mtime < cutoff:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    count += 1
            except Exception as e:
                logger.debug(f"Nuk u fshi {item}: {e}")

        if count:
            logger.debug(f"Temp cleanup: {count} file/folder u fshinë")
        return count

    def get_storage_info(self) -> dict:
        total, used, free = shutil.disk_usage(self.base_dir)
        videos = list(self.base_dir.glob("*.mp4"))
        video_size = sum(f.stat().st_size for f in videos)

        return {
            "total_gb"       : total / (1024**3),
            "used_gb"        : used  / (1024**3),
            "free_gb"        : free  / (1024**3),
            "video_count"    : len(videos),
            "video_size_gb"  : video_size / (1024**3),
            "output_dir"     : str(self.base_dir),
        }

    def _check_disk_space(self, required_gb: float) -> None:
        _, _, free = shutil.disk_usage(self.base_dir)
        free_gb = free / (1024**3)
        if free_gb < required_gb:
            raise DiskSpaceError(required_gb, free_gb)

def _sanitize(name: str) -> str:
    import re, unicodedata
    name = unicodedata.normalize("NFD", name.lower())
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r'[^\w]', '_', name)
    return name[:30].strip('_')