from utils.logger import get_logger

logger = get_logger(__name__)

class Upscaler:
    def __init__(self, device="cpu", model_path=None):
        self.device = device
        logger.info(f"Upscaler initialized on {device}")

    def run(self, video_path):
        logger.info(f"Skipping upscaling for: {video_path}")
        return video_path