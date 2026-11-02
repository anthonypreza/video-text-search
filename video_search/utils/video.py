import os
from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


@lru_cache(maxsize=4096)
def _decoded_frame_count(path, size, mtime_ns):
    # File metadata only invalidates the cache; container frame counts are not
    # reliable for stream-copied clips cut between keyframes.
    capture = cv2.VideoCapture(path)
    try:
        count = 0
        while True:
            readable, image = capture.read()
            if not readable:
                break
            if image is None or image.size == 0:
                raise ValueError(f"Invalid decoded frame in {path}")
            count += 1
        return count
    finally:
        capture.release()


class VideoProcessor:
    """Handles video frame extraction and preprocessing"""

    def __init__(self, max_frames: int = 8, frame_size: tuple[int, int] = (224, 224)):
        self.max_frames = max_frames
        self.frame_size = frame_size

        # Image preprocessing pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize(frame_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_frames(self, video_path: str) -> torch.Tensor:
        """Sample actual decoded frames without seeking through H.264 references."""
        path = os.path.abspath(video_path)
        try:
            stat = os.stat(path)
            total_frames = _decoded_frame_count(path, stat.st_size, stat.st_mtime_ns)
            if total_frames == 0:
                raise ValueError("No decodable frames")
            frame_indices = set(
                np.linspace(
                    0, total_frames - 1, min(total_frames, self.max_frames), dtype=int
                ).tolist()
            )
            capture = cv2.VideoCapture(path)
            frames = []
            try:
                for index in range(total_frames):
                    readable, frame = capture.read()
                    if not readable:
                        raise ValueError(
                            "Clip stopped decoding before the expected frame count"
                        )
                    if index in frame_indices:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(self.transform(Image.fromarray(rgb)))
            finally:
                capture.release()
            tensor = torch.stack(frames)
            if len(frames) < self.max_frames:
                padding = torch.zeros(
                    (self.max_frames - len(frames), 3, *self.frame_size)
                )
                tensor = torch.cat([tensor, padding], dim=0)
            return tensor
        except (OSError, ValueError, cv2.error) as error:
            raise ValueError(
                f"Cannot extract frames from {video_path}: {error}"
            ) from error

    def create_dummy_video(self, save_path: str, duration_seconds: int = 5):
        """Create a dummy video for testing purposes"""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(save_path, fourcc, 30.0, (640, 480))

        for i in range(duration_seconds * 30):  # 30 FPS
            # Create a simple colored frame
            color = (i % 256, (i * 2) % 256, (i * 3) % 256)
            frame = np.full((480, 640, 3), color, dtype=np.uint8)
            out.write(frame)

        out.release()
        print(f"Dummy video created at: {save_path}")
