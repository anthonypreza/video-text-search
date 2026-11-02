from typing import TypedDict

import numpy as np
import torch
import tqdm
from numpy.typing import NDArray

from ..core.model import VideoTextContrastiveModel


class VideoIndex(TypedDict):
    embeddings: NDArray[np.float32]
    paths: list[str]
    index_map: dict[str, int]


def build_video_index(
    model: VideoTextContrastiveModel, video_paths: list[str], device: str = "cpu"
) -> VideoIndex:
    """Pre-compute embeddings for all bideos and store in index"""
    model.to(device)
    model.eval()

    video_embeddings: list[NDArray[np.float32]] = []
    valid_paths: list[str] = []

    with torch.no_grad():
        for video_path in tqdm.tqdm(video_paths, desc="Building video index"):
            try:
                # Compute embeddings once and store
                video_embedding = model.encode_video(video_path)
                video_embeddings.append(video_embedding.cpu().numpy())
                valid_paths.append(video_path)
            except Exception as e:
                print(f"Skipping {video_path}: {e}")

    # Stack into matrix [num_videos, embed_dim]
    embedding_matrix = np.vstack(video_embeddings)

    return {
        "embeddings": embedding_matrix,
        "paths": valid_paths,
        "index_map": {path: idx for idx, path in enumerate(valid_paths)},
    }
