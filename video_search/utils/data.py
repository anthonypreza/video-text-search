import json
import os
import random
import re
from pathlib import Path
from typing import TypedDict, cast

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .downloads import atomic_json, prepare_clips
from .video import VideoProcessor


class VideoQueryTriplet:
    """Data structure for training triplets: (video_path, positive_query, negative_queries)"""

    def __init__(
        self, video_path: str, positive_query: str, negative_queries: list[str]
    ):
        self.video_path = video_path
        self.positive_query = positive_query
        self.negative_queries = negative_queries

    def get_all_queries(self) -> list[str]:
        """Returns [positive_query, negative_query1, negative_query2, ...]"""
        return [self.positive_query] + self.negative_queries

    def get_target_index(self) -> int:
        """Returns the index of the positive query (always 0)"""
        return 0


class VideoQuerySample(TypedDict):
    video_frames: torch.Tensor
    text_input_ids: torch.Tensor
    text_attention_mask: torch.Tensor
    target_index: torch.Tensor
    video_path: str
    queries: list[str]


class VideoQueryDataset(Dataset[VideoQuerySample]):
    """Dataset class for loading and processing video-query triplets"""

    def __init__(
        self,
        triplets: list[VideoQueryTriplet],
        video_processor: VideoProcessor,
        tokenizer: PreTrainedTokenizerBase,
        max_text_length: int = 128,
    ):
        self.triplets = triplets
        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> VideoQuerySample:
        triplet = self.triplets[idx]

        # Extract video frames
        video_frames = self.video_processor.extract_frames(triplet.video_path)

        # Get all queries (positive + negatives)
        all_queries = triplet.get_all_queries()
        target_index = triplet.get_target_index()

        # Tokenize all queries
        encoding = self.tokenizer(
            all_queries,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        return {
            "video_frames": video_frames,  # [num_frames, 3, H, W]
            "text_input_ids": cast(
                torch.Tensor, encoding["input_ids"]
            ),  # [num_texts, seq_len]
            "text_attention_mask": cast(
                torch.Tensor, encoding["attention_mask"]
            ),  # [num_texts, seq_len]
            "target_index": torch.tensor(target_index, dtype=torch.long),  # scalar
            "video_path": triplet.video_path,  # for debugging
            "queries": all_queries,  # for debugging
        }


def _read_prepared_manifest(data_dir, manifest_path, max_videos):
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise ValueError("Prepared manifest missing; run --prepare-only first")
    content = json.loads(manifest.read_text())
    if content.get("version") != 1:
        raise ValueError("Unsupported prepared manifest version")
    records = content["videos"]
    if max_videos is not None:
        if len(records) < max_videos:
            raise ValueError(
                f"Manifest has {len(records)} videos, requested {max_videos}; prepare more videos first"
            )
        records = records[:max_videos]
    for record in records:
        video_id = str(record["video_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
            raise ValueError("Unsafe video ID in prepared manifest")
        path = Path(data_dir) / "videos" / f"{video_id}.mp4"
        if not path.is_file():
            raise ValueError(
                f"Prepared video missing: {video_id}; rerun --prepare-only"
            )
        stat = path.stat()
        if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError(
                f"Prepared video changed: {video_id}; rerun --prepare-only to validate it"
            )
    return records


def prepare_msrvtt_dataset(
    data_dir="./data/msrvtt",
    max_videos=None,
    *,
    download_workers=3,
    retry_failures=False,
    cached_only=False,
    manifest_path=None,
):
    """Prepare decodable clips once; cached-only reads the resulting manifest offline."""
    if max_videos is not None and max_videos < 1:
        raise ValueError("max_videos must be positive")
    manifest_path = (
        Path(manifest_path) if manifest_path else Path(data_dir) / "prepared.json"
    )
    if cached_only:
        return _read_prepared_manifest(data_dir, manifest_path, max_videos)
    print("Loading MSR-VTT metadata...")
    train_data = load_dataset("friedrichor/MSR-VTT", "train_7k")["train"]
    grouped = {}
    # Keep all candidates available: max_videos limits successful clips, not rows.
    for item in tqdm(train_data, desc="Processing captions"):
        video_id = str(item["video_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
            raise ValueError(f"Unsafe video ID: {video_id!r}")
        captions = (
            item["caption"] if isinstance(item["caption"], list) else [item["caption"]]
        )
        if video_id not in grouped:
            start = float(item.get("start time") or 0)
            end = float(item.get("end time") or start + 10)
            grouped[video_id] = {
                "video_id": video_id,
                "url": item["url"],
                "start_time": start,
                "end_time": end,
                "captions": [],
            }
        grouped[video_id]["captions"].extend(
            c for c in captions if isinstance(c, str) and c.strip()
        )
    items = []
    for record in grouped.values():
        record["captions"] = list(dict.fromkeys(record["captions"]))
        if record["captions"]:
            items.append(record)
    video_dir = Path(data_dir) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    selected = prepare_clips(
        items,
        max_videos or len(items),
        video_dir,
        Path(data_dir) / "download-failures",
        download_workers,
        retry_failures,
    )
    if not selected:
        raise ValueError("No usable videos were prepared; check the download failures")
    records = []
    for item in selected:
        stat = (video_dir / f"{item['video_id']}.mp4").stat()
        records.append({**item, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    atomic_json(manifest_path, {"version": 1, "videos": records})
    print(f"Prepared manifest: {manifest_path}")
    return records


def build_msrvtt_triplets(records, data_dir, num_negatives=7, seed=42):
    """Sample another usable video, then a caption; never use same-video positives."""
    if num_negatives < 1:
        raise ValueError("num_negatives must be positive")
    if len(records) < 2:
        raise ValueError(
            "At least two usable videos are required for cross-video negatives"
        )
    rng = random.Random(seed)
    triplets = []
    for index, record in enumerate(tqdm(records, desc="Building triplets")):
        positives = set(record["captions"])
        fallback = None
        for caption in record["captions"]:
            negatives = []
            attempts = 0
            while len(negatives) < num_negatives and attempts < max(
                100, num_negatives * 20
            ):
                other_index = rng.randrange(len(records) - 1)
                if other_index >= index:
                    other_index += 1
                candidate = rng.choice(records[other_index]["captions"])
                if candidate not in positives:
                    negatives.append(candidate)
                attempts += 1
            if len(negatives) < num_negatives:
                # Rare fallback for very repetitive data, computed once per video.
                if fallback is None:
                    fallback = [
                        candidate
                        for other in records
                        if other["video_id"] != record["video_id"]
                        for candidate in other["captions"]
                        if candidate not in positives
                    ]
                if not fallback:
                    raise ValueError(
                        f"No distinct cross-video negative captions for {record['video_id']}"
                    )
                negatives.extend(
                    rng.choices(fallback, k=num_negatives - len(negatives))
                )
            triplets.append(
                VideoQueryTriplet(
                    str(Path(data_dir) / "videos" / f"{record['video_id']}.mp4"),
                    caption,
                    negatives,
                )
            )
    print(f"Created {len(triplets)} triplets from {len(records)} usable videos")
    return triplets


def load_msrvtt_dataset(
    data_dir="./data/msrvtt",
    max_videos=None,
    num_negatives=7,
    *,
    download_workers=3,
    retry_failures=False,
    cached_only=False,
    manifest_path=None,
    seed=42,
):
    records = prepare_msrvtt_dataset(
        data_dir,
        max_videos,
        download_workers=download_workers,
        retry_failures=retry_failures,
        cached_only=cached_only,
        manifest_path=manifest_path,
    )
    return build_msrvtt_triplets(records, data_dir, num_negatives, seed)


def create_synthetic_dataset(
    num_videos: int = 20, num_negatives: int = 7
) -> list[VideoQueryTriplet]:
    """Create a synthetic dataset for testing"""

    # Create some dummy videos
    video_dir = "/tmp/video_search_test"
    os.makedirs(video_dir, exist_ok=True)

    # Video categories and their associated queries
    categories = {
        "cats": {
            "positive_queries": [
                "cat playing with toy",
                "kitten running around",
                "cat sleeping peacefully",
                "cats chasing laser pointer",
                "cute cat meowing",
                "tabby cat eating",
            ],
            "video_style": "cat_video",
        },
        "dogs": {
            "positive_queries": [
                "dog running in park",
                "puppy playing fetch",
                "golden retriever swimming",
                "dog barking at stranger",
                "cute puppy sleeping",
                "dog catching frisbee",
            ],
            "video_style": "dog_video",
        },
        "cooking": {
            "positive_queries": [
                "cooking pasta in kitchen",
                "chopping vegetables",
                "baking chocolate cake",
                "frying eggs in pan",
                "making pizza dough",
                "grilling chicken",
            ],
            "video_style": "cooking_video",
        },
        "sports": {
            "positive_queries": [
                "basketball player shooting",
                "soccer match goal",
                "tennis player serving",
                "running marathon race",
                "swimming competition",
                "football touchdown",
            ],
            "video_style": "sports_video",
        },
    }

    # All possible queries for negatives
    all_queries = []
    for cat_data in categories.values():
        all_queries.extend(cat_data["positive_queries"])

    triplets = []
    video_processor = VideoProcessor()

    for i in range(num_videos):
        # Choose random category
        category = random.choice(list(categories.keys()))
        cat_data = categories[category]

        # Create video file
        video_path = os.path.join(video_dir, f"{category}_{i}.mp4")
        if not os.path.exists(video_path):
            video_processor.create_dummy_video(video_path, duration_seconds=3)

        # Choose positive query
        positive_query = random.choice(cat_data["positive_queries"])

        # Choose negative queries (from other categories or different queries from same category)
        other_queries = [q for q in all_queries if q != positive_query]
        negative_queries = random.sample(
            other_queries, min(num_negatives, len(other_queries))
        )

        # Create triplet
        triplet = VideoQueryTriplet(
            video_path=video_path,
            positive_query=positive_query,
            negative_queries=negative_queries,
        )

        triplets.append(triplet)

    return triplets
