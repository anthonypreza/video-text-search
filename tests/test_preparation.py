"""Behavior checks with local videos and substitutes at network boundaries."""

import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from video_search.utils import data, downloads
from video_search.utils.video import VideoProcessor


def sample_records(count):
    return [
        {
            "video_id": f"video{i}",
            "url": f"https://example.invalid/{i}",
            "caption": ["shared caption", f"unique scene {i}"],
        }
        for i in range(count)
    ]


def write_video(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 32))
    if not writer.isOpened():
        raise RuntimeError("Cannot create the local video fixture")
    try:
        for value in range(5):
            writer.write(np.full((32, 32, 3), value * 20, dtype=np.uint8))
    finally:
        writer.release()


def write_cut_video(path):
    """Generate original footage, then cut between keyframes like the downloader."""
    path = Path(path)
    source = path.with_name("source.mp4")
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64))
    if not writer.isOpened():
        raise RuntimeError("Cannot create the local video fixture")
    try:
        for index in range(150):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[10:25, index % 40 : index % 40 + 15] = 200
            writer.write(image)
    finally:
        writer.release()
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-v",
            "error",
            "-ss",
            "1.37",
            "-i",
            str(source),
            "-t",
            "2",
            "-c",
            "copy",
            str(path),
        ],
        check=True,
    )
    source.unlink()


def fetch_local_video(url, output, *args, **kwargs):
    write_video(output)
    return downloads.DownloadResult("downloaded")


class PreparationTests(unittest.TestCase):
    def test_triplets_use_only_other_videos_captions(self):
        records = [
            {"video_id": item["video_id"], "captions": item["caption"]}
            for item in sample_records(3)
        ]
        triplets = data.build_msrvtt_triplets(records, "/unused", num_negatives=4)
        self.assertCountEqual(
            [t.positive_query for t in triplets],
            [caption for item in records for caption in item["captions"]],
        )
        for triplet in triplets:
            source = next(
                item
                for item in records
                if item["video_id"] == Path(triplet.video_path).stem
            )
            allowed = {
                caption
                for item in records
                if item is not source
                for caption in item["captions"]
            } - set(source["captions"])
            self.assertIn(triplet.positive_query, source["captions"])
            self.assertEqual(len(triplet.negative_queries), 4)
            self.assertTrue(set(triplet.negative_queries) <= allowed)

    def test_preparation_replaces_failures_and_reuses_cached_outcomes(self):
        workers = 2
        barrier = threading.Barrier(workers, timeout=5)
        lock = threading.Lock()
        active = peak = 0
        first_round = True
        unavailable = True

        def fetch(url, output, *args, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if first_round and url in {
                    "https://example.invalid/0",
                    "https://example.invalid/1",
                }:
                    barrier.wait()
                if url.endswith("/0") and unavailable:
                    if not first_round:
                        self.fail("Previously unavailable video was downloaded again")
                    return downloads.DownloadResult("unavailable")
                return fetch_local_video(url, output)
            finally:
                with lock:
                    active -= 1

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                data, "load_dataset", return_value={"train": sample_records(5)}
            ),
            patch.object(downloads, "download_clip", side_effect=fetch),
        ):
            prepared = data.prepare_msrvtt_dataset(
                directory, 2, download_workers=workers
            )
            self.assertEqual(
                [item["video_id"] for item in prepared], ["video1", "video2"]
            )
            self.assertGreater(peak, 1)
            self.assertLessEqual(peak, workers)

            first_round = False
            with patch.object(
                downloads,
                "download_clip",
                side_effect=AssertionError("Unexpected network access"),
            ):
                reused = data.prepare_msrvtt_dataset(
                    directory, 2, download_workers=workers
                )
            self.assertEqual(
                [item["video_id"] for item in reused],
                [item["video_id"] for item in prepared],
            )

            unavailable = False
            retried = data.prepare_msrvtt_dataset(
                directory, 2, download_workers=workers, retry_failures=True
            )
            self.assertEqual(
                [item["video_id"] for item in retried], ["video0", "video1"]
            )

    def test_prepared_data_loads_offline_and_rejects_changed_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    data, "load_dataset", return_value={"train": sample_records(2)}
                ),
                patch.object(downloads, "download_clip", side_effect=fetch_local_video),
            ):
                prepared = data.prepare_msrvtt_dataset(directory, 2)

            with (
                patch.object(
                    data,
                    "load_dataset",
                    side_effect=AssertionError("Unexpected metadata request"),
                ),
                patch.object(
                    downloads,
                    "download_clip",
                    side_effect=AssertionError("Unexpected video request"),
                ),
            ):
                cached = data.prepare_msrvtt_dataset(directory, 2, cached_only=True)
                self.assertEqual(cached, prepared)
                (Path(directory) / "videos/video0.mp4").write_bytes(b"corrupted")
                with self.assertRaises(ValueError):
                    data.prepare_msrvtt_dataset(directory, 2, cached_only=True)

    def test_download_promotes_only_decodable_clips_and_preserves_originals(self):
        # Snap FFmpeg cannot see the host /tmp directory; use a temporary sibling.
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1]
        ) as directory:
            output = Path(directory) / "clip.mp4"
            original = b"original invalid clip"
            output.write_bytes(original)
            processor = VideoProcessor(max_frames=8, frame_size=(32, 32))
            with self.assertRaises(ValueError):
                processor.extract_frames(str(output))

            def fetch(command, valid):
                # Substitute yt-dlp at its process boundary, using its output path.
                destination = Path(
                    command[command.index("-o") + 1].replace("%(ext)s", "mp4")
                )
                if valid:
                    write_cut_video(destination)
                else:
                    destination.write_bytes(b"invalid downloaded clip")
                return 0, ""

            with (
                patch.object(downloads, "check_download_tools"),
                patch.object(
                    downloads,
                    "run_downloader",
                    side_effect=lambda command: fetch(command, False),
                ),
            ):
                failed = downloads.download_clip(
                    "https://example.invalid", output, 0, 1, retries=0
                )
            self.assertFalse(failed.ok)
            self.assertEqual(output.read_bytes(), original)

            with (
                patch.object(downloads, "check_download_tools"),
                patch.object(
                    downloads,
                    "run_downloader",
                    side_effect=lambda command: fetch(command, True),
                ),
            ):
                succeeded = downloads.download_clip(
                    "https://example.invalid", output, 0, 1
                )
            self.assertTrue(succeeded.ok)
            self.assertTrue(downloads.valid_clip(output))
            frames = processor.extract_frames(str(output))
            self.assertEqual(tuple(frames.shape), (8, 3, 32, 32))
            self.assertTrue(torch.isfinite(frames).all())
            self.assertTrue(frames.flatten(1).ne(0).any(dim=1).all())
            self.assertTrue(
                any(
                    path.read_bytes() == original
                    for path in Path(directory).rglob("*.mp4")
                )
            )


if __name__ == "__main__":
    unittest.main()
