"""Copy a prepared cache, re-encoding only clips with decoder diagnostics."""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
from tqdm import tqdm


def inspect_video(path):
    # This standalone command runs serially; capture native decoder diagnostics
    # without hiding them in the training process or changing its log settings.
    with tempfile.TemporaryFile() as errors:
        previous = os.dup(2)
        os.dup2(errors.fileno(), 2)
        try:
            capture = cv2.VideoCapture(str(path))
            try:
                count = 0
                while True:
                    readable, frame = capture.read()
                    if not readable:
                        break
                    if frame is None or frame.size == 0:
                        raise ValueError(f"Invalid decoded frame: {path}")
                    count += 1
            finally:
                capture.release()
        finally:
            os.dup2(previous, 2)
            os.close(previous)
        errors.seek(0)
        return count, errors.read().decode(errors="replace")


def normalize_cache(data_dir, output_dir):
    source = Path(data_dir).resolve()
    output = Path(output_dir).resolve()
    if (
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError("Output must be a separate cache directory")
    if output.exists():
        raise ValueError("Output directory already exists; choose a new path")
    manifest = json.loads((source / "prepared.json").read_text())
    if manifest.get("version") != 1:
        raise ValueError("Unsupported prepared manifest version")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg must be on PATH")
    videos = output / "videos"
    videos.mkdir(parents=True)
    records = []
    repaired = 0
    for record in tqdm(manifest["videos"], desc="Checking video cache"):
        video_id = str(record["video_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
            raise ValueError("Unsafe video ID in manifest")
        original = source / "videos" / f"{video_id}.mp4"
        target = videos / original.name
        original_stat = original.stat()
        if (original_stat.st_size, original_stat.st_mtime_ns) != (
            record["size"],
            record["mtime_ns"],
        ):
            raise ValueError(f"Source changed since preparation: {video_id}")
        count, diagnostic = inspect_video(original)
        if count == 0:
            raise ValueError(f"No frames decode from {video_id}; prepare a replacement")
        if diagnostic.strip():
            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(original),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-vsync",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(target),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode:
                raise RuntimeError(
                    f"Re-encoding failed for {video_id}: {result.stderr[-1000:]}"
                )
            new_count, new_diagnostic = inspect_video(target)
            if new_count != count or new_diagnostic.strip():
                raise ValueError(
                    f"Re-encoding did not preserve clean decoded frames for {video_id}"
                )
            repaired += 1
            tqdm.write(f"Re-encoded {video_id}: {count} decoded frames")
        else:
            shutil.copy2(original, target)
        stat = target.stat()
        records.append({**record, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    # A failed or interrupted cleanup never leaves a manifest claiming success.
    (output / "prepared.json").write_text(
        json.dumps({"version": 1, "videos": records}, indent=2)
    )
    print(
        f"Prepared {len(records)} clips; re-encoded {repaired}. Clean cache: {output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/msrvtt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    normalize_cache(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
