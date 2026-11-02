"""Bounded clip preparation with atomic files and resumable failure records."""

import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

FORMAT = (
    "bestvideo*[height<=480][ext=mp4][vcodec^=avc1]/"
    "bestvideo*[height<=480][ext=mp4]/bestvideo*[height<=480]/bestvideo*"
)


@dataclass
class DownloadResult:
    status: str
    error: str = ""

    @property
    def ok(self):
        return self.status in {"downloaded", "cached"}


class DownloadGate:
    """A rate-limit response pauses subsequent attempts across all workers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.resume_at = 0.0

    def defer(self, seconds):
        with self.lock:
            self.resume_at = max(self.resume_at, time.monotonic() + seconds)

    def wait(self):
        while True:
            with self.lock:
                delay = self.resume_at - time.monotonic()
            if delay <= 0:
                return
            time.sleep(delay)


def check_download_tools():
    if importlib.util.find_spec("yt_dlp") is None:
        raise RuntimeError("yt-dlp is not installed in this Python environment.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg executable was not found on PATH.")


def valid_clip(path):
    """Check decoded frames without trusting frame-count metadata in cut clips."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        decoded = False
        while True:
            readable, image = capture.read()
            if not readable:
                return decoded
            if image is None or image.size == 0:
                return False
            decoded = True
    finally:
        capture.release()


def classify_error(error):
    error = error.lower()
    if any(
        term in error
        for term in (
            "429",
            "too many requests",
            "rate limit",
            "confirm you're not a bot",
            "confirm you’re not a bot",
        )
    ):
        return "rate_limited"
    if any(
        term in error
        for term in (
            "video unavailable",
            "video is unavailable",
            "video is not available",
            "private video",
            "has been removed",
            "has been deleted",
            "account has been terminated",
            "not available in your country",
            "sign in to confirm your age",
            "members-only",
        )
    ):
        return "unavailable"
    return "transient"


def run_downloader(command, timeout=120):
    # Kill the process group on timeout so FFmpeg is not left running.
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name != "nt",
    )
    try:
        _, error = process.communicate(timeout=timeout)
        return process.returncode, error
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.communicate()
        return 1, "Download timed out"


def download_clip(
    url, output_path, start_time, end_time, *, gate=None, retries=2, retry_backoff=5
):
    check_download_tools()
    if end_time <= start_time:
        return DownloadResult("unavailable", "Clip end time must exceed start time")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    gate = gate or DownloadGate()
    error = ""
    for attempt in range(retries + 1):
        gate.wait()
        # Use a sibling directory: Snap FFmpeg has a private /tmp namespace.
        with tempfile.TemporaryDirectory(
            prefix=f"{output.stem}-download-", dir=output.parent
        ) as directory:
            pattern = str(Path(directory) / "clip.mp4")
            command = [
                sys.executable,
                "-m",
                "yt_dlp",
                "--no-playlist",
                "--format",
                FORMAT,
                "--retries",
                "0",
                "--fragment-retries",
                "0",
                "--extractor-retries",
                "0",
                "--socket-timeout",
                "20",
                "--external-downloader",
                "ffmpeg",
                "--external-downloader-args",
                f"ffmpeg_i:-ss {start_time} -t {end_time - start_time}",
                # Encode at the cut boundary so the saved clip has its own
                # reference frames. Output options override yt-dlp's stream copy.
                "--external-downloader-args",
                "ffmpeg_o:-map 0:v:0 -an -c:v libx264 -preset fast -crf 18 "
                "-vsync 0 -pix_fmt yuv420p -movflags +faststart -f mp4",
                "-o",
                pattern,
                url,
            ]
            code, error = run_downloader(command)
            if code == 0:
                candidates = [
                    p
                    for p in Path(directory).glob("clip.*")
                    if p.suffix in {".mp4", ".webm", ".mkv", ".mov"}
                ]
                if len(candidates) == 1 and valid_clip(candidates[0]):
                    if output.exists():
                        # Preserve an invalid original rather than deleting user data.
                        quarantine = output.parent / "invalid-clips"
                        quarantine.mkdir(exist_ok=True)
                        shutil.move(
                            str(output),
                            str(
                                quarantine
                                / f"{output.stem}-{time.time_ns()}{output.suffix}"
                            ),
                        )
                    os.replace(candidates[0], output)
                    return DownloadResult("downloaded")
                error = "Downloaded clip could not be decoded by OpenCV"
            kind = classify_error(error)
            if kind == "unavailable":
                return DownloadResult(kind, error.strip()[-1200:])
            delay = min(
                60, (30 if kind == "rate_limited" else retry_backoff) * 2**attempt
            )
            if kind == "rate_limited":
                gate.defer(delay)
            elif attempt < retries:
                time.sleep(delay)
    return DownloadResult(classify_error(error), error.strip()[-1200:])


def download_youtube_clip(
    youtube_url, output_path, start_time, end_time, verbose=False
):
    """Compatibility wrapper used by existing scripts; returns a Boolean."""
    result = download_clip(youtube_url, output_path, start_time, end_time)
    if not result.ok:
        print(f"Failed to download {youtube_url}: {result.status}\n{result.error}")
    return result.ok


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def request_signature(item):
    value = [item["url"], item["start_time"], item["end_time"]]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def ensure_clip(item, video_dir, failure_dir, retry_failures, gate):
    video_id = str(item["video_id"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise ValueError(f"Unsafe video ID: {video_id!r}")
    output = Path(video_dir) / f"{video_id}.mp4"
    if valid_clip(output):
        return DownloadResult("cached")
    failure = Path(failure_dir) / f"{video_id}.json"
    signature = request_signature(item)
    if failure.exists() and not retry_failures:
        try:
            record = json.loads(failure.read_text())
            old_validation_failure = (
                record.get("error") == "Downloaded clip could not be decoded by OpenCV"
                and record.get("validation_version", 0) < 2
            )
            if (
                not old_validation_failure
                and record["signature"] == signature
                and (
                    record["status"] == "unavailable"
                    or record["retry_after"] > time.time()
                )
            ):
                return DownloadResult("failure_cached", record["status"])
        except (ValueError, KeyError, TypeError, OSError):
            pass
    result = download_clip(
        item["url"], output, item["start_time"], item["end_time"], gate=gate
    )
    if result.ok:
        failure.unlink(missing_ok=True)
    else:
        atomic_json(
            failure,
            {
                "signature": signature,
                "status": result.status,
                "error": result.error,
                "validation_version": 2,
                "failed_at": time.time(),
                "retry_after": time.time()
                + (900 if result.status == "rate_limited" else 3600),
            },
        )
    return result


def prepare_clips(
    items, target, video_dir, failure_dir, workers=3, retry_failures=False
):
    """Keep workers busy without exceeding the target number of successful clips."""
    if workers < 1 or workers > 8:
        raise ValueError("Download workers must be between 1 and 8")
    if target < 1:
        raise ValueError("Requested video count must be positive")
    selected, counts, offset, pending = [], {}, 0, {}
    gate = DownloadGate()
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        tqdm(total=target, desc="Preparing videos") as progress,
    ):
        while pending or (len(selected) < target and offset < len(items)):
            # Count in-flight jobs against the remaining target, so no surplus
            # successful downloads are launched just to keep a worker occupied.
            while (
                len(pending) < workers
                and offset < len(items)
                and len(selected) + len(pending) < target
            ):
                item = items[offset]
                future = executor.submit(
                    ensure_clip, item, video_dir, failure_dir, retry_failures, gate
                )
                pending[future] = (offset, item)
                offset += 1
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                index, item = pending.pop(future)
                result = future.result()
                counts[result.status] = counts.get(result.status, 0) + 1
                if result.ok:
                    selected.append((index, item))
                    progress.update(1)
                elif result.status != "failure_cached":
                    tqdm.write(
                        f"{item['video_id']}: {result.status}: {result.error[-180:]}"
                    )
    print(f"Video preparation: {len(selected)}/{target} usable; {counts}")
    if len(selected) < target:
        print("Source exhausted before the requested usable-video count was reached.")
    # Completion order affects neither the manifest nor negative sampling.
    return [item for _, item in sorted(selected, key=lambda pair: pair[0])]
