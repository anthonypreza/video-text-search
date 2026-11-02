# Video-Text Search with Contrastive Learning

Multimodal video search system that enables natural language queries over video content using contrastive learning.

## Overview

This project implements a video-text retrieval system that learns joint embeddings for videos and text descriptions. Users can search through video collections using natural language queries like "cat playing with toy" or "person cooking pasta".

## Architecture

```
Video Frames → ResNet50 → Video Encoder (512-dim)
                                          ↘
                                      Contrastive Loss
                                          ↗
Text Query → BERT → Text Encoder (512-dim)
```

**Key Components:**

- **Video Encoder**: ResNet50 backbone + temporal aggregation + projection to 512-dim embeddings
- **Text Encoder**: BERT + projection layer with L2 normalization
- **Contrastive Learning**: Temperature-scaled cosine similarity with cross-entropy loss
- **Training**: 1 positive + N negative text descriptions per video

## Features

- **Real Dataset Support**: MSR-VTT dataset integration with YouTube video downloading
- **Synthetic Data**: Fallback synthetic dataset for testing without downloads
- **Flexible Search**: Both linear search and indexed search options
- **Video Processing**: Automatic frame extraction and preprocessing
- **Robust Downloads**: YouTube clip extraction with precise timing using yt-dlp + ffmpeg

## Dataset Options

### MSR-VTT (Recommended)

- **Source**: Microsoft Research Video to Text dataset
- **Content**: ~7,000 YouTube video clips with multiple captions each
- **Download**: Automatic downloading of video segments with precise start/end times
- **Captions**: ~20 human-annotated captions per video clip

### Synthetic (Testing)

- **Categories**: Cats, dogs, cooking, sports
- **Content**: Generated dummy videos with associated text descriptions
- **Use Case**: Quick testing without video downloads

## Quick Start

Examples use a Unix shell and `python` from the activated environment. Optional
GPU environments are described at the end; the pipeline arguments stay the same.

### 0. Install Python dependencies

Use Python 3.12+ and run these commands from the `video-text-search/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the system FFmpeg executable as described below and check `ffmpeg -version`.
The dependency file pins direct Python dependencies; it is not a complete lockfile.
PyTorch GPU support depends on your platform and driver setup; CPU execution is supported.
The first model initialization downloads pretrained ResNet50 and BERT weights.

YouTube changes can require a newer downloader. If extraction stops working, update
it inside the activated environment:

```bash
python -m pip install --upgrade "yt-dlp[default]"
```

For full YouTube support, yt-dlp also recommends a supported JavaScript runtime;
see its [runtime setup guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS).

### 1. Training Phase

Train the model and create the search index (run from video-text-search/ directory):

```bash
# MSR-VTT (recommended)
python scripts/pipeline.py --dataset msrvtt --max-videos 100

# Synthetic dataset (quick test)
python scripts/pipeline.py --dataset synthetic
```

This creates:

- `./models/video_search_model.pth` - Trained model
- `./models/video_index.pkl` - Pre-computed video embeddings cache
- `./models/video_paths.txt` - Video file paths

### Choose the training device

Use `--device auto` (the default), `--device cuda`, or `--device cpu`.
`auto` selects an available GPU, otherwise CPU. `cuda` also names AMD HIP
devices and raises an error if no GPU is available; it does not silently fall
back to CPU. With a compatible GPU-enabled PyTorch environment:

```bash
python scripts/pipeline.py --dataset msrvtt --max-videos 1000 --cached-only --device cuda
python scripts/pipeline.py --dataset msrvtt --max-videos 1000 --cached-only --device cpu
```

Frame extraction decodes sequentially and samples across the actual frame
count, avoiding repeated random seeks through stream-copied H.264 clips.
Decoded frame counts are cached in memory and invalidated when the file changes.
Undecodable inputs raise an error rather than silently becoming blank videos.

### Prepare once, then train from the cache

Downloading and clipping are separate from triplet construction. Preparation
uses three concurrent workers by default (configurable from one to eight),
validates decoded frames, and saves a manifest of usable videos. The requested
maximum counts successful videos; unavailable source clips are skipped.

```bash
# CPU preparation; no model weights are loaded.
python scripts/pipeline.py --dataset msrvtt --max-videos 1000 --prepare-only --download-workers 3

# Train from that manifest without metadata or video download requests.
python scripts/pipeline.py --dataset msrvtt --max-videos 1000 --cached-only
```

FFmpeg must be on PATH for preparation. The default one-command training flow
still prepares clips automatically. Use `--data-dir` to choose the cache and
`--prepared-manifest` to name a particular manifest. Cached-only mode checks
file size and modification time against the validated manifest and rejects
missing or changed clips; it does not prevent the initial pretrained-model
download if those weights are not already cached.

Known unavailable/private/removed clips are recorded and skipped on subsequent
preparation attempts. Transient failures expire after an hour, and rate-limit
failures after 15 minutes. `--retry-failures` explicitly retries recorded failures.
Rate-limit responses pause subsequent attempts across workers, and retry counts
are bounded. Format alternatives are resolved in one yt-dlp extraction attempt.

New downloads are encoded directly as video-only H.264 MP4 clips (CRF 18),
rather than stream-copied at potentially unsafe reference-frame boundaries.
This adds CPU encoding work during preparation but avoids a separate cleanup
pass for newly downloaded clips. Existing cached files are reused unchanged.
Downloads are validated before being promoted to the cache. An invalid prior
clip is preserved separately when a valid replacement succeeds. Avoid running
multiple preparations against the same cache simultaneously.

Negative captions are sampled only from other successfully prepared videos,
excluding every known caption of the positive video. Sampling uses replacement
to keep the number of negatives consistent, even for a small corpus. `--seed`
controls this sampling; it does not seed model training. Source-order selection
keeps results independent of download completion order, although availability
can change the selected videos between preparation runs.

Run the synthetic, offline regression tests with:

```bash
python -m unittest discover -s tests -v
```

### Clean a cache with H.264 decoder warnings

Older stream-copied clips can emit missing-reference warnings even with sequential decoding.
This one-time command checks the prepared clips and creates a separate cache,
re-encoding only files with decoder diagnostics. Original videos are untouched.
Re-encoded video is lossy (H.264 CRF 18); already undecodable image details
cannot be recovered. The command verifies decoded frame counts and new decoder
diagnostics before writing the new manifest.

```bash
python scripts/normalize_video_cache.py --data-dir data/msrvtt --output-dir data/msrvtt-clean
python scripts/pipeline.py --dataset msrvtt --max-videos 1000 --cached-only --device cuda --data-dir data/msrvtt-clean
```

Choose a new output directory for each cleanup attempt. The original cache is
never modified; the second command starts a fresh training run using the clean cache.

### 2. Search Phase

Use the trained model for interactive or command-line search:

```bash
# Interactive search (recommended)
python scripts/search_interactive.py

# Single query search
python scripts/search_cli.py "cat playing with toy"

# JSON output for scripting
python scripts/search_cli.py "cooking pasta" --format json --top-k 3
```

### 3. (Optional) Serve as a Weaviate-backed API

If you want to expose the model through the Rust HTTP API and Weaviate, follow the
deployment guide in [Deploying the Docker API](#deploying-the-docker-api).

## System Requirements

- **Python**: 3.12+
- **ffmpeg**: Required for video processing and YouTube downloads
- **yt-dlp**: For downloading YouTube video clips (installed via pip)

### Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## Configuration

Edit `video_search/core/config.py` to adjust model parameters:

```python
class VideoSearchConfig:
    max_frames = 8  # Frames per video
    frame_size = (224, 224)  # Frame resolution
    text_embed_dim = 512  # Text embedding dimension
    video_embed_dim = 512  # Video embedding dimension
    temperature = 0.07  # Contrastive learning temperature
    num_negatives = 7  # Negative samples per triplet
    batch_size = 16  # Training batch size
    learning_rate = 1e-4  # Adam learning rate
    num_epochs = 10  # Training epochs
```

## Workflow

### Training Process

1. **Data Loading**: Downloads video clips from YouTube with precise timing
2. **Triplet Creation**: Each video caption becomes a positive query with N negative samples
3. **Contrastive Training**: Learns to maximize similarity between videos and their captions
4. **Model & Index Saving**: Automatically saves trained model and search index
5. **Evaluation**: Reports held-out text-to-video Recall@K after MSR-VTT training

### Evaluation scope

MSR-VTT training holds out source videos before creating training triplets and
reports final text-to-video Recall@1, @5, and @10. See **Held-out text-to-video
evaluation** below for the split, candidate-pool size, and limitations.

Synthetic runs and `--test-fraction 0` retain the original first-five-triplet
ranking sanity check (`accuracy_at_1` and `accuracy_at_3`), which is explicitly
labeled as training-data evaluation. Neither mode claims verified benchmark
results or an official MSR-VTT evaluation protocol.

### Search Process

1. **Model Loading**: Loads pre-trained model and video index
2. **Query Processing**: Encodes search query using BERT text encoder
3. **Similarity Search**: Uses pre-computed embeddings with dot product similarity
4. **Results Ranking**: Returns top-k videos ranked by similarity score

## Interactive Search Interface

The interactive search provides a rich user experience:

```
🎥 INTERACTIVE VIDEO SEARCH
============================================================
📊 Index contains 145 videos
💡 Type your search query and press Enter
💡 Type 'quit' or 'exit' to stop
💡 Type 'help' for more options
============================================================

🔍 Search query: cat playing with toy

🎯 Top 5 results:
--------------------------------------------------
 1. cats_5.mp4
    Score: 0.847 [████████████████████] 84.7%
    Path:  /tmp/video_search_test/cats_5.mp4

 2. cats_2.mp4
    Score: 0.723 [██████████████▒▒▒▒▒▒] 72.3%
    Path:  /tmp/video_search_test/cats_2.mp4
```

## Project Structure

### Scripts

- `scripts/pipeline.py` - Training pipeline that saves model and index
- `scripts/search_interactive.py` - Interactive search interface with rich UI
- `scripts/search_cli.py` - Command-line search for single queries and scripting
- `scripts/build_index.py` - Build an index from an already trained model

### Package: `video_search`

- `video_search/core/config.py` - Configuration and save paths
- `video_search/core/model.py` - Contrastive learning model and training/eval
- `video_search/core/encoders.py` - Video and text encoder architectures
- `video_search/search/search.py` - Search routines (linear and indexed)
- `video_search/search/index.py` - Pre-computing and caching video embeddings
- `video_search/utils/data.py` - Dataset loading, MSR-VTT integration, synthetic data
- `video_search/utils/video.py` - Video processing and frame extraction

## Performance Notes

- **CPU Training**: Works on CPU but GPU recommended for larger datasets
- **Memory Usage**: Adjust `batch_size` and `max_frames` based on available RAM
- **Video Downloads**: Some YouTube videos may be unavailable (handled gracefully)
- **Search Speed**: Pre-computed embeddings enable fast similarity search without re-encoding videos
- **Model Persistence**: Train once, search many times - no need to retrain for new queries

## Command Line Options

### Training (`scripts/pipeline.py`)

```bash
python scripts/pipeline.py --dataset [synthetic|msrvtt] --max-videos N
```

### Interactive Search (`scripts/search_interactive.py`)

```bash
python scripts/search_interactive.py [--top-k N] [--model-path PATH] [--index-path PATH]
```

### CLI Search (`scripts/search_cli.py`)

```bash
python scripts/search_cli.py "query" [--top-k N] [--format table|json|simple] [--quiet]
```

## Deploying the Docker API

These steps take you from a trained checkpoint to a Weaviate-backed HTTP API that
serves semantic search results from the `api/` crate.

### 0. Preconditions

- You have already run the training pipeline at least once.
- The `models/` directory contains `video_search_model.pth` and the pre-built
  `video_index.pkl` produced by the pipeline.

### 1. Export the text tower to ONNX (and tokenizer)

Re-run the pipeline with export flags so the API can load the text encoder via
ONNX Runtime. The command below trains if needed, builds the index, exports the
tokenizer, and writes the ONNX file under `models/`:

```bash
python scripts/pipeline.py \
  --dataset msrvtt \
  --max-videos 100 \
  --skip-training \
  --export-onnx-text \
  --export-tokenizer \
  --onnx-outdir models \
  --tokenizer-outdir models
```

Tips:

- Drop `--skip-training` if you want to fine-tune before exporting.
- The ONNX export uses opset 17. You can request `--export-onnx-text`
  even if you already have a model – the command regenerates `models/text_encoder.onnx`.

### 2. Push fresh embeddings to Weaviate

Weaviate needs the latest vectors for the clips you want to serve. Use the helper
script to (re)create the `VideoClip` class and batch upsert the embeddings:

```bash
python scripts/build_and_push_weaviate.py \
  --dataset msrvtt \
  --max-videos 100 \
  --normalize \
  --channel msrvtt
```

Notes:

- Omit `--skip-build` unless you intentionally want to reuse the cached
  `video_index.pkl`. Reusing it repeatedly will duplicate objects in Weaviate.
- The script auto-creates the `VideoClip` schema if it is missing.
- To wipe the class entirely, run:
  `./scripts/delete-weaviate-schema.sh` (requires Weaviate offline).
- The Weaviate class can be created alternatively with `./scripts/create-weaviate-schema.sh`

### 3. Start Weaviate + API via Docker Compose

The repo already ships with a `docker-compose.yml` that launches Weaviate and the
Rust API. Build the image (once per change) and start the stack:

```bash
# From repo root
docker compose build api
docker compose up -d
```

Key details:

- Weaviate listens on `http://localhost:8080`; the API runs on `http://localhost:8088`.
- The compose file mounts `./models` read-only, so the container picks up
  `text_encoder.onnx` and `tokenizer.json` automatically.
- You can follow logs with `docker compose logs -f api`.
- `WEAVIATE_URL` defaults to `http://weaviate:8080` inside Docker. Override via
  env var if you run the API outside Compose.

### 4. Query the HTTP API

Send JSON payloads to `/v1/query/text`:

```bash
curl -s -X POST http://localhost:8088/v1/query/text \
  -H 'Content-Type: application/json' \
  -d '{
        "q": "cat videos",
        "top_k": 5,
        "channel": "msrvtt"
      }' | jq
```

Example output:

```json
{
  "results": [
    {
      "video_id": "video57",
      "title": "video57.mp4",
      "url": "file://./data/msrvtt/videos/video57.mp4",
      "start_s": 0.0,
      "end_s": 0.0,
      "distance": 0.15331495,
      "score": null
    }
  ]
}
```

The response lists the top-k clips with the cosine distance returned by Weaviate.
`ef_search` is optional and currently ignored by Weaviate 1.32’s GraphQL API.

### 5. Verify or debug with helper scripts

- `scripts/query_weaviate.py "cat videos" --onnx models/text_encoder.onnx`
  performs the exact same ONNX + GraphQL flow from your shell.
- `scripts/build_and_push_weaviate.py` ensures the schema exists; rerun it any
  time you refresh embeddings.
- To inspect schema or counts manually:
  ```bash
  curl -s http://localhost:8080/v1/schema | jq
  curl -s -X POST http://localhost:8080/v1/graphql \
       -H 'Content-Type: application/json' \
       -d '{"query":"{Aggregate{VideoClip{meta{count}}}}"}' | jq
  ```

### 6. Cleaning up

Stop the services with `docker compose down`. Include `-v` if you also want to
drop Weaviate’s local volume. Re-run steps 2–4 whenever you produce new model
weights or a fresh ONNX export.

## Example Output

### Interactive Search

```
🔍 Search query: cat playing

🎯 Top 3 results:
--------------------------------------------------
 1. cats_5.mp4
    Score: 0.847 [████████████████████] 84.7%
    Path:  /tmp/video_search_test/cats_5.mp4
```

### CLI JSON Output

```bash
python scripts/search_cli.py "cooking" --format json --top-k 2
```

```json
{
  "query": "cooking",
  "results": [
    {"path": "cooking_3.mp4", "score": 0.923, "filename": "cooking_3.mp4"},
    {"path": "cooking_1.mp4", "score": 0.856, "filename": "cooking_1.mp4"}
  ]
}
```

## Training checkpoints

The pipeline saves model weights, Adam optimizer state, completed epochs, loss
history, and Python/PyTorch random state after every completed epoch. Checkpoints
are written to a temporary file and atomically replace the previous checkpoint.
A crash during an epoch requires repeating that epoch. A machine or disk failure
can still damage files; this is not a backup system.

For a 100-video, five-epoch run using an existing prepared cache:

```bash
python scripts/pipeline.py \
  --dataset msrvtt --max-videos 100 --cached-only --device cuda \
  --epochs 5 --checkpoint checkpoints/msrvtt-100.pt
```

Resume with the same cache, video count, test fraction, seed, and batch size:

```bash
python scripts/pipeline.py \
  --dataset msrvtt --max-videos 100 --cached-only --device cuda \
  --epochs 5 --resume checkpoints/msrvtt-100.pt
```

`--epochs 5` means five total epochs, including completed epochs. To extend the
run, increase that total. `--batch-size` defaults to 16 and must stay the same
when resuming. A dataset/settings mismatch is rejected. Device changes are
allowed, but numerical results need not match across devices or environments.
Only resume checkpoints you trust. Starting a fresh run refuses to overwrite an
existing training checkpoint; choose another `--checkpoint` path instead.
Final model and index outputs retain their existing paths under `models/`.

## Held-out text-to-video evaluation

MSR-VTT training holds out 10% of source videos by default. All clips sharing
the same source URL stay together, and all captions stay with their clip. The
split uses `--seed` (default 42); training negatives are sampled only from the
training partition. Files stay in the existing cache. `--max-videos` counts the
total selected clips before splitting; a 100-clip selection is usually about
90 training / 10 test clips, depending on source grouping.

After the final epoch, each held-out caption ranks every held-out video by
embedding similarity. The pipeline reports Recall@1, @5, and @10, the number of
caption queries, and the number of candidate videos. Recall is the fraction of
queries whose annotated matching clip is in the top K. Every test caption is
used; evaluation does not sample negative candidates. With ten candidates,
Recall@10 is automatically 1.0, so compare results using the same candidate pool.
The model export records the selected training/test video filenames and seed.

Start a fresh run to get an unseen test set:

```bash
python scripts/pipeline.py \
  --dataset msrvtt --max-videos 100 --cached-only --device auto \
  --batch-size 4 --epochs 5 --test-fraction 0.1 \
  --checkpoint checkpoints/msrvtt-100-heldout.pt
```

Resume using the same arguments with `--resume` instead of `--checkpoint`.
Old checkpoints trained on all selected clips cannot produce a valid held-out
test on that selection; the changed training-data signature prevents resuming
them into the new split. `--test-fraction 0` preserves the previous all-data
training behavior and its explicitly labeled training-sample sanity check.
`--skip-training` only rebuilds the search index; it does not run the held-out test.

This is an initial train/test experiment, not an official MSR-VTT benchmark split.
There is no validation set or checkpoint selection by test performance. Run the
test after training and avoid tuning repeatedly against it; add a validation
split when comparing training choices. Exact source URLs are grouped, but this
does not detect duplicate content under different URLs.

## Optional AMD GPU environment on WSL

For Ubuntu 24.04, Python 3.12, and an installed ROCm 6.4.2 WSL runtime:

```bash
./scripts/setup-wsl-gpu.sh
./scripts/run-wsl-gpu.sh -c 'import torch; print(torch.cuda.get_device_name(0))'
./scripts/run-wsl-gpu.sh scripts/pipeline.py --help
```

This uses an isolated `.venv-wsl-gpu` environment with AMD's matching PyTorch
and Torchvision wheels. PyTorch uses the `cuda` device name for AMD HIP too.
The launcher sets the WSL runtime library paths. The environment supports both CPU and GPU execution.
The setup preserves the wheel's bundled HSA libraries inside the environment
and uses the system's WSL-compatible HSA runtime instead, following
[AMD's ROCm 6.4.2 WSL instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.4.2/docs/install/wsl/install-pytorch.html).

Do not install the default `requirements.txt` into this GPU environment: it
selects a different PyTorch build and NumPy version. Existing environments are
unaffected. The pipeline still uses the parameters in `video_search/core/config.py`
and writes its model and index to the configured paths. Back up prior outputs
before rerunning if you want to retain them. Download failures can reduce the
number of usable videos below the requested maximum. MSR-VTT training now uses
a held-out video split and final text-to-video evaluation, described above.

For the commands above, replace `python` with `./scripts/run-wsl-gpu.sh`.
Use `--device cuda` for GPU execution or `--device cpu` for CPU execution.

## Optional native Windows AMD environment

The separate `environments/amd/` uv project targets Windows x86-64, Python 3.12,
and AMD `gfx1100` GPUs with compatible drivers. It uses AMD's PyTorch 2.12.0 and
torchvision 0.27.0 ROCm 7.14.1 builds; do not install `requirements.txt` into it.

Only this optional setup uses PowerShell. From the project directory:

```powershell
uv sync --project environments/amd --locked --python 3.12
.\gpu.ps1 scripts/pipeline.py --help
```

For any Python command above, replace `python` with `.\gpu.ps1` and keep the
same arguments. Enter multiline Unix examples as a single line in PowerShell.
Use `--device cuda` for the AMD GPU or `--device cpu` for CPU.
The launcher selects its own environment, so no activation is required.
`HIP_VISIBLE_DEVICES` can select a GPU when the system has multiple devices.

BERT uses the standard math attention backend on Windows AMD GPUs to avoid a
[ROCm fused-attention issue](https://github.com/ROCm/TheRock/issues/7992).
The model architecture and training objective are unchanged; this backend can
use more memory or run slower than fused attention.
