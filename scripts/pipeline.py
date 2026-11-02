import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure the project root (containing the `video_search` package) is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from video_search.core.config import VideoSearchConfig
from video_search.core.encoders import TextEncoder
from video_search.core.model import (
    VideoTextContrastiveModel,
    evaluate_retrieval,
    train_video_search_model,
)
from video_search.search.index import build_video_index
from video_search.search.search import search_videos_indexed
from video_search.utils.data import (
    VideoQueryDataset,
    create_synthetic_dataset,
    load_msrvtt_dataset,
    prepare_msrvtt_dataset,
)
from video_search.utils.video import VideoProcessor


def _ensure_outdir(path: str) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def export_text_tower_onnx(
    model, tokenizer, config, outdir="models", filename="text_encoder.onnx"
):
    """
    Exports TextEncoder as ONNX:
      inputs:  input_ids[int64][B, L], attention_mask[int64][B, L]
      outputs: embedding[float32][B, D]  (L2-normalized by your encoder)
    """
    model.eval()
    text_encoder = model.text_encoder.eval().to(config.device)

    # Dummy batch (dynamic axes let you vary B and L at runtime)
    dummy = tokenizer(
        ["hello world"],
        padding="max_length",
        truncation=True,
        max_length=getattr(config, "max_text_length", 128),
        return_tensors="pt",
    )
    input_ids = dummy["input_ids"].to(config.device)
    attention_mask = dummy["attention_mask"].to(config.device)

    class TextWrapper(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, input_ids, attention_mask):
            return self.enc(input_ids=input_ids, attention_mask=attention_mask)

    wrapped = TextWrapper(text_encoder).to(config.device)
    _ensure_outdir(outdir)
    out_path = str(Path(outdir) / filename)

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (input_ids, attention_mask),
            out_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["text_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "text_embedding": {0: "batch"},
            },
        )
    print(f"✅ Exported text encoder ONNX → {out_path}")


def export_video_tower_onnx(
    model, config, outdir="models", filename="video_encoder.onnx"
):
    """
    Exports VideoEncoder as ONNX:
      input:  video[float32][B, T, 3, H, W]
      output: embedding[float32][B, D]  (L2-normalized by your encoder)
    """
    model.eval()
    video_encoder = model.video_encoder.eval().to(config.device)

    T = getattr(config, "max_frames", 8)
    H, W = getattr(config, "frame_size", (224, 224))
    C = 3

    # Dummy batch (B=1). If your runtime varies T/H/W, dynamic axes below cover it.
    dummy_video = torch.randn(1, T, C, H, W, device=config.device, dtype=torch.float32)

    class VideoWrapper(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, video):
            # Your encoder expects (B, T, 3, H, W); no permute needed.
            return self.enc(video)

    wrapped = VideoWrapper(video_encoder).to(config.device)
    _ensure_outdir(outdir)
    out_path = str(Path(outdir) / filename)

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy_video,),
            out_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["video"],
            output_names=["embedding"],
            dynamic_axes={
                "video": {0: "batch", 1: "frames", 3: "height", 4: "width"},
                "embedding": {0: "batch"},
            },
        )
    print(f"✅ Exported video encoder ONNX → {out_path}")


def run_tokenizer_export(tokenizer, outdir: str = "models"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # This writes: tokenizer.json, tokenizer_config.json, special_tokens_map.json, vocab.txt, etc.
    tokenizer.save_pretrained(outdir)
    print(f"✅ Tokenizer saved to {outdir} (includes tokenizer.json)")


def main(
    use_msrvtt: bool = False,
    max_videos: int = 100,
    skip_training: bool = False,
    export_onnx_text: bool = False,
    export_onnx_video: bool = False,
    onnx_outdir: str = "models",
    export_tokenizer: bool = False,
    tokenizer_outdir: str = "models",
    prepare_only: bool = False,
    cached_only: bool = False,
    download_workers: int = 3,
    retry_failures: bool = False,
    data_dir: str = "./data/msrvtt",
    prepared_manifest: str | None = None,
    seed: int = 42,
    device: str = "auto",
    epochs: int = 10,
    batch_size: int = 16,
    checkpoint_path: str | None = None,
    resume: str | None = None,
):
    """Main function for training video search pipeline

    Args:
        use_msrvtt: Whether to use MSR-VTT dataset (True) or synthetic data (False)
        max_videos: Maximum number of videos to process from MSR-VTT
        skip_training: Whether to skip training and just build index from existing model
    """

    # Initialize configuration
    config = VideoSearchConfig(device=device)
    if epochs < 1 or batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if resume and (skip_training or prepare_only):
        raise ValueError("--resume requires training")
    config.num_epochs = epochs
    config.batch_size = batch_size
    checkpoint_path = checkpoint_path or resume or "checkpoints/last.pt"
    if not skip_training and not prepare_only:
        if resume and not Path(resume).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume}")
        if not resume and Path(checkpoint_path).exists():
            raise FileExistsError(
                f"Checkpoint already exists: {checkpoint_path}; use --resume or a new --checkpoint"
            )
    print(f"Configuration: {vars(config)}")
    print()

    if (prepare_only or cached_only) and not use_msrvtt:
        raise ValueError("--prepare-only and --cached-only require --dataset msrvtt")
    if prepare_only:
        prepare_msrvtt_dataset(
            data_dir,
            max_videos,
            download_workers=download_workers,
            retry_failures=retry_failures,
            manifest_path=prepared_manifest,
        )
        print("Preparation complete; use --cached-only to train without downloads.")
        return

    # Create dataset
    if use_msrvtt:
        triplets = load_msrvtt_dataset(
            data_dir=data_dir,
            max_videos=max_videos,
            num_negatives=config.num_negatives,
            download_workers=download_workers,
            retry_failures=retry_failures,
            cached_only=cached_only,
            manifest_path=prepared_manifest,
            seed=seed,
        )
        dataset_name = "MSR-VTT"
    else:
        print("Creating synthetic dataset...")
        triplets = create_synthetic_dataset(
            num_videos=20, num_negatives=config.num_negatives
        )
        dataset_name = "synthetic"

    print(f"Created {len(triplets)} training triplets from {dataset_name} dataset")
    print("\nExample triplet:")
    example = triplets[0]
    print(f"Video: {os.path.basename(example.video_path)}")
    print(f"Positive: {example.positive_query}")
    print(f"Negatives: {example.negative_queries[:3]}...")
    print()

    # Create dataset and dataloader
    video_processor = VideoProcessor(
        max_frames=config.max_frames, frame_size=config.frame_size
    )

    # Initialize tokenizer (we'll get it from the model's text encoder)
    temp_text_encoder = TextEncoder(
        model_name=config.text_model_name, embedding_dim=config.text_embed_dim
    )
    tokenizer = temp_text_encoder.tokenizer

    dataset = VideoQueryDataset(
        triplets=triplets,
        video_processor=video_processor,
        tokenizer=tokenizer,
        max_text_length=config.max_text_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with videos
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"DataLoader batch size: {config.batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    print()

    # Initialize model
    model = VideoTextContrastiveModel(
        video_embed_dim=config.video_embed_dim,
        text_embed_dim=config.text_embed_dim,
        text_model_name=config.text_model_name,
        temperature=config.temperature,
    )

    print("VideoTextContrastiveModel initialized")
    print(
        f"Video encoder params: {sum(p.numel() for p in model.video_encoder.parameters()):,}"
    )
    print(
        f"Text encoder params: {sum(p.numel() for p in model.text_encoder.parameters()):,}"
    )
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Handle skip training or normal training flow
    if skip_training:
        # Load existing trained model
        if not os.path.exists(config.model_save_path):
            print(f"❌ No trained model found at {config.model_save_path}")
            print("💡 Train a model first or remove --skip-training flag")
            return

        print("Loading existing trained model...")
        import torch

        checkpoint = torch.load(
            config.model_save_path, map_location=config.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(config.device)
        model.eval()
        print("✅ Model loaded successfully")
    else:
        # Train the model
        print("Starting training...")
        signature = hashlib.sha256()
        for triplet in triplets:
            path = Path(triplet.video_path)
            stat = path.stat()
            signature.update(
                json.dumps(
                    [
                        path.name,
                        stat.st_size,
                        stat.st_mtime_ns,
                        triplet.positive_query,
                        triplet.negative_queries,
                    ],
                    ensure_ascii=False,
                ).encode("utf-8")
            )
        train_losses = train_video_search_model(
            model,
            dataloader,
            config,
            checkpoint_path=checkpoint_path,
            resume=resume,
            dataset_signature=signature.hexdigest(),
        )

        print("Training completed!")
        print(f"Final loss: {train_losses[-1]:.4f}")
        print()

        # Save trained model
        print("Saving trained model...")
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)

        # Save model state dict
        import torch

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "dataset_name": dataset_name,
            },
            config.model_save_path,
        )
        print(f"Model saved to {config.model_save_path}")

        # Evaluate the model
        print("Evaluating model...")
        test_triplets = triplets[
            :5
        ]  # Training-sample sanity check, not held-out evaluation

        eval_metrics = evaluate_retrieval(
            model=model, test_triplets=test_triplets, device=config.device, top_k=3
        )

        print("Evaluation Results:")
        for metric, value in eval_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
        print()

    # Get all video paths
    video_paths = list(set(triplet.video_path for triplet in triplets))

    # Save video paths for later reference
    with open(config.video_paths_save_path, "w") as f:
        for video_path in video_paths:
            f.write(f"{video_path}\n")
    print(f"Video paths saved to {config.video_paths_save_path}")

    # Always create and save video index for search
    print("Creating and saving video index...")
    video_index = build_video_index(
        model=model,
        video_paths=video_paths,
        device=config.device,
    )

    # Save video index
    import pickle

    with open(config.index_save_path, "wb") as f:
        pickle.dump(video_index, f)
    print(f"Video index saved to {config.index_save_path}")

    # Test search with the saved index
    print("\nTesting search with saved index...")
    search_query = "cat playing"
    search_results = search_videos_indexed(
        model=model, query=search_query, video_index=video_index, top_k=5
    )

    print(f"Search Results for '{search_query}':")
    for i, (video_path, score) in enumerate(search_results, 1):
        video_name = os.path.basename(video_path)
        print(f"  {i}. {video_name} (score: {score:.3f})")

    # Optionally export to ONNX
    if export_onnx_text:
        export_text_tower_onnx(
            model=model,
            tokenizer=tokenizer,
            config=config,
            outdir=onnx_outdir,
            filename="text_encoder.onnx",
        )

    if export_onnx_video:
        export_video_tower_onnx(
            model=model,
            config=config,
            outdir=onnx_outdir,
            filename="video_encoder.onnx",
        )

    if export_tokenizer:
        run_tokenizer_export(
            tokenizer=tokenizer,
            outdir=tokenizer_outdir,
        )

    print(
        "\nTraining completed! Use search_interactive.py to search with saved model and index."
    )
    print(f"Model: {config.model_save_path}")
    print(f"Index: {config.index_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Search Training Pipeline")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "msrvtt"],
        default="synthetic",
        help="Dataset to use for training (default: synthetic)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=100,
        help="Maximum number of videos to use from MSR-VTT dataset (default: 100)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and build index from existing trained model",
    )
    parser.add_argument(
        "--export-onnx-text",
        action="store_true",
        help="Export the trained/loaded text tower to ONNX",
    )
    parser.add_argument(
        "--export-onnx-video",
        action="store_true",
        help="Export the trained/loaded video tower to ONNX",
    )
    parser.add_argument(
        "--onnx-outdir",
        type=str,
        default="models",
        help="Directory to write ONNX files (default: models)",
    )
    parser.add_argument(
        "--export-tokenizer",
        action="store_true",
        help="Export the tokenizer (writes tokenizer.json next to the model).",
    )
    parser.add_argument(
        "--tokenizer-outdir",
        type=str,
        default="models",
        help="Directory to write tokenizer files (default: models)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare validated MSR-VTT clips and a manifest without loading a model",
    )
    mode.add_argument(
        "--cached-only",
        action="store_true",
        help="Train from the prepared manifest without metadata or video downloads",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=3,
        help="Concurrent downloads (1-8; default: 3)",
    )
    parser.add_argument(
        "--retry-failures", action="store_true", help="Retry cached download failures"
    )
    parser.add_argument(
        "--data-dir", default="./data/msrvtt", help="MSR-VTT cache directory"
    )
    parser.add_argument(
        "--prepared-manifest",
        default=None,
        help="Prepared manifest path (default: DATA_DIR/prepared.json)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Negative-sampling seed")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device; auto selects GPU when available (AMD uses cuda too)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Total epochs, including completed epochs when resuming",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Epoch checkpoint destination (default: checkpoints/last.pt, or the resume path)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="Resume a trusted training checkpoint at the next epoch",
    )
    args = parser.parse_args()

    use_msrvtt = args.dataset == "msrvtt"
    main(
        use_msrvtt=use_msrvtt,
        max_videos=args.max_videos,
        skip_training=args.skip_training,
        export_onnx_text=args.export_onnx_text,
        export_onnx_video=args.export_onnx_video,
        onnx_outdir=args.onnx_outdir,
        export_tokenizer=args.export_tokenizer,
        tokenizer_outdir=args.tokenizer_outdir,
        prepare_only=args.prepare_only,
        cached_only=args.cached_only,
        download_workers=args.download_workers,
        retry_failures=args.retry_failures,
        data_dir=args.data_dir,
        prepared_manifest=args.prepared_manifest,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
    )
