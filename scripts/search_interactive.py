#!/usr/bin/env python3
"""
Interactive Video Search Interface

Load a trained model and video index to perform interactive video search queries.
"""

import argparse
import os
import pickle
import sys

import torch

# Ensure the project root (containing the `video_search` package) is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from video_search.core.config import VideoSearchConfig
from video_search.core.model import VideoTextContrastiveModel
from video_search.search.index import VideoIndex
from video_search.search.search import search_videos_indexed


def load_model_and_index(
    config: VideoSearchConfig,
) -> tuple[VideoTextContrastiveModel, VideoIndex, list[str]]:
    """Load trained model, video index, and video paths from saved files"""

    # Check if files exist
    if not os.path.exists(config.model_save_path):
        raise FileNotFoundError(f"Model file not found: {config.model_save_path}")
    if not os.path.exists(config.index_save_path):
        raise FileNotFoundError(f"Index file not found: {config.index_save_path}")
    if not os.path.exists(config.video_paths_save_path):
        raise FileNotFoundError(
            f"Video paths file not found: {config.video_paths_save_path}"
        )

    print("Loading trained model...")
    checkpoint = torch.load(
        config.model_save_path, map_location=config.device, weights_only=False
    )

    # Initialize model with same configuration
    model = VideoTextContrastiveModel(
        video_embed_dim=config.video_embed_dim,
        text_embed_dim=config.text_embed_dim,
        text_model_name=config.text_model_name,
        temperature=config.temperature,
    )

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    dataset_name = checkpoint.get("dataset_name", "unknown")
    print(f"Model loaded successfully (trained on {dataset_name} dataset)")

    # Load video index
    print("Loading video index...")
    with open(config.index_save_path, "rb") as f:
        video_index = pickle.load(f)
    print(f"Video index loaded with {len(video_index)} videos")

    # Load video paths
    print("Loading video paths...")
    with open(config.video_paths_save_path, "r") as f:
        video_paths = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(video_paths)} video paths")

    return model, video_index, video_paths


def interactive_search(
    model: VideoTextContrastiveModel,
    video_index: VideoIndex,
    video_paths: list[str],
    top_k: int = 5,
):
    """Run interactive search loop"""

    print("\n" + "=" * 60)
    print("🎥 INTERACTIVE VIDEO SEARCH")
    print("=" * 60)
    print(f"📊 Index contains {len(video_paths)} videos")
    print("💡 Type your search query and press Enter")
    print("💡 Type 'quit' or 'exit' to stop")
    print("💡 Type 'help' for more options")
    print("=" * 60 + "\n")

    while True:
        try:
            # Get user input
            query = input("🔍 Search query: ").strip()

            # Handle special commands
            if query.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break
            elif query.lower() == "help":
                print("\n📖 Available commands:")
                print("  • Enter any text to search for videos")
                print("  • 'quit' or 'exit' - Exit the program")
                print("  • 'help' - Show this help message")
                print("  • 'stats' - Show index statistics")
                print(f"  • Returns top {top_k} results by default\n")
                continue
            elif query.lower() == "stats":
                print("\n📈 Index Statistics:")
                print("  • Total videos: {len(video_paths)}")
                print(
                    "  • Index type: Pre-computed embedding cache with dot product similarity"
                )
                print(f"  • Model device: {next(model.parameters()).device}")
                print(
                    f"  • Embedding dimension: {model.video_encoder.projection.out_features}"
                )
                print()
                continue
            elif not query:
                print("❌ Please enter a search query")
                continue

            # Perform search
            print(f"🔎 Searching for: '{query}'...")
            search_results = search_videos_indexed(
                model=model, query=query, video_index=video_index, top_k=top_k
            )

            # Display results
            if not search_results:
                print("😞 No results found")
            else:
                print(f"\n🎯 Top {len(search_results)} results:")
                print("-" * 50)
                for i, (video_path, score) in enumerate(search_results, 1):
                    video_name = os.path.basename(video_path)
                    # Create progress bar for score visualization
                    bar_length = 20
                    filled_length = int(bar_length * min(score, 1.0))
                    bar = "█" * filled_length + "▒" * (bar_length - filled_length)

                    print(f"{i:2d}. {video_name}")
                    print(f"    Score: {score:.3f} [{bar}] {score * 100:.1f}%")
                    print(f"    Path:  {video_path}")
                    print()

        except KeyboardInterrupt:
            print("\n\n👋 Search interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during search: {e}")
            print("Please try a different query.\n")


def main():
    """Main function for interactive search"""
    parser = argparse.ArgumentParser(description="Interactive Video Search")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Custom path to model file (overrides config default)",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        help="Custom path to index file (overrides config default)",
    )

    args = parser.parse_args()

    # Initialize configuration
    config = VideoSearchConfig()

    # Override paths if provided
    if args.model_path:
        config.model_save_path = args.model_path
    if args.index_path:
        config.index_save_path = args.index_path

    try:
        # Load model and index
        model, video_index, video_paths = load_model_and_index(config)

        # Start interactive search
        interactive_search(model, video_index, video_paths, top_k=args.top_k)

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Make sure you've run the training pipeline first:")
        print("   python pipeline.py --dataset msrvtt --max-videos 100")
        print("   python pipeline.py --dataset synthetic")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
