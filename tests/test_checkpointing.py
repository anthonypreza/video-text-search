"""Resume real optimizer updates on a tiny CPU model, without pretrained downloads."""

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from video_search.core.config import VideoSearchConfig
from video_search.core.model import train_video_search_model


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_encoder = nn.Module()
        self.video_encoder.projection = nn.Linear(2, 2)
        self.text_encoder = nn.Module()
        self.text_encoder.projection = nn.Linear(2, 2)

    def forward(self, frames, ids, mask, targets):
        values = nn.functional.dropout(frames, p=0.2, training=self.training)
        logits = self.video_encoder.projection(values) + self.text_encoder.projection(
            ids.float()
        )
        return nn.functional.cross_entropy(logits, targets), logits


def loader():
    samples = [
        {
            "video_frames": torch.tensor([i / 10, 1.0]),
            "text_input_ids": torch.tensor([i % 2, 1]),
            "text_attention_mask": torch.ones(2),
            "target_index": torch.tensor(i % 2),
        }
        for i in range(6)
    ]
    return DataLoader(samples, batch_size=2, shuffle=True)


def model():
    torch.manual_seed(17)
    random.seed(17)
    return TinyModel()


class CheckpointBehaviorTests(unittest.TestCase):
    def test_interrupted_run_resumes_identical_optimizer_updates(self):
        config = VideoSearchConfig(device="cpu")
        config.num_epochs = 3
        config.batch_size = 2
        baseline = model()
        expected_losses = train_video_search_model(baseline, loader(), config)

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "last.pt")
            interrupted = model()
            batches = loader()

            class InterruptedLoader:
                iterations = 0

                def __len__(self):
                    return len(batches)

                def __iter__(self):
                    self.iterations += 1
                    if self.iterations == 2:
                        raise RuntimeError("simulated interruption")
                    return iter(batches)

            with self.assertRaises(RuntimeError):
                train_video_search_model(
                    interrupted,
                    InterruptedLoader(),
                    config,
                    checkpoint_path=path,
                    dataset_signature="fixture",
                )
            resumed = model()
            actual_losses = train_video_search_model(
                resumed, loader(), config, resume=path, dataset_signature="fixture"
            )
            self.assertEqual(expected_losses, actual_losses)
            for expected, actual in zip(baseline.parameters(), resumed.parameters()):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                train_video_search_model(
                    model(),
                    loader(),
                    config,
                    resume=path,
                    dataset_signature="different data",
                )

    def test_failed_save_preserves_previous_checkpoint(self):
        config = VideoSearchConfig(device="cpu")
        config.num_epochs = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            train_video_search_model(
                model(), loader(), config, checkpoint_path=str(path)
            )
            original = path.read_bytes()
            config.num_epochs = 2
            with (
                patch(
                    "video_search.core.model.torch.save",
                    side_effect=OSError("disk write failed"),
                ),
                self.assertRaises(OSError),
            ):
                train_video_search_model(model(), loader(), config, resume=str(path))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
