"""Behavior checks for leakage prevention and actual caption-to-video ranking."""

import unittest

import torch
from torch import nn

from video_search.core.model import evaluate_text_to_video
from video_search.utils.data import (
    VideoQueryTriplet,
    build_msrvtt_triplets,
    split_video_records,
)


class HeldoutBehaviorTests(unittest.TestCase):
    def test_split_keeps_sources_and_all_captions_out_of_training(self):
        records = [
            {
                "video_id": f"video{i}",
                "url": f"source-{i // 2}",
                "captions": [f"caption-{i}-a", f"caption-{i}-b"],
            }
            for i in range(20)
        ]
        train, test = split_video_records(records, 0.2, seed=42)
        self.assertEqual(len(train) + len(test), len(records))
        self.assertTrue(train and test)
        self.assertTrue({r["url"] for r in train}.isdisjoint(r["url"] for r in test))
        _, reordered_test = split_video_records(list(reversed(records)), 0.2, seed=42)
        self.assertEqual(
            {r["video_id"] for r in test}, {r["video_id"] for r in reordered_test}
        )
        triplets = build_msrvtt_triplets(train, "unused-cache", num_negatives=3)
        train_captions = {caption for r in train for caption in r["captions"]}
        test_captions = {caption for r in test for caption in r["captions"]}
        self.assertEqual({t.positive_query for t in triplets}, train_captions)
        for triplet in triplets:
            self.assertTrue(set(triplet.negative_queries) <= train_captions)
            self.assertTrue(test_captions.isdisjoint(triplet.get_all_queries()))

    def test_recall_ranks_all_videos_for_every_caption(self):
        paths = [f"video{i}" for i in range(12)]
        vectors = torch.eye(12)
        queries = {}
        triplets = []
        for index, path in enumerate(paths):
            # The correct candidate has rank index + 1, with no tied scores.
            ordering = [i for i in range(12) if i != index]
            ordering.insert(index, index)
            query = torch.zeros(12)
            query[ordering] = torch.arange(12, 0, -1).float()
            queries[path] = query
            triplets.append(VideoQueryTriplet(path, path, ["not-an-evaluation-query"]))
        # Multiple captions for the same clip must not duplicate gallery videos.
        triplets.append(VideoQueryTriplet(paths[0], paths[0], []))

        class LookupModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("vectors", vectors)

            def encode_video(self, path):
                return self.vectors[paths.index(path)]

            def encode_texts(self, captions):
                return torch.stack([queries[caption] for caption in captions])

        model = LookupModel()
        metrics = evaluate_text_to_video(model, triplets, batch_size=4)
        self.assertAlmostEqual(metrics["recall_at_1"], 2 / 13)
        self.assertAlmostEqual(metrics["recall_at_5"], 6 / 13)
        self.assertAlmostEqual(metrics["recall_at_10"], 11 / 13)
        self.assertEqual(metrics["candidate_videos"], 12)
        self.assertEqual(metrics["total_queries"], 13)
        self.assertEqual(metrics, evaluate_text_to_video(model, triplets, batch_size=1))


if __name__ == "__main__":
    unittest.main()
