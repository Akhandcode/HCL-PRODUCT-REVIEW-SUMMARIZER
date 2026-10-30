"""
Unit tests for ReviewDataLoader
"""

import unittest
import pandas as pd
from src.data_loader import ReviewDataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = ReviewDataLoader()

    def test_clean_text(self):
        dirty = "  Great headphones! &amp; nice sound... https://example.com/item   "
        cleaned = self.loader.clean_text(dirty)
        self.assertEqual(cleaned, "Great headphones! & nice sound...")

    def test_preprocess_dataframe(self):
        df = pd.DataFrame({
            "review_id": ["1", "2"],
            "review_text": ["Superb active noise cancellation!", "Bad battery life drops immediately."],
            "rating": [5, 1],
            "review_title": ["Awesome", "Terrible"]
        })
        processed = self.loader.preprocess(df)
        self.assertEqual(len(processed), 2)
        self.assertIn("full_text", processed.columns)
        self.assertIn("word_count", processed.columns)
        self.assertEqual(processed["word_count"].iloc[0], 5)  # "Awesome. Superb active noise cancellation!"

    def test_get_dataset_stats(self):
        df = pd.DataFrame({
            "review_id": ["1", "2"],
            "review_text": ["Superb active noise cancellation!", "Bad battery life drops immediately."],
            "rating": [5, 1]
        })
        processed = self.loader.preprocess(df)
        stats = self.loader.get_dataset_stats(processed)
        self.assertEqual(stats["total_reviews"], 2)
        self.assertEqual(stats["avg_rating"], 3.0)


if __name__ == "__main__":
    unittest.main()
