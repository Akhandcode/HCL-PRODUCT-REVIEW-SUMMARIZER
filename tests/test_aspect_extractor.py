"""
Unit tests for AspectExtractor
"""

import unittest
import pandas as pd
from src.aspect_extractor import AspectExtractor


class TestAspectExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = AspectExtractor()

    def test_sentence_split(self):
        text = "Sound quality is amazing. Battery life is okay! However, the app crashes."
        sentences = self.extractor.split_into_sentences(text)
        self.assertEqual(len(sentences), 3)

    def test_aspect_signals(self):
        df = pd.DataFrame({
            "full_text": [
                "The active noise cancellation is world-class.",
                "Battery life is insane with over 38 hours.",
                "Microphone quality for phone calls is disappointing."
            ],
            "sentiment_label": ["POSITIVE", "POSITIVE", "NEGATIVE"]
        })
        signals = self.extractor.extract_aspect_signals(df)
        self.assertIn("Active Noise Cancellation", signals)
        self.assertIn("Battery & Charging", signals)
        self.assertIn("Microphone & Calls", signals)
        self.assertEqual(signals["Active Noise Cancellation"]["positive_pct"], 100.0)
        self.assertEqual(signals["Microphone & Calls"]["negative_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
