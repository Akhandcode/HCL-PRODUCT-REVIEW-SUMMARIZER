"""
Aspect-Based Feature Extraction Module
Extracts recurring product aspects (Battery, Sound, ANC, Comfort, App, Mic, Build, etc.),
calculates sentiment breakdown per aspect, and extracts representative review quotes.
"""

import re
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import pandas as pd


ASPECT_TAXONOMY: Dict[str, List[str]] = {
    "Active Noise Cancellation": ["anc", "noise cancellation", "noise cancelling", "cancels", "transparency", "airplane"],
    "Sound & Audio Quality": ["sound", "audio", "bass", "treble", "mids", "highs", "clarity", "audiophile", "eq", "music"],
    "Battery & Charging": ["battery", "charge", "charging", "hours", "playback", "usb-c", "fast charge", "lifesaver"],
    "Comfort & Fit": ["comfort", "comfortable", "earcups", "ear pads", "memory foam", "headband", "fatigue", "heavy", "sweaty"],
    "Connectivity & Latency": ["bluetooth", "pairing", "multi-point", "connect", "connection", "latency", "stutter", "sync", "drops"],
    "Microphone & Calls": ["mic", "microphone", "call", "calls", "voice", "muffled", "phone calls", "zoom"],
    "Build Quality & Design": ["build", "durability", "plastic", "hinge", "aluminum", "leatherette", "squeak", "case", "pouch", "materials"],
    "App & Controls": ["app", "companion app", "buttons", "controls", "swipe", "firmware", "sensor", "auto-pause", "login", "bugs"],
    "Value & Pricing": ["price", "retail", "cost", "expensive", "value", "worth", "purchase", "replacement"]
}


class AspectExtractor:
    """Categorizes review sentences into aspects and aggregates sentiment per aspect."""

    def __init__(self, aspect_taxonomy: Dict[str, List[str]] = None):
        self.taxonomy = aspect_taxonomy or ASPECT_TAXONOMY

    def split_into_sentences(self, text: str) -> List[str]:
        """Splits review text into discrete sentences."""
        if not isinstance(text, str):
            return []
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw_sentences if len(s.strip()) > 8]

    def extract_aspect_signals(self, df: pd.DataFrame, text_column: str = "full_text") -> Dict[str, Any]:
        """
        Scans all reviews in the DataFrame, mapping sentences to aspects and tracking
        pros (positive sentences) vs. cons (negative sentences).
        """
        aspect_data = defaultdict(lambda: {
            "mention_count": 0,
            "positive_mentions": 0,
            "negative_mentions": 0,
            "neutral_mentions": 0,
            "positive_quotes": [],
            "negative_quotes": []
        })

        for _, row in df.iterrows():
            review_sentiment = row.get("sentiment_label", "NEUTRAL")
            text = row.get(text_column, "")
            sentences = self.split_into_sentences(text)

            for sentence in sentences:
                s_lower = sentence.lower()
                matched_aspects = []

                for aspect, keywords in self.taxonomy.items():
                    if any(re.search(rf"\b{re.escape(kw)}\b", s_lower) for kw in keywords):
                        matched_aspects.append(aspect)

                for aspect in matched_aspects:
                    data = aspect_data[aspect]
                    data["mention_count"] += 1

                    # Check sentence-level polarity cues
                    neg_cues = ["however", "downside", "disappoint", "squeak", "bugs", "drops", "heavy", "stutter",
                                "muffled", "loses", "expensive", "not", "delay", "sweaty", "annoying", "poor", "cheap"]
                    pos_cues = ["great", "world-class", "insane", "crystal clear", "super plush", "flawless",
                                "superb", "best", "excellent", "comfortable", "exceeded", "cancels", "plush"]

                    is_neg_sentence = any(cue in s_lower for cue in neg_cues)
                    is_pos_sentence = any(cue in s_lower for cue in pos_cues)

                    if is_neg_sentence and not is_pos_sentence:
                        sent_type = "NEGATIVE"
                    elif is_pos_sentence and not is_neg_sentence:
                        sent_type = "POSITIVE"
                    else:
                        sent_type = review_sentiment

                    if sent_type == "POSITIVE":
                        data["positive_mentions"] += 1
                        if len(data["positive_quotes"]) < 8 and len(sentence) > 20:
                            data["positive_quotes"].append(sentence)
                    elif sent_type == "NEGATIVE":
                        data["negative_mentions"] += 1
                        if len(data["negative_quotes"]) < 8 and len(sentence) > 20:
                            data["negative_quotes"].append(sentence)
                    else:
                        data["neutral_mentions"] += 1

        # Calculate final aspect scores and rankings
        aspect_summary = {}
        for aspect, data in aspect_data.items():
            total = data["mention_count"]
            if total == 0:
                continue

            pos_pct = round((data["positive_mentions"] / total) * 100, 1)
            neg_pct = round((data["negative_mentions"] / total) * 100, 1)
            
            # Net Aspect Sentiment: range [-100, 100]
            net_score = round(pos_pct - neg_pct, 1)

            aspect_summary[aspect] = {
                "mention_count": total,
                "positive_pct": pos_pct,
                "negative_pct": neg_pct,
                "net_sentiment": net_score,
                "status": "Strong Pro" if net_score >= 35 else ("Strong Con" if net_score <= -20 else "Mixed/Neutral"),
                "sample_positive_quotes": data["positive_quotes"][:3],
                "sample_negative_quotes": data["negative_quotes"][:3]
            }

        # Sort aspects by total mentions
        sorted_aspects = dict(sorted(aspect_summary.items(), key=lambda item: item[1]["mention_count"], reverse=True))
        return sorted_aspects
