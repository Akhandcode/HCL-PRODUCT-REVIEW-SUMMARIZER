"""
Transformer Sentiment Analysis Module
Uses HuggingFace DistilBERT model to classify review sentiment in high-throughput batches,
compute sentiment distributions, and segment reviews for Map-Reduce summarization.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Batch-oriented Sentiment Analyzer powered by HuggingFace Transformers."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        batch_size: int = 128,
        max_length: int = 128,
        neutral_margin: float = 0.15
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.neutral_margin = neutral_margin
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """Loads and optimizes tokenizer and model (quantized for CPU)."""
        if self._model is None:
            logger.info(f"Loading sentiment model '{self.model_name}' on {self.device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            raw_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            if self.device.type == "cpu":
                try:
                    # Dynamic INT8 Quantization: ~3-4x speedup on CPU with zero accuracy drop
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self._model = torch.quantization.quantize_dynamic(
                            raw_model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                    logger.info("Applied PyTorch dynamic INT8 quantization for fast CPU execution.")
                except Exception as e:
                    logger.warning(f"Could not quantize model: {e}. Using standard model.")
                    self._model = raw_model
            else:
                self._model = raw_model.to(self.device)
            
            self._model.eval()

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Processes a list of review texts using fast vectorized batch inference."""
        if not texts:
            return []

        self._load_model()
        clean_inputs = [t.strip() if isinstance(t, str) and t.strip() else "Neutral" for t in texts]

        analyzed: List[Dict[str, Any]] = []

        with torch.inference_mode():
            for i in range(0, len(clean_inputs), self.batch_size):
                batch_texts = clean_inputs[i:i + self.batch_size]
                inputs = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                if self.device.type != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

                for p in probs:
                    neg_score = float(p[0])
                    pos_score = float(p[1])
                    
                    # SST-2: index 0 = NEGATIVE, index 1 = POSITIVE
                    if pos_score >= neg_score:
                        raw_label = "POSITIVE"
                        confidence = pos_score
                        polarity = pos_score
                    else:
                        raw_label = "NEGATIVE"
                        confidence = neg_score
                        polarity = -neg_score

                    # Calibrate label with neutral band
                    if abs(polarity) < self.neutral_margin:
                        assigned_label = "NEUTRAL"
                    elif polarity > 0:
                        assigned_label = "POSITIVE"
                    else:
                        assigned_label = "NEGATIVE"

                    analyzed.append({
                        "sentiment_label": assigned_label,
                        "confidence": round(confidence, 4),
                        "polarity_score": round(polarity, 4)
                    })

        return analyzed

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = "full_text") -> pd.DataFrame:
        """Analyzes all reviews in a DataFrame and appends sentiment columns."""
        df = df.copy()
        if text_column not in df.columns:
            text_column = "cleaned_text" if "cleaned_text" in df.columns else df.columns[0]

        texts = df[text_column].tolist()
        logger.info(f"Running sentiment analysis on {len(texts)} reviews (batch_size={self.batch_size})...")

        results = self.analyze_batch(texts)
        
        df["sentiment_label"] = [r["sentiment_label"] for r in results]
        df["sentiment_confidence"] = [r["confidence"] for r in results]
        df["polarity_score"] = [r["polarity_score"] for r in results]

        # If star rating is present, blend for nuanced neutral detection
        if "rating" in df.columns:
            # 3-star reviews with weak polarity are treated as Neutral
            rating_neutral_mask = (df["rating"] == 3) & (df["polarity_score"].abs() < 0.85)
            df.loc[rating_neutral_mask, "sentiment_label"] = "NEUTRAL"

        return df

    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates macro-level sentiment analytics."""
        total = len(df)
        if total == 0:
            return {"total": 0}

        counts = df["sentiment_label"].value_counts().to_dict()
        pos_count = counts.get("POSITIVE", 0)
        neg_count = counts.get("NEGATIVE", 0)
        neu_count = counts.get("NEUTRAL", 0)

        pos_pct = round((pos_count / total) * 100, 1)
        neg_pct = round((neg_count / total) * 100, 1)
        neu_pct = round((neu_count / total) * 100, 1)

        # Net Sentiment Score: Pos% - Neg% (-100 to +100)
        nss = round(pos_pct - neg_pct, 1)
        avg_polarity = round(float(df["polarity_score"].mean()), 3)

        return {
            "total_reviews": total,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "neutral_pct": neu_pct,
            "net_sentiment_score": nss,
            "avg_polarity": avg_polarity,
            "overall_sentiment": "Positive" if nss > 15 else ("Negative" if nss < -15 else "Mixed/Neutral")
        }

    @staticmethod
    def partition_by_sentiment(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Partitions dataframe into (positive_df, negative_df, neutral_df)."""
        pos_df = df[df["sentiment_label"] == "POSITIVE"].copy()
        neg_df = df[df["sentiment_label"] == "NEGATIVE"].copy()
        neu_df = df[df["sentiment_label"] == "NEUTRAL"].copy()
        return pos_df, neg_df, neu_df
