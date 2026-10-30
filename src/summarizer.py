"""
Review Summarization Module
Synthesizes 1,000+ reviews into structured Pros, Cons, Sentiment Analytics,
and an Executive Recommendation using Map-Reduce Transformer logic.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from src.aspect_extractor import AspectExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewSummarizer:
    """Summarization engine extracting structured Pros, Cons, and Executive Verdict."""

    def __init__(
        self,
        use_transformer_seq2seq: bool = True,
        seq2seq_model_name: str = "t5-small"
    ):
        self.use_transformer_seq2seq = use_transformer_seq2seq
        self.seq2seq_model_name = seq2seq_model_name
        self._seq2seq_pipeline = None

    @property
    def seq2seq_pipeline(self):
        """Lazy loader for transformer seq2seq summarization pipeline."""
        if self.use_transformer_seq2seq and self._seq2seq_pipeline is None:
            try:
                from transformers import pipeline
                logger.info(f"Loading Seq2Seq summarization model '{self.seq2seq_model_name}'...")
                self._seq2seq_pipeline = pipeline(
                    "summarization",
                    model=self.seq2seq_model_name,
                    device=-1
                )
            except Exception as e:
                logger.warning(f"Could not load Seq2Seq model '{self.seq2seq_model_name}': {e}. Using extractive synthesis.")
                self.use_transformer_seq2seq = False
        return self._seq2seq_pipeline

    def _generate_abstractive_summary(self, context_text: str, max_length: int = 100, min_length: int = 30) -> str:
        """Generates an abstractive summary paragraph using Seq2Seq Transformer."""
        if self.use_transformer_seq2seq and self.seq2seq_pipeline:
            try:
                # Truncate context to ~1500 chars to avoid memory overload on CPU
                truncated = context_text[:1500]
                res = self.seq2seq_pipeline(truncated, max_length=max_length, min_length=min_length, do_sample=False)
                if res and "summary_text" in res[0]:
                    return res[0]["summary_text"].strip()
            except Exception as e:
                logger.warning(f"Seq2Seq inference failed: {e}. Falling back to rule-based synthesis.")
        
        # Rule-based fallback synthesis
        return context_text[:280] + ("..." if len(context_text) > 280 else "")

    def summarize(
        self,
        df: pd.DataFrame,
        sentiment_summary: Dict[str, Any],
        aspect_signals: Dict[str, Any],
        product_name: str = "Product"
    ) -> Dict[str, Any]:
        """
        Combines macro sentiment, aspect clustering, and review quotes to produce
        the final Pros, Cons, Sentiment Breakdown, and Executive Verdict.
        """
        total_reviews = len(df)
        avg_rating = sentiment_summary.get("avg_rating") or (round(float(df["rating"].mean()), 2) if "rating" in df.columns else None)

        # 1. Identify Pros (Aspects with high positive percentage & positive volume)
        pros = []
        for aspect_name, data in aspect_signals.items():
            if data["positive_pct"] >= 50.0 and data["positive_pct"] > data["negative_pct"]:
                quotes = data["sample_positive_quotes"]
                headline = quotes[0] if quotes else f"Strong performance in {aspect_name}."
                pros.append({
                    "aspect": aspect_name,
                    "headline": headline,
                    "sentiment_score": f"{data['positive_pct']}% Positive",
                    "mentions": data["mention_count"],
                    "confidence": "High" if data["mention_count"] > 100 else "Moderate",
                    "sample_quotes": quotes[:2]
                })

        # Sort pros by positive mentions
        pros = sorted(pros, key=lambda x: x["mentions"], reverse=True)

        # 2. Identify Cons (Aspects with notable negative percentage)
        cons = []
        for aspect_name, data in aspect_signals.items():
            if data["negative_pct"] >= 35.0:
                quotes = data["sample_negative_quotes"]
                headline = quotes[0] if quotes else f"Recurrent user issues with {aspect_name}."
                cons.append({
                    "aspect": aspect_name,
                    "headline": headline,
                    "dissatisfaction_rate": f"{data['negative_pct']}% Negative",
                    "mentions": data["mention_count"],
                    "severity": "Critical" if data["negative_pct"] > 60 else "Moderate",
                    "sample_quotes": quotes[:2]
                })

        # Sort cons by severity & mentions
        cons = sorted(cons, key=lambda x: x["mentions"], reverse=True)

        # 3. Compute Product Health Score (0.0 to 10.0 scale)
        pos_ratio = (sentiment_summary.get("positive_pct", 50.0) / 100.0)
        neg_ratio = (sentiment_summary.get("negative_pct", 25.0) / 100.0)
        
        # Base 10 score
        if avg_rating:
            score = (avg_rating * 1.2) + (pos_ratio * 4.0) - (neg_ratio * 2.0)
        else:
            score = (pos_ratio * 7.5) + 2.5 - (neg_ratio * 2.0)
        health_score = max(1.0, min(10.0, round(score, 1)))

        # 4. Executive Verdict & Recommendation
        if health_score >= 8.0:
            verdict_badge = "Highly Recommended"
            verdict_desc = f"{product_name} demonstrates exceptional performance and customer satisfaction across core features."
        elif health_score >= 6.5:
            verdict_badge = "Recommended with Reservations"
            verdict_desc = f"{product_name} delivers strong foundational quality but suffers from specific recurring flaws that buyers should evaluate."
        else:
            verdict_badge = "Proceed with Caution"
            verdict_desc = f"{product_name} received high dissatisfaction rates regarding key usability and hardware components."

        # Compile context for executive abstractive summary
        top_pro_text = ", ".join([p["aspect"] for p in pros[:3]])
        top_con_text = ", ".join([c["aspect"] for c in cons[:3]])
        synthesis_context = (
            f"Analysis of {total_reviews} customer reviews for {product_name}. "
            f"The overall customer sentiment is {sentiment_summary.get('overall_sentiment', 'Positive')} "
            f"with {sentiment_summary.get('positive_pct', 0)}% positive feedback and an average rating of {avg_rating or 'N/A'}/5.0. "
            f"Key strengths highlighted by buyers include {top_pro_text}. "
            f"However, primary customer complaints center on {top_con_text}."
        )

        executive_summary = self._generate_abstractive_summary(synthesis_context, max_length=90, min_length=35)

        # 5. Target Audience Recommendations
        ideal_for = [
            f"Users prioritizing {pros[0]['aspect']}" if len(pros) > 0 else "General consumers",
            f"Buyers seeking long battery endurance and travel noise cancellation" if any("Battery" in p["aspect"] or "Noise" in p["aspect"] for p in pros) else "Daily casual users",
            f"Frequent commuters who need reliable physical button controls" if any("Controls" in p["aspect"] for p in pros) else "Commuters"
        ]

        not_ideal_for = [
            f"Users who rely heavily on {cons[0]['aspect']}" if len(cons) > 0 else "None",
            f"Users needing pristine microphone call quality in noisy/windy environments" if any("Microphone" in c["aspect"] for c in cons) else "Professional call agents",
            f"Competitive gamers needing ultra-low latency bluetooth codecs" if any("Latency" in c["aspect"] for c in cons) else "Gaming setups"
        ]

        return {
            "product_name": product_name,
            "total_reviews_analyzed": total_reviews,
            "product_score": health_score,
            "verdict": verdict_badge,
            "verdict_description": verdict_desc,
            "executive_summary": executive_summary,
            "sentiment_overview": sentiment_summary,
            "pros": pros,
            "cons": cons,
            "aspect_breakdown": aspect_signals,
            "target_audience": {
                "who_should_buy": ideal_for,
                "who_should_avoid": not_ideal_for
            }
        }
