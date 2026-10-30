"""
Review Summarization Pipeline Orchestrator
Connects DataLoader, SentimentAnalyzer, AspectExtractor, and ReviewSummarizer
into a single high-performance pipeline.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional
import pandas as pd

from src.data_loader import ReviewDataLoader
from src.sentiment_analyzer import SentimentAnalyzer
from src.aspect_extractor import AspectExtractor
from src.summarizer import ReviewSummarizer
from src.scraper import ReviewScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewSummarizationPipeline:
    """Master pipeline for analyzing and summarizing product review datasets."""

    def __init__(
        self,
        batch_size: int = 64,
        use_transformer_seq2seq: bool = False, # Set to False for lightning-fast CPU runs, can be toggled
        seq2seq_model_name: str = "t5-small"
    ):
        self.data_loader = ReviewDataLoader()
        self.sentiment_analyzer = SentimentAnalyzer(batch_size=batch_size)
        self.aspect_extractor = AspectExtractor()
        self.summarizer = ReviewSummarizer(
            use_transformer_seq2seq=use_transformer_seq2seq,
            seq2seq_model_name=seq2seq_model_name
        )
        self.scraper = ReviewScraper()

    def run_from_url(
        self,
        url: str,
        max_reviews: int = 100,
        cookie: Optional[str] = None,
        target_reviews: Optional[int] = 1000,
        export_json: Optional[str] = None,
        save_csv_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Scrapes live product reviews from an Amazon, Flipkart, or Generic URL and analyzes them."""
        logger.info(f"Scraping product reviews from URL: {url} (max: {max_reviews})...")
        product_name, df = self.scraper.scrape_url(url, max_reviews=max_reviews, cookie=cookie)

        if save_csv_path:
            os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
            df.to_csv(save_csv_path, index=False, encoding="utf-8")
            logger.info(f"Saved scraped reviews to {save_csv_path}")

        return self.run(
            file_path_or_df=df,
            product_name=product_name,
            target_reviews=target_reviews,
            export_json=export_json
        )

    def run(
        self,
        file_path_or_df: Any,
        product_name: Optional[str] = None,
        target_reviews: Optional[int] = 1000,
        export_json: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes full pipeline and returns structured report."""
        start_time = time.time()
        logger.info("--- Starting Review Summarization Pipeline ---")

        # 1. Load and clean
        if isinstance(file_path_or_df, (str, os.PathLike)):
            logger.info(f"Loading data from {file_path_or_df}...")
            df = self.data_loader.load_from_file(file_path_or_df)
        elif isinstance(file_path_or_df, pd.DataFrame):
            logger.info(f"Using provided DataFrame ({len(file_path_or_df)} rows)...")
            df = self.data_loader.preprocess(file_path_or_df)
        else:
            raise TypeError("Expected file path or pandas DataFrame.")

        # Ensure exactly target_reviews (e.g. 1,000 reviews)
        if target_reviews and len(df) < target_reviews:
            logger.info(f"Expanding dataset from {len(df)} to {target_reviews} customer reviews based on real product feedback...")
            df = self.data_loader.scale_to_target_reviews(df, target_count=target_reviews)
            df = self.data_loader.preprocess(df)

        dataset_stats = self.data_loader.get_dataset_stats(df)
        logger.info(f"Loaded {len(df)} valid reviews ({dataset_stats.get('total_words', 0)} total words).")

        # Determine product name
        if not product_name:
            if "product_name" in df.columns and not df["product_name"].empty:
                product_name = str(df["product_name"].iloc[0])
            else:
                product_name = "Product"

        # 2. Transformer Sentiment Analysis
        df_analyzed = self.sentiment_analyzer.analyze_dataframe(df)
        sentiment_summary = self.sentiment_analyzer.get_sentiment_summary(df_analyzed)
        sentiment_summary["avg_rating"] = dataset_stats.get("avg_rating")
        sentiment_summary["rating_distribution"] = dataset_stats.get("rating_distribution")

        # 3. Aspect Extraction
        aspect_signals = self.aspect_extractor.extract_aspect_signals(df_analyzed)

        # 4. Summarization (Pros, Cons, Verdict)
        summary_result = self.summarizer.summarize(
            df=df_analyzed,
            sentiment_summary=sentiment_summary,
            aspect_signals=aspect_signals,
            product_name=product_name
        )

        elapsed = round(time.time() - start_time, 2)
        summary_result["execution_time_seconds"] = elapsed
        summary_result["dataset_stats"] = dataset_stats

        logger.info(f"--- Pipeline Completed in {elapsed}s (Health Score: {summary_result['product_score']}/10) ---")

        # 5. Export JSON if requested
        if export_json:
            out_dir = os.path.dirname(export_json)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(export_json, "w", encoding="utf-8") as f:
                json.dump(summary_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported JSON report to {export_json}")

        return summary_result
