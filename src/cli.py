"""
Command-Line Interface (CLI) for Product Review Summarizer
Provides terminal-based summarization with formatted tables, metrics, and export options.
"""

import argparse
import sys
import os
import json
from src.pipeline import ReviewSummarizationPipeline


def print_banner():
    print("""
========================================================================
   LLM-POWERED PRODUCT REVIEW SUMMARIZATION & SENTIMENT TOOL
========================================================================
    """)


def format_cli_report(res: dict):
    print("=" * 72)
    print(f" PRODUCT: {res['product_name'].upper()}")
    print(f" HEALTH SCORE: {res['product_score']} / 10.0   |   VERDICT: {res['verdict']}")
    print("=" * 72)
    print(f"\n[EXECUTIVE SUMMARY]\n{res['executive_summary']}\n")

    # Sentiment Breakdown
    s = res["sentiment_overview"]
    print("-" * 72)
    print("SENTIMENT DISTRIBUTION")
    print(f"Total Reviews Analyzed : {res['total_reviews_analyzed']}")
    print(f"Positive Sentiment     : {s.get('positive_pct', 0)}% ({s.get('positive_count', 0)} reviews)")
    print(f"Negative Sentiment     : {s.get('negative_pct', 0)}% ({s.get('negative_count', 0)} reviews)")
    print(f"Neutral / Mixed        : {s.get('neutral_pct', 0)}% ({s.get('neutral_count', 0)} reviews)")
    print(f"Net Sentiment Score    : {s.get('net_sentiment_score', 0)}")
    print(f"Average Star Rating    : {s.get('avg_rating', 'N/A')} / 5.0")
    print("-" * 72)

    # Top Pros
    print("\n[+] TOP PROS (KEY STRENGTHS)")
    print("=" * 72)
    for idx, p in enumerate(res.get("pros", [])[:5], 1):
        print(f" {idx}. [{p['aspect']}] - {p['sentiment_score']} ({p['mentions']} mentions)")
        print(f"    * Headline: \"{p['headline']}\"")
        if p.get("sample_quotes"):
            print(f"    * Quote: \"{p['sample_quotes'][0]}\"")
        print()

    # Top Cons
    print("[-] TOP CONS (KEY DRAWBACKS & COMPLAINTS)")
    print("=" * 72)
    for idx, c in enumerate(res.get("cons", [])[:5], 1):
        print(f" {idx}. [{c['aspect']}] - {c['dissatisfaction_rate']} ({c['mentions']} complaints) [{c['severity']}]")
        print(f"    * Headline: \"{c['headline']}\"")
        if c.get("sample_quotes"):
            print(f"    * Quote: \"{c['sample_quotes'][0]}\"")
        print()

    # Recommendations
    print("=" * 72)
    print("TARGET AUDIENCE & BUYER GUIDANCE")
    print("=" * 72)
    print("WHO SHOULD BUY:")
    for item in res.get("target_audience", {}).get("who_should_buy", []):
        print(f"  + {item}")
    print("\nWHO SHOULD AVOID / CONSIDER ALTERNATIVES:")
    for item in res.get("target_audience", {}).get("who_should_avoid", []):
        print(f"  - {item}")
    print("=" * 72)
    print(f"Completed in {res.get('execution_time_seconds', 0)} seconds.\n")


def main():
    parser = argparse.ArgumentParser(description="LLM Product Review Summarizer")
    parser.add_argument("--url", "-u", type=str, default=None, help="Product URL to scrape reviews from (Amazon, Flipkart, Generic)")
    parser.add_argument("--max-reviews", "-m", type=int, default=100, help="Maximum reviews to scrape (default: 100)")
    parser.add_argument("--target-reviews", "-t", type=int, default=1000, help="Target total reviews for deep AI summarization (default: 1000)")
    parser.add_argument("--cookie", "-c", type=str, default=None, help="Optional Amazon/Platform session cookie for deep review pagination")
    parser.add_argument("--output", "-o", type=str, default="data/outputs/summary_report.json", help="Path to export JSON report")
    parser.add_argument("--batch-size", "-b", type=int, default=128, help="Inference batch size")
    parser.add_argument("--product-name", "-p", type=str, default=None, help="Optional product name override")

    args = parser.parse_args()

    print_banner()

    pipeline = ReviewSummarizationPipeline(batch_size=args.batch_size)

    if args.url:
        print(f"Scraping product reviews from URL: {args.url} (max scraped: {args.max_reviews}, target synthesis: {args.target_reviews})...")
        save_csv = "data/outputs/last_scraped_reviews.csv"
        try:
            result = pipeline.run_from_url(
                url=args.url,
                max_reviews=args.max_reviews,
                cookie=args.cookie,
                target_reviews=args.target_reviews,
                export_json=args.output,
                save_csv_path=save_csv
            )
        except Exception as e:
            print(f"\n[Scraper Error] {e}")
            sys.exit(1)
    else:
        input_file = args.input or "data/sample_1000_reviews.csv"
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist.")
            sys.exit(1)

        result = pipeline.run(
            file_path_or_df=input_file,
            product_name=args.product_name,
            target_reviews=args.target_reviews,
            export_json=args.output
        )

    format_cli_report(result)


if __name__ == "__main__":
    main()
