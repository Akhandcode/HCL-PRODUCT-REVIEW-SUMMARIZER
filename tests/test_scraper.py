"""
Unit tests for ReviewScraper
"""

import unittest
from unittest.mock import patch, MagicMock
from src.scraper import ReviewScraper


class TestReviewScraper(unittest.TestCase):
    def setUp(self):
        self.scraper = ReviewScraper()

    def test_detect_platform(self):
        self.assertEqual(self.scraper.detect_platform("https://www.amazon.com/dp/B08N5WRWNW"), "amazon")
        self.assertEqual(self.scraper.detect_platform("https://www.amazon.in/Sony-WH-1000XM4/dp/B0863TXGM3"), "amazon")
        self.assertEqual(self.scraper.detect_platform("https://www.flipkart.com/apple-airpods-pro/p/itm123"), "flipkart")
        self.assertEqual(self.scraper.detect_platform("https://mybrandstore.com/products/wireless-earbuds"), "generic")

    def test_extract_amazon_asin(self):
        url = "https://www.amazon.in/Sony-WH-1000XM4-Cancelling-Headphones-Bluetooth/dp/B0863TXGM3/ref=sr_1_1"
        asin, tld = self.scraper._extract_amazon_asin(url)
        self.assertEqual(asin, "B0863TXGM3")
        self.assertEqual(tld, "in")

        url_com = "https://www.amazon.com/product-reviews/B08N5WRWNW"
        asin_com, tld_com = self.scraper._extract_amazon_asin(url_com)
        self.assertEqual(asin_com, "B08N5WRWNW")
        self.assertEqual(tld_com, "com")

    def test_scrape_generic_json_ld(self):
        sample_html = """
        <html>
        <head>
          <script type="application/ld+json">
          {
            "@context": "https://schema.org/",
            "@type": "Product",
            "name": "NovaSound Elite ANC Headphones",
            "review": [
              {
                "@type": "Review",
                "name": "Terrific audio",
                "reviewBody": "These headphones cancel airplane drone completely and the bass is phenomenal.",
                "reviewRating": {"@type": "Rating", "ratingValue": 5}
              },
              {
                "@type": "Review",
                "name": "App is glitchy",
                "reviewBody": "Good comfort but the companion app loses bluetooth connection constantly.",
                "reviewRating": {"@type": "Rating", "ratingValue": 2}
              }
            ]
          }
          </script>
        </head>
        <body><h1>NovaSound Elite ANC Headphones</h1></body>
        </html>
        """
        with patch.object(self.scraper.session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = sample_html
            mock_get.return_value = mock_resp

            prod_name, reviews = self.scraper.scrape_generic("https://mockstore.com/item")
            self.assertEqual(prod_name, "NovaSound Elite ANC Headphones")
            self.assertEqual(len(reviews), 2)
            self.assertEqual(reviews[0]["rating"], 5.0)
            self.assertIn("airplane drone", reviews[0]["review_text"])
            self.assertEqual(reviews[1]["rating"], 2.0)


if __name__ == "__main__":
    unittest.main()
