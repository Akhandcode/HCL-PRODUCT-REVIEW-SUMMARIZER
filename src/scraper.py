"""
Multi-Platform Product Review Scraper
Extracts customer reviews from Amazon, Flipkart, and Generic E-Commerce sites (Schema.org / JSON-LD / HTML).
"""

import os
import re
import json
import time
import random
import logging
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
]


class ReviewScraper:
    """Scrapes product reviews from Amazon, Flipkart, and Generic e-commerce web pages."""

    def __init__(self, timeout: int = 15, delay_range: Tuple[float, float] = (0.3, 0.7)):
        self.timeout = timeout
        self.delay_range = delay_range
        self.session = requests.Session()

    def _get_headers(self, referer: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin" if referer else "none",
            "Sec-Fetch-User": "?1",
        }
        if referer:
            headers["Referer"] = referer
        return headers

    def detect_platform(self, url: str) -> str:
        """Identifies target platform based on domain."""
        domain = urlparse(url).netloc.lower()
        if "amazon" in domain:
            return "amazon"
        elif "flipkart" in domain:
            return "flipkart"
        return "generic"

    # =========================================================================
    # Amazon Scraper
    # =========================================================================
    def _extract_amazon_asin(self, url: str) -> Tuple[Optional[str], str]:
        """Extracts ASIN (10-char alphanumeric) and domain TLD from Amazon URL."""
        domain = urlparse(url).netloc
        tld = "com"
        for t in ["in", "co.uk", "de", "ca", "fr", "es", "it", "co.jp", "com.au"]:
            if domain.endswith(f".{t}"):
                tld = t
                break

        patterns = [
            r"/dp/([A-Z0-9]{10})",
            r"/product-reviews/([A-Z0-9]{10})",
            r"/gp/product/([A-Z0-9]{10})",
            r"/ASIN/([A-Z0-9]{10})"
        ]
        for pat in patterns:
            match = re.search(pat, url, re.IGNORECASE)
            if match:
                return match.group(1).upper(), tld
        return None, tld

    def _parse_amazon_review_card(self, card, product_name: str, index: int) -> Optional[Dict[str, Any]]:
        """Extracts fields from an Amazon review card element with modern selectors."""
        # Decompose teaser divs to prevent duplicate preview text
        for teaser in card.find_all("div", class_=re.compile(r"a-teaser")):
            teaser.decompose()

        # Review ID
        rev_id = card.get("id") or f"AMZ-{index:04d}"

        # Rating
        rating = 4.0
        rating_elem = card.find("i", attrs={"data-hook": re.compile(r"star-rating", re.I)}) or card.find("i", class_=re.compile(r"a-star-"))
        if rating_elem:
            rating_match = re.search(r"(\d+(\.\d+)?)", rating_elem.text)
            if rating_match:
                try:
                    rating = float(rating_match.group(1))
                except ValueError:
                    pass

        # Title
        title = ""
        title_elem = (
            card.find(attrs={"data-hook": re.compile(r"review.*title", re.I)}) or
            card.find("h5") or
            card.select_one("a[data-hook='review-title']")
        )
        if title_elem:
            title = title_elem.text.strip()

        # Body
        body = ""
        body_elem = (
            card.find("div", attrs={"data-hook": re.compile(r"reviewRichContent", re.I)}) or
            card.find(attrs={"data-hook": re.compile(r"reviewText", re.I)}) or
            card.find(attrs={"data-hook": re.compile(r"review-body", re.I)}) or
            card.select_one("span[data-hook='review-body']")
        )
        if body_elem:
            body = body_elem.text.strip()
            # Remove "Read more" / "Read less" UI expander text
            body = re.sub(r"Read more|Read less", "", body, flags=re.IGNORECASE).strip()

        if not body or len(body) < 10:
            return None

        # Verified purchase
        verified = bool(card.find(attrs={"data-hook": re.compile(r"avp-badge", re.I)}))

        # Date
        date_elem = card.find(attrs={"data-hook": re.compile(r"review-date", re.I)})
        rev_date = date_elem.text.strip() if date_elem else ""

        return {
            "review_id": rev_id,
            "product_name": product_name,
            "rating": int(rating) if rating.is_integer() else rating,
            "review_title": title,
            "review_text": body,
            "verified_purchase": verified,
            "review_date": rev_date,
            "helpful_votes": 0
        }

    def scrape_amazon(
        self,
        url: str,
        max_reviews: int = 100,
        cookie: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Scrapes reviews from Amazon product detail and reviews pages across variations."""
        asin, tld = self._extract_amazon_asin(url)
        if not asin:
            raise ValueError(f"Could not extract Amazon ASIN from URL: {url}")

        logger.info(f"Targeting Amazon ASIN: {asin} (amazon.{tld})")
        product_name = f"Amazon Product ({asin})"
        reviews = []
        seen_ids = set()
        swatches = []

        # 1. Primary Pass: Fetch Product Detail Page (always public, no login redirect)
        dp_url = f"https://www.amazon.{tld}/dp/{asin}"
        logger.info(f"Fetching product detail page: {dp_url}")

        headers = self._get_headers()
        if cookie:
            headers["Cookie"] = cookie

        try:
            resp_dp = self.session.get(dp_url, headers=headers, timeout=self.timeout)
            if resp_dp.status_code == 200:
                soup_dp = BeautifulSoup(resp_dp.text, "html.parser")

                # Extract product title
                title_elem = soup_dp.select_one("span#productTitle") or soup_dp.select_one("h1")
                if title_elem and title_elem.text.strip():
                    product_name = title_elem.text.strip()
                    logger.info(f"Detected product title: {product_name}")

                # Extract review cards present on detail page
                dp_cards = soup_dp.find_all("div", attrs={"data-hook": "review"})
                logger.info(f"Found {len(dp_cards)} review cards on main detail page.")
                for card in dp_cards:
                    rev = self._parse_amazon_review_card(card, product_name, len(reviews) + 1)
                    if rev and rev["review_id"] not in seen_ids:
                        seen_ids.add(rev["review_id"])
                        reviews.append(rev)

                # Discover sibling variation swatches (other colors/editions of this product)
                raw_swatches = [
                    li.get("data-asin") or li.get("data-defaultasin")
                    for li in soup_dp.select("li[data-asin], li[data-defaultasin], div[data-asin]")
                ]
                swatches = [s for s in set(raw_swatches) if s and len(s) == 10 and s != asin]
                if swatches:
                    logger.info(f"Found {len(swatches)} variation swatches for this product: {swatches}")

        except Exception as e:
            logger.warning(f"Error fetching product detail page: {e}")

        # 2. Secondary Pass: If cookie is provided, attempt deep review pagination
        if len(reviews) < max_reviews and cookie:
            page = 1
            max_pages = (max_reviews // 10) + 1
            while len(reviews) < max_reviews and page <= max_pages:
                review_url = f"https://www.amazon.{tld}/product-reviews/{asin}?pageNumber={page}&sortBy=recent"
                logger.info(f"Attempting authenticated reviews page {page}: {review_url}")
                try:
                    resp = self.session.get(review_url, headers=headers, timeout=self.timeout)
                    if resp.status_code != 200:
                        break
                    soup = BeautifulSoup(resp.text, "html.parser")
                    page_title = soup.title.string.lower() if soup.title and soup.title.string else ""
                    if "sign in" in page_title or "captcha" in page_title:
                        logger.warning("Amazon sign-in wall encountered despite cookie. Switching to variant harvesting.")
                        break

                    cards = soup.find_all("div", attrs={"data-hook": "review"})
                    if not cards:
                        break

                    for card in cards:
                        if len(reviews) >= max_reviews:
                            break
                        rev = self._parse_amazon_review_card(card, product_name, len(reviews) + 1)
                        if rev and rev["review_id"] not in seen_ids:
                            seen_ids.add(rev["review_id"])
                            reviews.append(rev)

                    page += 1
                    time.sleep(random.uniform(*self.delay_range))
                except Exception as e:
                    logger.warning(f"Pagination error: {e}")
                    break

        # 3. Tertiary Pass: Multi-Variant Auto-Harvesting across product variations
        if len(reviews) < max_reviews and swatches:
            logger.info(f"Expanding review harvesting across {len(swatches)} product variants...")
            for swatch_asin in swatches:
                if len(reviews) >= max_reviews:
                    break
                swatch_url = f"https://www.amazon.{tld}/dp/{swatch_asin}"
                logger.info(f"Fetching sibling variant: {swatch_url}")
                try:
                    time.sleep(random.uniform(*self.delay_range))
                    resp_swatch = self.session.get(swatch_url, headers=headers, timeout=self.timeout)
                    if resp_swatch.status_code == 200:
                        soup_swatch = BeautifulSoup(resp_swatch.text, "html.parser")
                        swatch_cards = soup_swatch.find_all("div", attrs={"data-hook": "review"})
                        for card in swatch_cards:
                            if len(reviews) >= max_reviews:
                                break
                            rev = self._parse_amazon_review_card(card, product_name, len(reviews) + 1)
                            if rev and rev["review_id"] not in seen_ids:
                                seen_ids.add(rev["review_id"])
                                reviews.append(rev)
                except Exception as e:
                    logger.warning(f"Error harvesting variant {swatch_asin}: {e}")

        logger.info(f"Completed Amazon scraping. Harvested total {len(reviews)} unique reviews.")
        return product_name, reviews

    # =========================================================================
    # Flipkart Scraper
    # =========================================================================
    def scrape_flipkart(self, url: str, max_reviews: int = 100) -> Tuple[str, List[Dict[str, Any]]]:
        """Scrapes reviews from Flipkart product or review pages."""
        logger.info(f"Targeting Flipkart URL: {url}")
        reviews = []
        product_name = "Flipkart Product"

        # Transform product URL to reviews page if needed
        review_url_base = url
        if "/p/" in url and "/product-reviews/" not in url:
            review_url_base = url.replace("/p/", "/product-reviews/")

        page = 1
        max_pages = (max_reviews // 10) + 2

        while len(reviews) < max_reviews and page <= max_pages:
            sep = "&" if "?" in review_url_base else "?"
            target_page_url = f"{review_url_base}{sep}page={page}"
            logger.info(f"Fetching Flipkart page {page}: {target_page_url}")

            try:
                resp = self.session.get(target_page_url, headers=self._get_headers(url), timeout=self.timeout)
                if resp.status_code != 200:
                    break

                soup = BeautifulSoup(resp.text, "html.parser")

                # Product title
                title_elem = soup.select_one("div._2s4QId") or soup.select_one("span.B_NuCI") or soup.select_one("h1")
                if title_elem and product_name == "Flipkart Product":
                    product_name = title_elem.text.strip()

                # Review containers on Flipkart usually have class combinations
                # e.g., div.col.EPCmJX, div._27M-vq, div._16PBlm, or [class*="review"]
                review_cards = (
                    soup.select("div.col.EPCmJX") or
                    soup.select("div._27M-vq") or
                    soup.select("div._16PBlm") or
                    soup.select("div[class*='cPHDOP'] div[class*='col']")
                )

                if not review_cards:
                    logger.info("No review cards found on Flipkart page.")
                    break

                for idx, card in enumerate(review_cards):
                    if len(reviews) >= max_reviews:
                        break

                    # Rating
                    rating = 4.0
                    rating_elem = card.select_one("div._3LWZlK") or card.select_one("div.XQDdHH")
                    if rating_elem:
                        try:
                            rating = float(rating_elem.text.strip())
                        except ValueError:
                            pass

                    # Title
                    title_elem = card.select_one("p._2-N8zT") or card.select_one("p.z9E0IG")
                    title = title_elem.text.strip() if title_elem else ""

                    # Body
                    body_elem = card.select_one("div.t-ZTKy") or card.select_one("div.ZmyHeo")
                    body = body_elem.text.strip() if body_elem else ""
                    # Remove "READ MORE" button text Flipkart includes
                    body = re.sub(r"READ MORE", "", body, flags=re.IGNORECASE).strip()

                    if not body:
                        continue

                    # Verified purchase
                    verified = bool(card.select_one("p._2mcPpE") or card.select_one("span._1J7GCh"))

                    reviews.append({
                        "review_id": f"FLP-{len(reviews)+1:04d}",
                        "product_name": product_name,
                        "rating": int(rating) if rating.is_integer() else rating,
                        "review_title": title,
                        "review_text": body,
                        "verified_purchase": verified,
                        "review_date": "",
                        "helpful_votes": 0
                    })

                page += 1
                time.sleep(random.uniform(*self.delay_range))

            except Exception as e:
                logger.error(f"Error scraping Flipkart page {page}: {e}")
                break

        return product_name, reviews

    # =========================================================================
    # Generic E-Commerce Scraper (JSON-LD Schema & HTML)
    # =========================================================================
    def scrape_generic(self, url: str, max_reviews: int = 50) -> Tuple[str, List[Dict[str, Any]]]:
        """Extracts reviews using Schema.org JSON-LD or standard HTML structure."""
        logger.info(f"Extracting generic reviews from: {url}")
        resp = self.session.get(url, headers=self._get_headers(), timeout=self.timeout)
        if resp.status_code != 200:
            raise ConnectionError(f"HTTP {resp.status_code} while fetching {url}")

        soup = BeautifulSoup(resp.text, "html.parser")
        product_name = "E-Commerce Product"
        title_elem = soup.select_one("h1") or soup.select_one("meta[property='og:title']")
        if title_elem:
            product_name = title_elem.get("content") if title_elem.name == "meta" else title_elem.text.strip()

        reviews = []

        # 1. Search for Schema.org JSON-LD Review blocks
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "{}")
                items = data if isinstance(data, list) else [data]

                for item in items:
                    # Check Product schema
                    if item.get("@type") in ["Product", "IndividualProduct"] and "name" in item:
                        product_name = item["name"]

                    # Extract reviews inside product schema or top-level
                    rev_list = item.get("review") or item.get("reviews") or []
                    if isinstance(rev_list, dict):
                        rev_list = [rev_list]

                    for r in rev_list:
                        body = r.get("reviewBody") or r.get("description") or ""
                        if not body:
                            continue
                        
                        rating = 5.0
                        rating_obj = r.get("reviewRating", {})
                        if isinstance(rating_obj, dict) and "ratingValue" in rating_obj:
                            try:
                                rating = float(rating_obj["ratingValue"])
                            except ValueError:
                                pass

                        reviews.append({
                            "review_id": f"GEN-{len(reviews)+1:04d}",
                            "product_name": product_name,
                            "rating": rating,
                            "review_title": r.get("name") or "",
                            "review_text": body,
                            "verified_purchase": True,
                            "review_date": r.get("datePublished", ""),
                            "helpful_votes": 0
                        })
            except Exception:
                continue

        # 2. Fallback to common review HTML card classes if JSON-LD had few reviews
        if len(reviews) < 5:
            candidate_selectors = [
                ".review-item", ".review-card", ".user-review", ".comment-body",
                "[itemprop='review']", ".customer-review", ".feedback-item"
            ]
            for sel in candidate_selectors:
                cards = soup.select(sel)
                if cards:
                    for card in cards:
                        if len(reviews) >= max_reviews:
                            break
                        body_elem = card.select_one(".review-text") or card.select_one("p") or card
                        body = body_elem.text.strip() if body_elem else ""
                        if len(body) < 20:
                            continue

                        reviews.append({
                            "review_id": f"GEN-{len(reviews)+1:04d}",
                            "product_name": product_name,
                            "rating": 4,
                            "review_title": "Customer Review",
                            "review_text": body,
                            "verified_purchase": True,
                            "review_date": "",
                            "helpful_votes": 0
                        })
                    if reviews:
                        break

        return product_name, reviews[:max_reviews]

    # =========================================================================
    # High-Level Orchestrator
    # =========================================================================
    def scrape_url(
        self,
        url: str,
        max_reviews: int = 100,
        cookie: Optional[str] = None
    ) -> Tuple[str, pd.DataFrame]:
        """
        Scrapes reviews from any supported URL (Amazon, Flipkart, or Generic).
        Returns (product_name, DataFrame).
        """
        platform = self.detect_platform(url)
        logger.info(f"Detected platform '{platform}' for URL: {url}")

        if platform == "amazon":
            product_name, reviews = self.scrape_amazon(url, max_reviews=max_reviews, cookie=cookie)
        elif platform == "flipkart":
            product_name, reviews = self.scrape_flipkart(url, max_reviews=max_reviews)
        else:
            product_name, reviews = self.scrape_generic(url, max_reviews=max_reviews)

        # If direct scraping returned 0 reviews due to aggressive anti-bot protection,
        # provide helpful diagnostics
        if not reviews:
            raise RuntimeError(
                f"No reviews could be scraped from '{url}'. The site may be blocking automated requests "
                f"or the product page has 0 customer reviews. Check the URL or try with a review page link."
            )

        df = pd.DataFrame(reviews)
        logger.info(f"Successfully scraped {len(df)} reviews for '{product_name}'.")
        return product_name, df
