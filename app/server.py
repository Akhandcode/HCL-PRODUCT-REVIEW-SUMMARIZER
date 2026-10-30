"""
FastAPI Server for Product Review Summarizer Web Dashboard
"""

import os
import json
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.pipeline import ReviewSummarizationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Product Review Summarizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "outputs", "summary_report.json")
DEFAULT_CSV = os.path.join(DATA_DIR, "sample_1000_reviews.csv")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "outputs"), exist_ok=True)

pipeline = ReviewSummarizationPipeline(batch_size=64)


@app.get("/api/results")
async def get_latest_results():
    """Returns the latest generated summary report."""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return JSONResponse(content=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading results: {e}")

    # If no results cached, run pipeline on default sample dataset
    if os.path.exists(DEFAULT_CSV):
        res = pipeline.run(DEFAULT_CSV, export_json=OUTPUT_FILE)
        return JSONResponse(content=res)

    raise HTTPException(status_code=404, detail="No review results found. Please run an analysis first.")


from pydantic import BaseModel, Field

class ScrapeRequest(BaseModel):
    url: str = Field(..., description="E-commerce Product or Reviews URL")
    max_reviews: int = Field(100, ge=5, le=1000, description="Maximum raw reviews to scrape")
    target_reviews: int = Field(1000, ge=10, le=2000, description="Target review count for synthesis (default: 1000)")
    cookie: Optional[str] = Field(None, description="Optional session cookie for deep review pagination")


@app.post("/api/scrape-and-analyze")
async def scrape_and_analyze(req: ScrapeRequest):
    """Scrapes product reviews from Amazon, Flipkart, or Generic URL and analyzes them."""
    try:
        scraped_csv = os.path.join(DATA_DIR, "outputs", "last_scraped_reviews.csv")
        result = pipeline.run_from_url(
            url=req.url,
            max_reviews=req.max_reviews,
            cookie=req.cookie,
            target_reviews=req.target_reviews,
            export_json=OUTPUT_FILE,
            save_csv_path=scraped_csv
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Scraping & Analysis error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/reviews")
async def get_raw_reviews(search: Optional[str] = None, limit: int = 50):
    """Returns a list of raw reviews from the most recently analyzed dataset."""
    scraped_csv = os.path.join(DATA_DIR, "outputs", "last_scraped_reviews.csv")
    csv_to_use = scraped_csv if os.path.exists(scraped_csv) else DEFAULT_CSV

    if not os.path.exists(csv_to_use):
        return {"reviews": []}

    try:
        df = pd.read_csv(csv_to_use)
        if search:
            s = search.lower()
            text_match = df["review_text"].astype(str).str.lower().str.contains(s, na=False)
            title_match = df["review_title"].astype(str).str.lower().str.contains(s, na=False) if "review_title" in df.columns else False
            df = df[text_match | title_match]

        records = df.head(limit).to_dict(orient="records")
        return {"total": len(df), "reviews": records}
    except Exception as e:
        return {"total": 0, "reviews": [], "error": str(e)}


@app.post("/api/analyze")
async def analyze_custom_file(file: UploadFile = File(...), product_name: Optional[str] = Form(None)):
    """Uploads and analyzes a custom review CSV/JSON."""
    try:
        temp_path = os.path.join(DATA_DIR, f"temp_{file.filename}")
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        res = pipeline.run(temp_path, product_name=product_name, export_json=OUTPUT_FILE)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(content=res)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Review Summarizer Dashboard Loading...</h1>")


# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
