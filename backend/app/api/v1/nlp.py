"""\
SmartShelf AI - NLP API v1

Endpoints for NLP parsing, semantic search, and customer sentiment analysis.
"""

from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any, Optional, List
import pandas as pd
import io
from datetime import datetime, timedelta
import logging

from ...database import get_db
from ...database import Product, CustomerReview, CompetitorPrice
from ...core.exceptions import ValidationError, NotFoundError, DataProcessingError
from ...models.reviews import (
    ParseRequest,
    SemanticSearchRequest,
    SentimentRequest,
    ReviewImportResponse,
)

# Reuse NLP components from copilot_chatbot package
from copilot_chatbot.nlp.intent_engine import NLPIntentEngine
from copilot_chatbot.nlp.semantic_search import SemanticSearchEngine
from copilot_chatbot.nlp.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()

_intent_engine = NLPIntentEngine()
_semantic_search = SemanticSearchEngine()
_sentiment = SentimentAnalyzer()


def _parse_time_period(time_period: str) -> datetime:
    mapping = {
        "7_days": 7,
        "30_days": 30,
        "90_days": 90,
    }
    if time_period not in mapping:
        raise ValidationError("time_period must be one of: 7_days, 30_days, 90_days")
    return datetime.utcnow() - timedelta(days=mapping[time_period])


@router.post("/parse")
async def parse_user_query(request: ParseRequest):
    """Parse raw query into intent + entities."""
    result = await _intent_engine.parse_query(request.raw_query, request.context_history)
    return {
        "raw_query": request.raw_query,
        "parsed_intent": {
            "primary_intent": result.primary_intent.value,
            "secondary_intents": [i.value for i in result.secondary_intents],
            "confidence": result.confidence,
        },
        "entities": {
            "items": [
                {
                    "type": e.type.value,
                    "value": e.value,
                    "normalized_value": e.normalized_value,
                    "confidence": e.confidence,
                }
                for e in result.entities
            ]
        },
        "query_type": result.query_type,
        "required_functions": result.required_functions,
        "disambiguation_needed": result.disambiguation_needed,
    }


@router.post("/search")
async def semantic_product_search(request: SemanticSearchRequest):
    """Semantic product search (demo catalog inside SemanticSearchEngine)."""
    return _semantic_search.semantic_product_search(request.query, request.filters, request.top_k)


@router.post("/sentiment")
async def analyze_customer_sentiment(request: SentimentRequest, db: Session = Depends(get_db)):
    """Analyze sentiment for reviews in DB filtered by product/category/time/source."""
    start_date = _parse_time_period(request.time_period)

    q = db.query(CustomerReview)

    if request.source != "all":
        q = q.filter(CustomerReview.source == request.source)

    if request.product_id is not None:
        q = q.filter(CustomerReview.product_id == request.product_id)

    if request.category is not None:
        # Join products for category filtering
        q = q.join(Product, Product.id == CustomerReview.product_id).filter(Product.category == request.category)

    q = q.filter(CustomerReview.review_date >= start_date)

    reviews = q.order_by(CustomerReview.review_date.desc()).limit(2000).all()
    texts = [r.review_text for r in reviews]

    analysis = _sentiment.analyze_texts(texts, aspect_level=request.aspect_level)

    return {
        "filters": {
            "product_id": request.product_id,
            "category": request.category,
            "time_period": request.time_period,
            "source": request.source,
        },
        "review_count": len(texts),
        "analysis": analysis,
    }


@router.post("/reviews/import", response_model=ReviewImportResponse)
async def import_reviews_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Import reviews from a CSV file into SQLite.

    Expected columns:
    - review_text (required)
    - product_id (optional)
    - sku (optional)
    - source (optional)
    - rating (optional)
    - review_date (optional)
    """
    if not file.filename.endswith(".csv"):
        raise ValidationError("Only CSV files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise DataProcessingError(f"Failed to read CSV: {str(e)}")

    if df.empty:
        raise ValidationError("CSV file is empty")

    if "review_text" not in df.columns:
        raise ValidationError("Missing required column: review_text")

    imported = 0
    skipped = 0
    errors: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        try:
            text = str(row.get("review_text", "")).strip()
            if not text:
                skipped += 1
                continue

            product_id = row.get("product_id")
            if pd.notna(product_id):
                try:
                    product_id = int(product_id)
                except Exception:
                    product_id = None
            else:
                product_id = None

            sku = row.get("sku")
            if pd.isna(sku):
                sku = None
            else:
                sku = str(sku)

            source = row.get("source")
            if pd.isna(source) or not str(source).strip():
                source = "reviews"
            source = str(source).strip()

            rating = row.get("rating")
            if pd.isna(rating):
                rating = None
            else:
                try:
                    rating = int(rating)
                except Exception:
                    rating = None

            review_date = row.get("review_date")
            if pd.notna(review_date):
                try:
                    review_date = pd.to_datetime(review_date).to_pydatetime()
                except Exception:
                    review_date = datetime.utcnow()
            else:
                review_date = datetime.utcnow()

            # If product_id missing but sku present, try resolve sku
            if product_id is None and sku:
                p = db.query(Product).filter(Product.sku == sku).first()
                if p:
                    product_id = p.id

            rec = CustomerReview(
                product_id=product_id,
                sku=sku,
                source=source,
                review_text=text,
                rating=rating,
                review_date=review_date,
            )
            db.add(rec)
            imported += 1

            if imported % 500 == 0:
                db.commit()

        except Exception as e:
            errors.append({"row": int(idx), "error": str(e)})

    db.commit()

    return ReviewImportResponse(imported=imported, skipped=skipped, errors=errors)
