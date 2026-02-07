"""Customer reviews/feedback Pydantic models."""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class ReviewBase(BaseModel):
    product_id: Optional[int] = Field(None, description="Product id (optional)")
    sku: Optional[str] = Field(None, max_length=50)
    source: str = Field("reviews", description="reviews|surveys|social_media|support")
    review_text: str = Field(..., min_length=1, max_length=20000)
    rating: Optional[int] = Field(None, ge=1, le=5)
    review_date: Optional[datetime] = None

    @validator("source")
    def validate_source(cls, v):
        allowed = {"reviews", "surveys", "social_media", "support", "all"}
        if v not in allowed:
            raise ValueError(f"source must be one of: {sorted(allowed)}")
        return v


class ReviewCreate(ReviewBase):
    pass


class ReviewResponse(ReviewBase):
    id: int
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    created_date: datetime

    class Config:
        from_attributes = True


class ReviewImportResponse(BaseModel):
    imported: int
    skipped: int
    errors: List[Dict[str, Any]] = Field(default_factory=list)


class SentimentRequest(BaseModel):
    product_id: Optional[int] = None
    category: Optional[str] = None
    time_period: str = Field("30_days", description="7_days|30_days|90_days")
    source: str = Field("all", description="reviews|social_media|surveys|support|all")
    aspect_level: bool = True


class ParseRequest(BaseModel):
    raw_query: str = Field(..., min_length=1)
    context_history: Optional[List[Dict[str, str]]] = None


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    filters: Optional[Dict[str, Any]] = None
    top_k: int = Field(10, ge=1, le=50)
