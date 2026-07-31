"""Pydantic schemas for request/response models."""

from pydantic import BaseModel, Field


class CommentInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class AnalysisResponse(BaseModel):
    intent: str
    intent_confidence: float
    sentiment: str
    sentiment_confidence: float
    toxicity: bool
    is_spam: bool
