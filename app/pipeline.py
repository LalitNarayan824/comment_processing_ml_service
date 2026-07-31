"""Heavy ML inference pipeline: model loading, caching, and prediction."""

from typing import Dict
from functools import lru_cache

from transformers import pipeline

from app.config import (
    DEVICE,
    INTENT_MODEL,
    SENTIMENT_MODEL,
    INTENT_LABEL_MAP,
    LRU_CACHE_SIZE,
)
from app.toxicity import is_toxic

# --- ML Models ---
intent_pipe = pipeline("text-classification", model=INTENT_MODEL, device=DEVICE)
sent_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=DEVICE)


@lru_cache(maxsize=LRU_CACHE_SIZE)
def get_heavy_predictions(text: str) -> Dict:
    """Run intent, toxicity, and sentiment inference (cached)."""
    # Intent
    intent_output = intent_pipe(text)[0]
    raw_label = str(intent_output["label"])

    # Toxicity
    tox_res = is_toxic(text)

    # Sentiment
    sent_res = sent_pipe(text)[0]

    return {
        "intent": INTENT_LABEL_MAP.get(raw_label, "Unknown"),
        "intent_confidence": round(intent_output["score"], 3),
        "sentiment": sent_res["label"].lower(),
        "sentiment_confidence": round(sent_res["score"], 3),
        "toxicity": tox_res,
    }
