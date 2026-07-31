"""Centralized configuration: env vars, model names, thresholds, and magic numbers."""

import os

# --- Device ---
DEVICE = int(os.getenv("DEVICE", "-1"))  # -1 = CPU

# --- Thread Pool ---
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# --- Model Identifiers ---
INTENT_MODEL = os.getenv(
    "INTENT_MODEL", "lalit-narayan/youtube-comment-intent-classifier"
)
TOXICITY_MODEL = os.getenv("TOXICITY_MODEL", "martin-ha/toxic-comment-model")
SENTIMENT_MODEL = os.getenv(
    "SENTIMENT_MODEL",
    "AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual",
)
SPAM_MODEL = os.getenv(
    "SPAM_MODEL", "valurank/distilroberta-spam-comments-detection"
)

# --- Thresholds ---
SPAM_ML_THRESHOLD = float(os.getenv("SPAM_ML_THRESHOLD", "0.75"))
SPAM_SCORE_THRESHOLD = int(os.getenv("SPAM_SCORE_THRESHOLD", "3"))
TOXICITY_THRESHOLD = float(os.getenv("TOXICITY_THRESHOLD", "0.7"))
LRU_CACHE_SIZE = int(os.getenv("LRU_CACHE_SIZE", "2048"))

# --- Paths ---
BLOCKLIST_PATH = os.getenv("BLOCKLIST_PATH", "data/blocklist.txt")

# --- Intent Label Mapping ---
INTENT_LABEL_MAP = {
    "LABEL_0": "appreciation",
    "LABEL_1": "question",
    "LABEL_2": "complaint",
}

# --- Server ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
