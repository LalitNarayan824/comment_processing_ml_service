"""FastAPI application and endpoints."""

import asyncio
import uvicorn
from typing import List
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.config import HOST, PORT, MAX_WORKERS
from app.models import CommentInput, AnalysisResponse
from app.spam import is_comment_spam
from app.pipeline import get_heavy_predictions

app = FastAPI(title="ML Comment Service", version="2.0.1")

# Thread pool for heavy inference
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def preprocess_text(text: str) -> str:
    """Normalize whitespace and lowercase."""
    return " ".join(text.strip().lower().split())


# --- ENDPOINTS ---
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "healthy", "device": "CPU"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_comment(data: CommentInput):
    cleaned = preprocess_text(data.text)
    spam_result = is_comment_spam(cleaned)

    if spam_result["is_spam"]:
        return {
            "intent": "complaint",
            "intent_confidence": 1.0,
            "sentiment": "neutral",
            "sentiment_confidence": 1.0,
            "toxicity": False,
            "is_spam": True,
        }

    loop = asyncio.get_event_loop()
    heavy_results = await loop.run_in_executor(
        executor, get_heavy_predictions, cleaned
    )

    return {**heavy_results, "is_spam": False}


@app.post("/analyze-batch", response_model=List[AnalysisResponse])
async def analyze_batch(data: List[CommentInput]):
    responses = []
    for item in data:
        res = await analyze_comment(item)
        responses.append(res)
    return responses


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
