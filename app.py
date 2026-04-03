import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from transformers import pipeline
from functools import lru_cache
import uvicorn

# --- CONFIGURATION & MODEL INIT ---
app = FastAPI(title="ML Comment Service", version="1.0.0")
device = -1  # Force CPU for Hugging Face 16GB Tier

print("🚀 Initializing Multilingual Model Stack...")

# Intent: mDeBERTa-v3-base for cross-lingual performance
intent_pipe = pipeline(
    "zero-shot-classification", 
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    device=device
)

# Toxicity: Multilingual XLM-RoBERTa for social media abuse
tox_pipe = pipeline(
    "text-classification", 
    model="unitary/multilingual-toxic-xlm-roberta",
    device=device
)

# Sentiment: XLM-T trained on 200M+ social posts
sent_pipe = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    device=device
)

# --- SCHEMAS (According to Design Doc) ---

class CommentInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

class LabelScore(BaseModel):
    label: str
    score: float

class AnalysisResponse(BaseModel):
    intent: str
    intent_confidence: float
    labels: List[LabelScore]
    sentiment: str
    sentiment_confidence: float
    toxicity: float

# --- PREPROCESSING (Internal Only) ---

def preprocess_text(text: str) -> str:
    """Normalize text without returning the cleaned version to client."""
    # Trim whitespace, lowercase, and normalize spaces
    text = " ".join(text.strip().lower().split())
    return text

# --- INFERENCE CORE ---

@lru_cache(max_size=2000)
def get_model_outputs(cleaned_text: str) -> Dict:
    # 1. Intent Classification
    candidate_labels = ["question", "appreciation", "complaint", "spam"]
    intent_res = intent_pipe(cleaned_text, candidate_labels=candidate_labels)
    
    # 2. Sentiment Analysis
    sent_res = sent_pipe(cleaned_text)[0]
    
    # 3. Toxicity Detection
    tox_res = tox_pipe(cleaned_text)[0]

    return {
        "intent": intent_res["labels"][0],
        "intent_confidence": round(intent_res["scores"][0], 3),
        "labels": [
            {"label": l, "score": round(s, 3)} 
            for l, s in zip(intent_res["labels"], intent_res["scores"])
        ],
        "sentiment": sent_res["label"].lower(),
        "sentiment_confidence": round(sent_res["score"], 3),
        "toxicity": round(tox_res["score"], 3)
    }

# --- ENDPOINTS ---

@app.get("/health")
def health():
    return {"status": "ready", "hardware": "CPU-16GB"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_comment(data: CommentInput):
    # Step 1: Internal Preprocessing
    cleaned = preprocess_text(data.text)
    
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text is empty after preprocessing")

    # Step 2: Model Inference (with LRU Cache)
    results = get_model_outputs(cleaned)
    
    # Step 3: Post-processing (Schema Consistency)
    return results

@app.post("/analyze-batch", response_model=List[AnalysisResponse])
async def analyze_batch(data: List[CommentInput]):
    responses = []
    for item in data:
        cleaned = preprocess_text(item.text)
        responses.append(get_model_outputs(cleaned))
    return responses

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)