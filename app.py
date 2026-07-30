import torch
import asyncio
import re
import uvicorn
import ahocorasick
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from transformers import pipeline
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import RedirectResponse

# Force CPU
device = -1
app = FastAPI(title="ML Comment Service", version="2.0.1")

# Optimization: Using a conservative worker count for 16GB CPU
executor = ThreadPoolExecutor(max_workers=4)

# --- MODELS (Updated & Verified Identifiers) ---
# Intent : fine-tuned MiniLM ( Question, Complaint , Appreciation )
# intent_model_path = "./my_intent_model" 

# intent_pipe = pipeline(
#     "text-classification", 
#     model=intent_model_path, 
#     tokenizer=intent_model_path, 
#     device=device
# )

# my intent model from HF models 
intent_pipe = pipeline("text-classification" , model="lalit-narayan/youtube-comment-intent-classifier" , device=device)

# Toxicity : martin-ha , toxic comment classifier
toxicity_model = pipeline("text-classification", model="martin-ha/toxic-comment-model" , device=device)

# Sentiment: Verified Multilingual Identifier 
sent_pipe = pipeline("sentiment-analysis",model="AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual",device=device)

# Spam: Original Tiny Model
spam_pipe = pipeline("text-classification", model="valurank/distilroberta-spam-comments-detection", device=device)

class CommentInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

class AnalysisResponse(BaseModel):
    intent: str
    intent_confidence: float
    sentiment: str
    sentiment_confidence: float
    toxicity: bool
    is_spam: bool

def preprocess_text(text: str) -> str:
    return " ".join(text.strip().lower().split())

# SPAM PROCESSING LOGIC - START -----------------------------------------------------------------------------------------------    

# --- YOUR ORIGINAL RULES  ---
def rule_signals(text: str):
    text_lower = re.sub(r"(.)\1{2,}", r"\1", text.lower())

    # LINK DETECTION
    link_pattern = r"(https?://|www\.|[a-z0-9]+\s*(\.|dot)\s*(com|in|net)|bit\s*\.?\s*ly|t\s*\.?\s*me|youtu\s*\.?\s*be)"
    has_link = bool(re.search(link_pattern, text_lower))

    # HANDLE
    handle_pattern = r"(@[A-Za-z0-9_]{3,})"
    has_handle = bool(re.search(handle_pattern, text_lower))

    # CTA
    cta_keywords = [
        "join", "contact", "dm", "dm me", "message", "msg",
        "whatsapp", "click", "come to", "check bio",
        "reach me", "telegram"
    ]
    has_cta = any(word in text_lower for word in cta_keywords)

    # PROMO
    promo_pattern = r"(s\s?[uv]\s?b|channel|video|subscribe|follow|mast|maza|op content|best teacher)"
    has_promo = bool(re.search(promo_pattern, text_lower))

    # SCAM (aggressive)
    scam_pattern = r"(earn|money|paisa|kamao|invest|profit|winner|giveaway|[\$₹£]|crypto|free|cash|sub\s*4\s*sub|leak|iphone|win|offer|limited|bonus)"
    has_scam = bool(re.search(scam_pattern, text_lower))

    # COMPLAINT
    complaint_pattern = r"(too\s(small|loud|blurry|fast|slow)|broken|not working|error|404|issue|waste|outdated)"
    has_complaint_signal = bool(re.search(complaint_pattern, text_lower))

    # ADULT
    has_adult = any(word in text_lower for word in ["18+", "xxx", "sex", "nude", "hot girl"])

    # PHONE
    phone_pattern = r"(\+?\d[\d\-\s]{8,}\d)"
    has_phone = bool(re.search(phone_pattern, text))

    # REPETITION
    repetition_pattern = r"\b(\w+)(\s+\1\b){2,}"
    has_repetition = bool(re.search(repetition_pattern, text_lower))

    return {
        "has_link": has_link,
        "has_promo": has_promo,
        "has_phone": has_phone,
        "has_scam": has_scam,
        "has_adult": has_adult,
        "has_handle": has_handle,
        "has_cta": has_cta,
        "has_complaint": has_complaint_signal,
        "has_repetition": has_repetition
    }


def is_comment_spam(text: str) -> dict:
    signals = rule_signals(text)

    res = spam_pipe(text)[0]
    spam_ml_score = res["score"] if res["label"] == "spam" else 1 - res["score"]

    # --- HARD BLOCKS (AGGRESSIVE) ---
    if signals["has_phone"] or signals["has_adult"]:
        return {"is_spam": True, "force_intent": "spam"}

    if signals["has_scam"]:
        return {"is_spam": True, "force_intent": "spam"}

    if signals.get("has_repetition"):
        return {"is_spam": True, "force_intent": "spam"}

    # --- ML OVERRIDE (LOWERED) ---
    if spam_ml_score > 0.75:
        return {"is_spam": True, "force_intent": "spam"}

    # --- SCORING ---
    score = 0

    if signals["has_link"]: score += 2
    if signals["has_handle"]: score += 2
    if signals["has_cta"]: score += 2
    if signals["has_promo"]: score += 1
    if signals["has_scam"]: score += 2

    score += int(spam_ml_score * 5)

    if score >= 3:
        return {"is_spam": True, "force_intent": "spam"}

    return {"is_spam": False}
        
    

# SPAM PROCESSING LOGIC - END --------------------------------------------------------------------------------------------------------------    


# TOXICITY PROCESSING LOGIC - START -----------------------------------------------------------------------------------------------------------------

# Build and Load the Aho-Corasick Engine
def init_moderator(filepath):
    A = ahocorasick.Automaton()
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            word = line.strip().lower()
            if word and not word.startswith("#"):
                A.add_word(word, (idx, word))
    A.make_automaton()
    return A

# Initialize once at startup
blocklist_engine = init_moderator("blocklist.txt")

def is_toxic(comment: str) -> bool:
    """
    Returns True if comment is toxic (via Blocklist or AI), False otherwise.
    """
    if not comment or len(comment.strip()) == 0:
        return False

    clean_comment = comment.lower().strip()

    # --- STEP 1: Aho-Corasick (Fast Pass) ---
    # We check for matches in the blocklist
    for end_index, (idx, matched_word) in blocklist_engine.iter(clean_comment):
        # Optional: Add boundary check here if you want to be stricter
        return True 

    # --- STEP 2: Martin-Ha (Deep Analysis) ---
    # Only runs if the blocklist found nothing
    try:
        result = toxicity_model(clean_comment)[0]
        # Martin-Ha labels are usually 'toxic' or 'non-toxic'
        if result['label'] == 'toxic' and result['score'] > 0.7:
            return True
    except Exception as e:
        print(f"Model Error: {e}")
        return False

    return False

# TOXICITY PROCESSING LOGIC - END -------------------------------------------------------------------------------------------------------------    


# --- CACHED HEAVY INFERENCE ---
@lru_cache(maxsize=2048)
def get_heavy_predictions(text: str) -> Dict:
    # 1. Direct Integer Mapping (based on your training map)
    # Pipeline usually returns these as strings: "0", "1", "2"
    id_to_label = {
        "LABEL_0": "appreciation",
        "LABEL_1": "question",
        "LABEL_2": "complaint"
    }

    # 2. Intent Inference
    intent_output = intent_pipe(text)[0]
    # --- ADD THESE PRINT LINES ---
    # print(f"--- DEBUG START ---")
    # print(f"Input Text: {text}")
    # print(f"Raw Intent Output: {intent_output}") 
    # This will show you if it's '0', 'LABEL_0', or something else
    raw_label = str(intent_output["label"]) # Ensure it's a string for dictionary lookup
    
    # 3. Toxicity 
    tox_res = is_toxic(text)
    

    # 4. Sentiment (AmaanP314)
    sent_res = sent_pipe(text)[0]

    # 5. Result Construction
    return {
        "intent": id_to_label.get(raw_label, "Unknown"),
        "intent_confidence": round(intent_output["score"], 3),
        "sentiment": sent_res["label"].lower(),
        "sentiment_confidence": round(sent_res["score"], 3),
        "toxicity": tox_res
    }

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
            "sentiment": "neutral", "sentiment_confidence": 1.0,
            "toxicity": False, "is_spam": True
        }

    # Pass the complaint signal to heavy predictions to help the AI map correctly
    loop = asyncio.get_event_loop()
    heavy_results = await loop.run_in_executor(executor, get_heavy_predictions, cleaned)
    
    

    return {**heavy_results, "is_spam": False}

@app.post("/analyze-batch", response_model=List[AnalysisResponse])
async def analyze_batch(data: List[CommentInput]):
    responses = []
    for item in data:
        res = await analyze_comment(item)
        responses.append(res)
    return responses

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
