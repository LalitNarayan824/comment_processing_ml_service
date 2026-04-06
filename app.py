import torch
import asyncio
import re
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from transformers import pipeline
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Force CPU
device = -1
app = FastAPI(title="ML Comment Service", version="1.0.1")

# Optimization: Using a conservative worker count for 16GB CPU
executor = ThreadPoolExecutor(max_workers=4)

# --- MODELS (Updated & Verified Identifiers) ---
# Intent: Multilingual MiniLM (Fast + Hinglish Support)
intent_pipe = pipeline("zero-shot-classification", model="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli", device=device)

# Toxicity: Original Multilingual XLM-RoBERTa
tox_pipe = pipeline("text-classification", model="unitary/multilingual-toxic-xlm-roberta", device=device)

# Sentiment: Verified Multilingual Identifier 
sent_pipe = pipeline("sentiment-analysis",model="finiteautomata/bertweet-base-sentiment-analysis",device=device)

# Spam: Original Tiny Model
spam_pipe = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection", device=device)

class CommentInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

class AnalysisResponse(BaseModel):
    intent: str
    intent_confidence: float
    labels: List[dict]
    sentiment: str
    sentiment_confidence: float
    toxicity: float
    is_spam: bool

def preprocess_text(text: str) -> str:
    return " ".join(text.strip().lower().split())

# --- YOUR ORIGINAL RULES  ---
def rule_signals(text: str):
    text_lower = text.lower()
    
    # LINK DETECTION (Catches obfuscated dots)
    link_pattern = r"(https?://|www\.|\.com|\.in|bit\s?\.?\s?ly|t\s?\.?\s?me|youtu\s?\.?\s?be)"
    has_link = bool(re.search(link_pattern, text_lower))

    # HANDLE/CTA DETECTION
    handle_pattern = r"(@[A-Za-z0-9_]{3,})"
    has_handle = bool(re.search(handle_pattern, text_lower))
    
    # Added 'come to', 'check bio', 'dm'
    cta_keywords = ["join", "contact", "dm me", "message me", "whatsapp", "click", "come to", "check bio"]
    has_cta = any(word in text_lower for word in cta_keywords)

    # PROMO & HINGLISH PRAISE (Added 'mast', 'maza', 'op')
    promo_pattern = r"(s\s?[u|v]\s?b|channel|video|subscribe|follow|mast|maza|op content|best teacher)"
    has_promo = bool(re.search(promo_pattern, text_lower))

    # SCAM & EXAM LEAKS (Added 'sub 4 sub', 'leak', 'iphone')
    scam_pattern = r"(earn|money|paisa|kamao|invest|profit|winner|giveaway|[\$₹£]|crypto|free|cash|sub 4 sub|leak|iphone|win)"
    has_scam = bool(re.search(scam_pattern, text_lower))

    # COMPLAINT PATTERNS (To help the AI with the 0% category)
    complaint_pattern = r"(too\s(small|loud|blurry|fast|slow)|broken|not working|error|404|issue|waste|outdated)"
    has_complaint_signal = bool(re.search(complaint_pattern, text_lower))

    # ADULT & PHONE
    has_adult = any(word in text_lower for word in ["18+", "xxx", "sex", "nude", "hot girl"])
    phone_pattern = r"(\+?\d[\d\-\s]{7,}\d)"
    has_phone = bool(re.search(phone_pattern, text))

    return {
        "has_link": has_link, "has_promo": has_promo, "has_phone": has_phone, 
        "has_scam": has_scam, "has_adult": has_adult, "has_handle": has_handle,
        "has_cta": has_cta, "has_complaint": has_complaint_signal
    }

def get_spam_score(text: str) -> float:
    result = spam_pipe(text)[0]
    return result["score"] if result["label"] == "spam" else 1 - result["score"]

def is_comment_spam(text: str) -> dict:
    signals = rule_signals(text)
    
    # Get BERT-Tiny score
    res = spam_pipe(text)[0]
    spam_ml_score = res["score"] if res["label"] == "spam" else 1 - res["score"]

    # --- AUTO-BLOCKS ---
    if signals["has_phone"] or signals["has_adult"]:
        return {"is_spam": True, "force_intent": "spam"}
    
    if (signals["has_link"] or signals["has_handle"]) and (signals["has_scam"] or signals["has_promo"]):
        return {"is_spam": True, "force_intent": "spam"}

    if signals["has_scam"] and signals["has_cta"]:
        return {"is_spam": True, "force_intent": "spam"}

    # --- SCORING ---
    score = 0
    if signals["has_link"]: score += 2
    if signals["has_handle"]: score += 1
    if signals["has_cta"]: score += 1
    if signals["has_promo"]: score += 1
    if signals["has_scam"]: score += 2
    if spam_ml_score > 0.6: score += 2
    
    if score >= 3:
        return {"is_spam": True, "force_intent": "spam"}
        
    # --- COMPLAINT OVERRIDE ---
    # If it's not spam, but has strong complaint signals, we let the AI know
    return {"is_spam": False, "has_complaint": signals["has_complaint"]}

# --- CACHED HEAVY INFERENCE ---
@lru_cache(maxsize=2048)
def get_heavy_predictions(text: str) -> Dict:
    # 1. Define "Magnet" labels for the AI and their professional "Display" mappings
    # 'thanks' is a much stronger magnet for Hinglish/Slang than 'appreciation'
    # 'issue' or 'problem' catches technical errors better than 'complaint'
    mapping = {
        "question": "question",
        "thanks": "appreciation",
        "issue": "complaint",
        "promotion": "spam"
    }
    magnet_labels = list(mapping.keys())

    # 2. Run Inference with Magnets
    intent_res = intent_pipe(text, candidate_labels=magnet_labels)
    
    # 3. Toxicity (XLM-R)
    tox_res = tox_pipe(text)[0]
    
    # 4. Sentiment (XLM-R)
    sent_res = sent_pipe(text)[0]

    # 5. Map the internal AI label back to your professional category
    raw_ai_label = intent_res["labels"][0]
    final_intent = mapping.get(raw_ai_label, raw_ai_label)

    return {
        "intent": final_intent,
        "intent_confidence": round(intent_res["scores"][0], 3),
        # This also remaps the full list of labels so your logs stay professional
        "labels": [
            {"label": mapping.get(l, l), "score": round(s, 3)} 
            for l, s in zip(intent_res["labels"], intent_res["scores"])
        ],
        "sentiment": sent_res["label"].lower(),
        "sentiment_confidence": round(sent_res["score"], 3),
        "toxicity": round(tox_res["score"], 3)
    }

# --- ENDPOINTS ---
@app.get("/")
def root():
    return {"service": "ML Comment Analyzer", "status": "online"}

@app.get("/health")
def health():
    return {"status": "healthy", "device": "CPU"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_comment(data: CommentInput):
    cleaned = preprocess_text(data.text)
    spam_result = is_comment_spam(cleaned)

    if spam_result["is_spam"]:
        return {
            "intent": "spam",
            "intent_confidence": 1.0,
            "labels": [{"label": "spam", "score": 1.0}],
            "sentiment": "neutral", "sentiment_confidence": 1.0,
            "toxicity": 0.0, "is_spam": True
        }

    # Pass the complaint signal to heavy predictions to help the AI map correctly
    loop = asyncio.get_event_loop()
    heavy_results = await loop.run_in_executor(executor, get_heavy_predictions, cleaned)
    
    # Manual Override: If rules say it's a complaint but AI is unsure
    if spam_result.get("has_complaint") and heavy_results["intent"] == "question":
        heavy_results["intent"] = "complaint"

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



# ============================================================
# 📊 FINAL REPORT

# 🟢 Sentiment Accuracy : 42.63% (8557/20071) this was the result for the previous model ->cardiffnlp/twitter-xlm-roberta-base-sentiment
# we are not benchmarking against the original model, but this is a good reference point to see the improvement with the new model (finiteautomata/bertweet-base-sentiment-analysis) which is more specialized for sentiment analysis and should perform better on social media text. we are not saying that the model was any bad but for our use case and the kind of comments we receive, the new model is expected to capture sentiments more accurately, especially in the context of slang, abbreviations, and mixed language (hinglish) which are common in our comments.
# 🔴 Toxicity Accuracy  : 77.81% (15617/20071)
# 🟡 Spam Accuracy      : 77.17% (15489/20071)

# ⏱️ Time: 3411.46s (0.17s per comment)
# ------------------------------------------------------------

# 🔍 SENTIMENT CONFUSION:
# positive → neutral: 3930
# neutral → negative: 1758
# positive → negative: 830
# negative → neutral: 474
# neutral → positive: 453

# 🔍 TOXICITY CONFUSION:
# 1 → 0: 321
# 0 → 1: 153

# 🔍 SPAM CONFUSION:
# 1 → 0: 375
# 0 → 1: 227
# ============================================================

# here the sentiment accuracy is weak because in our pipeline when we are absolutely sure that a particular comment is spam we output this -> if spam_result["is_spam"]: return { "intent": "spam", "intent_confidence": 1.0, "labels": [{"label": "spam", "score": 1.0}], "sentiment": "neutral", "sentiment_confidence": 1.0, "toxicity": 0.0, "is_spam": True }

# and we dont touch the models for any analysis for that comment , so because of this we can see that sentiment converges towards neutral in most cases and , toxicity = 0 becomes correct in most cases because most spams are non toxic in nature

# earlier we were using ->cardiffnlp/twitter-xlm-roberta-base-sentiment

# 📊 RAW MODEL REPORT Accuracy: 29.85% (2029/6797) Time: 16.95s (0.0025s per comment)

# this is for finiteautomata/bertweet-base-sentiment-analysis ,

# 📊 RAW MODEL REPORT Accuracy: 76.93% (5229/6797) Time: 16.44s (0.0024s per comment)