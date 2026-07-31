"""Spam detection: rule-based signals + ML scoring."""

import re

from transformers import pipeline

from app.config import DEVICE, SPAM_MODEL, SPAM_ML_THRESHOLD, SPAM_SCORE_THRESHOLD

# --- ML Spam Model ---
spam_pipe = pipeline("text-classification", model=SPAM_MODEL, device=DEVICE)


def rule_signals(text: str) -> dict:
    """Extract rule-based spam signals from text."""
    text_lower = re.sub(r"(.)\1{2,}", r"\1", text.lower())

    # LINK DETECTION
    link_pattern = (
        r"(https?://|www\.|[a-z0-9]+\s*(\.|dot)\s*(com|in|net)"
        r"|bit\s*\.?\s*ly|t\s*\.?\s*me|youtu\s*\.?\s*be)"
    )
    has_link = bool(re.search(link_pattern, text_lower))

    # HANDLE
    handle_pattern = r"(@[A-Za-z0-9_]{3,})"
    has_handle = bool(re.search(handle_pattern, text_lower))

    # CTA
    cta_keywords = [
        "join", "contact", "dm", "dm me", "message", "msg",
        "whatsapp", "click", "come to", "check bio",
        "reach me", "telegram",
    ]
    has_cta = any(word in text_lower for word in cta_keywords)

    # PROMO
    promo_pattern = (
        r"(s\s?[uv]\s?b|channel|video|subscribe|follow"
        r"|mast|maza|op content|best teacher)"
    )
    has_promo = bool(re.search(promo_pattern, text_lower))

    # SCAM (aggressive)
    scam_pattern = (
        r"(earn|money|paisa|kamao|invest|profit|winner|giveaway"
        r"|[\$₹£]|crypto|free|cash|sub\s*4\s*sub|leak|iphone|win|offer|limited|bonus)"
    )
    has_scam = bool(re.search(scam_pattern, text_lower))

    # COMPLAINT
    complaint_pattern = (
        r"(too\s(small|loud|blurry|fast|slow)"
        r"|broken|not working|error|404|issue|waste|outdated)"
    )
    has_complaint_signal = bool(re.search(complaint_pattern, text_lower))

    # ADULT
    has_adult = any(
        word in text_lower for word in ["18+", "xxx", "sex", "nude", "hot girl"]
    )

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
        "has_repetition": has_repetition,
    }


def is_comment_spam(text: str) -> dict:
    """Determine if a comment is spam using rules + ML."""
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

    # --- ML OVERRIDE ---
    if spam_ml_score > SPAM_ML_THRESHOLD:
        return {"is_spam": True, "force_intent": "spam"}

    # --- SCORING ---
    score = 0
    if signals["has_link"]:
        score += 2
    if signals["has_handle"]:
        score += 2
    if signals["has_cta"]:
        score += 2
    if signals["has_promo"]:
        score += 1
    if signals["has_scam"]:
        score += 2

    score += int(spam_ml_score * 5)

    if score >= SPAM_SCORE_THRESHOLD:
        return {"is_spam": True, "force_intent": "spam"}

    return {"is_spam": False}
