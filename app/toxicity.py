"""Toxicity detection: Aho-Corasick blocklist + ML model."""

import ahocorasick
from transformers import pipeline

from app.config import DEVICE, TOXICITY_MODEL, TOXICITY_THRESHOLD, BLOCKLIST_PATH

# --- ML Toxicity Model ---
toxicity_model = pipeline("text-classification", model=TOXICITY_MODEL, device=DEVICE)


def _init_moderator(filepath: str) -> ahocorasick.Automaton:
    """Build and return an Aho-Corasick automaton from a blocklist file."""
    automaton = ahocorasick.Automaton()
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            word = line.strip().lower()
            if word and not word.startswith("#"):
                automaton.add_word(word, (idx, word))
    automaton.make_automaton()
    return automaton


# Initialize once at import time
blocklist_engine = _init_moderator(BLOCKLIST_PATH)


def is_toxic(comment: str) -> bool:
    """Return True if comment is toxic (via blocklist or ML model)."""
    if not comment or len(comment.strip()) == 0:
        return False

    clean_comment = comment.lower().strip()

    # --- STEP 1: Aho-Corasick (Fast Pass) ---
    for _end_index, (_idx, _matched_word) in blocklist_engine.iter(clean_comment):
        return True

    # --- STEP 2: ML Model (Deep Analysis) ---
    try:
        result = toxicity_model(clean_comment)[0]
        if result["label"] == "toxic" and result["score"] > TOXICITY_THRESHOLD:
            return True
    except Exception as e:
        print(f"Toxicity model error: {e}")
        return False

    return False
