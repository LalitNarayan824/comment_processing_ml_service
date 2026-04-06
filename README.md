#  ML Comment Analysis Service

A **FastAPI microservice** for analyzing user comments in real-time.  
It generates actionable signals such as **intent, sentiment, toxicity, and spam detection**, designed for moderation systems and social platforms. 

---

##  Features

- **Multilingual Support** (English + Hinglish)
- **Intent Detection** (question, appreciation, complaint, spam)
- **Sentiment Analysis** (positive, neutral, negative)
- **Toxicity Detection**
- **Spam Detection (Hybrid System)**
  - Rule-based + ML-based scoring
- **Optimized for CPU** (16GB systems)
- **Async + Threaded inference** for speed
- **LRU Cache** for repeated queries
- **Batch Processing Support**

---

##  System Architecture

```text
              +------------------+
              |   Input Comment  |
              +--------+---------+
                       |
                       v
            +----------------------+
            |   Preprocessing      |
            +----------------------+
                       |
             +---------+---------+
             |                   |
             v                   v
    +----------------+   +----------------------+
    | Rule Engine    |   | ML Models            |
    | (Spam Signals) |   | Intent, Sentiment,   |
    |                |   | Toxicity             |
    +--------+-------+   +----------+-----------+
             |                      |
             +----------+-----------+
                        v
              +----------------------+
              |  Final Aggregation   |
              +----------------------+
                        |
                        v
              +----------------------+
              | API Response (JSON)  |
              +----------------------+
```

---

##  Architecture Overview

The pipeline combines:

### 1. Rule-Based System
Detects:
- Links (even obfuscated)
- Handles (`@username`)
- Call-to-actions (DM, join, etc.)
- Scam keywords (earn money, crypto, etc.)
- Adult content
- Phone numbers

### 2. ML Models

| Task | Model |
|------|-------|
| **Intent** | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| **Sentiment** | `finiteautomata/bertweet-base-sentiment-analysis` |
| **Toxicity** | `unitary/multilingual-toxic-xlm-roberta` |
| **Spam** | `mrm8488/bert-tiny-finetuned-sms-spam-detection` |

---

##  API Endpoints

### Analyze Single Comment

**`POST /analyze`**

**Request:**
```json
{
  "text": "This video is amazing bro!"
}
```

**Response:**
```json
{
  "intent": "appreciation",
  "intent_confidence": 0.91,
  "labels": [
    {"label": "appreciation", "score": 0.91}
  ],
  "sentiment": "positive",
  "sentiment_confidence": 0.95,
  "toxicity": 0.01,
  "is_spam": false
}
```

### Batch Analysis

**`POST /analyze-batch`**

**Request:**
```json
[
  {"text": "Nice video!"},
  {"text": "Earn money fast click here"}
]
```

---

##  Performance Optimizations

- **LRU Cache**: (2048 entries) for repeated comments
- **ThreadPoolExecutor**: (4 workers) for heavy inference
- **CPU-only execution**: (`device = -1`)
- **Async API**: FastAPI async endpoints

---

##  Final Report

### Model Performance

#### Sentiment Accuracy
**42.63% (8557 / 20071)**

> **Note:** The previous model used was `cardiffnlp/twitter-xlm-roberta-base-sentiment`. We are not strictly benchmarking against the original model, but it serves as a useful reference point. The newer model — `finiteautomata/bertweet-base-sentiment-analysis` — is more specialized for social media text and is expected to perform better.

It is particularly stronger in handling:
- Slang
- Abbreviations
- Mixed language (Hinglish)

**Comparisons:**
- **Previous Model** (`cardiffnlp/twitter-xlm-roberta-base-sentiment`):  
  RAW MODEL REPORT Accuracy: `29.85% (2029/6797)` | Time: `16.95s` (0.0025s per comment)
- **Current Model** (`finiteautomata/bertweet-base-sentiment-analysis`):  
  RAW MODEL REPORT Accuracy: `76.93% (5229/6797)` | Time: `16.44s` (0.0024s per comment)

#### Toxicity Accuracy
**77.81% (15617 / 20071)**

#### Spam Accuracy
**77.17% (15489 / 20071)**

---

### Overall System Performance

- **Total Time:** `3411.46 seconds`
- **Average:** `~0.17 seconds` per comment

---

###  Confusion Analysis

#### Sentiment Confusion
- **positive → neutral:** 3930
- **neutral → negative:** 1758
- **positive → negative:** 830
- **negative → neutral:** 474
- **neutral → positive:** 453

#### Toxicity Confusion
- **toxic (1) → non-toxic (0):** 321
- **non-toxic (0) → toxic (1):** 153

#### Spam Confusion
- **spam (1) → not spam (0):** 375
- **not spam (0) → spam (1):** 227

---

##  Key Observation (Sentiment Limitation)

Sentiment accuracy appears lower due to a **design decision in the pipeline**:

When a comment is confidently classified as spam, the system skips all ML models and directly returns:

```python
if spam_result["is_spam"]:
    return {
        "intent": "spam",
        "intent_confidence": 1.0,
        "labels": [{"label": "spam", "score": 1.0}],
        "sentiment": "neutral",
        "sentiment_confidence": 1.0,
        "toxicity": 0.0,
        "is_spam": True
    }
```

> **Note:** For now we didn't have any data to test the accuracy of intent. We are currently building a suitable dataset for that, we apologize for that.

---

##  Deployment

We have currently hosted this at Hugging Face. You can too just make a space and add the `app.py`, `Dockerfile`, and `requirements.txt`.

🔗 **Hugging Face Link:** [lalit-narayan/youtube-comment-analyzer](https://huggingface.co/spaces/lalit-narayan/youtube-comment-analyzer)

---

##  Future Work

- Better Hinglish fine-tuning
- GPU acceleration
- Dashboard UI
- Streaming inference

---

## ⭐ Star this repo if you like it!
