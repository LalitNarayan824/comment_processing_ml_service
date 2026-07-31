#  ML Comment Analysis Service

A **FastAPI microservice** for analyzing user comments in real-time.  
It generates actionable signals such as **intent, sentiment, toxicity, and spam detection**, designed for moderation systems and social platforms. 

🔗 **Hugging Face Space Link:** [lalit-narayan/youtube-comment-analyzer](https://huggingface.co/spaces/lalit-narayan/youtube-comment-analyzer)

🔗 **Hugging FaceDataset Link:** [lalit-narayan/youtube-comments-intent-sentiment](https://huggingface.co/datasets/lalit-narayan/youtube-comments-intent-sentiment)

---

##  Features

- **Multilingual Support** (English + Hinglish)
- **Intent Detection** (question, appreciation, complaint)
- **Sentiment Analysis** (positive, neutral, negative)
- **Toxicity Detection** (blocklist + ML)
- **Spam Detection (Hybrid System)**
  - Rule-based + ML-based scoring
- **Optimized for CPU** (16GB systems)
- **Async + Threaded inference** for speed
- **LRU Cache** for repeated queries
- **Batch Processing Support**

---

## 📁 Project Structure

```
comment_processing_ml_service/
├── app/
│   ├── main.py              # FastAPI app + endpoints
│   ├── models.py            # Pydantic schemas (CommentInput, AnalysisResponse)
│   ├── pipeline.py          # get_heavy_predictions, model loading
│   ├── spam.py              # rule_signals, is_comment_spam
│   ├── toxicity.py          # is_toxic, Aho-Corasick init
│   └── config.py            # env vars, model names, thresholds
├── data/
│   ├── blocklist.txt
│   └── README.md            # documents dataset schema/source
├── scripts/
│   └── csv_cleaner.py
├── notebooks/
│   └── *.ipynb              # training & evaluation notebooks
├── tests/
│   ├── test_spam.py
│   ├── test_toxicity.py
│   └── test_api.py
├── .github/workflows/ci.yml
├── .gitignore
├── .env.example
├── LICENSE
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt     # pandas, requests, pytest, jupyter — dev/eval-only deps
└── README.md
```

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
- Repetitive text

### 2. ML Models

| Task | Model |
|------|-------|
| **Intent** | `lalit-narayan/youtube-comment-intent-classifier` (fine-tuned MiniLM) |
| **Sentiment** | `AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual` |
| **Toxicity** | `martin-ha/toxic-comment-model` |
| **Spam** | `valurank/distilroberta-spam-comments-detection` |

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
  "sentiment": "positive",
  "sentiment_confidence": 0.95,
  "toxicity": false,
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

### Health Check

**`GET /health`** → `{"status": "healthy", "device": "CPU"}`

---

##  Performance Optimizations

- **LRU Cache**: (2048 entries) for repeated comments
- **ThreadPoolExecutor**: (4 workers) for heavy inference
- **CPU-only execution**: (`device = -1`)
- **Async API**: FastAPI async endpoints

---

##  Benchmarks & Evaluation

### Intent Model Training

Fine-tuned `microsoft/Multilingual-MiniLM-L12-H384` on ~50K YouTube comments (4 classes: appreciation, question, complaint, spam).

| Epoch | Train Loss | Val Loss | Accuracy | F1 | Precision | Recall |
|-------|-----------|----------|----------|------|-----------|--------|
| 1 | 0.641 | 0.601 | 75.72% | 75.75% | 77.92% | 75.72% |
| 2 | 0.522 | 0.546 | 78.22% | 78.09% | 78.17% | 78.22% |
| 3 | 0.433 | 0.572 | 77.05% | 76.70% | 77.08% | 77.05% |
| 4 | 0.353 | 0.631 | 77.90% | 77.73% | 77.59% | 77.90% |
| **5** | **0.290** | **0.632** | **78.43%** | **78.33%** | **78.28%** | **78.43%** |

> Published as [`lalit-narayan/youtube-comment-intent-classifier`](https://huggingface.co/lalit-narayan/youtube-comment-intent-classifier) on Hugging Face.

### System-Level Evaluation (End-to-End API)

Evaluated on 20,071 labeled YouTube comments via the hosted API:

| Metric | Accuracy |
|--------|----------|
| **Sentiment** | 42.63% (8,557 / 20,071) |
| **Toxicity** | 77.81% (15,617 / 20,071) |
| **Spam** | 77.17% (15,489 / 20,071) |

- **Total Time:** `3411.46 seconds`
- **Average:** `~0.17 seconds` per comment

### Sentiment Model Notes

The current model is `AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual`. It is particularly strong in handling:
- Slang
- Abbreviations
- Mixed language (Hinglish)

> **Note:** Sentiment accuracy appears lower due to a design decision — when a comment is classified as spam, the system skips ML models and returns a default `"neutral"` sentiment. This inflates false mismatches in the evaluation.

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

##  Deployment

Currently hosted on Hugging Face Spaces. The app runs via:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

To deploy, push the repo (with `app/`, `data/`, `Dockerfile`, and `requirements.txt`) to a Hugging Face Space.



---

##  Future Work

- Better Hinglish fine-tuning
- GPU acceleration
- Dashboard UI
- Streaming inference

---

## ⭐ Star this repo if you like it!
