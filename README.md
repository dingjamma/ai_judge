# AI Judge — SCOTUS Verdict Backtester

Can an AI judge predict Supreme Court rulings? This project finds out.

It fetches 200 landmark SCOTUS decisions from CourtListener, builds a RAG pipeline using FAISS vector search and Claude on AWS Bedrock, renders AI verdicts, and backtests accuracy against the actual historical outcomes — with temporal integrity enforced so the model never sees future rulings.

**Live demo → [ai-judge.streamlit.app](https://aijudge-mfdpnwy66vpjbv9qrdxmxn.streamlit.app)**

---

## Results

| Metric | Value |
|--------|-------|
| Cases judged | 46 (with ground-truth verdicts) |
| Overall accuracy | **45.7%** |
| Best decade | 1930s (100%) |
| Best case type | Intellectual property (100%) |
| Worst decade | 1940s–1950s (0%) |

> Accuracy dropped from 47.8% → 45.7% after fixing temporal leakage (filtering precedents to only those available before the case's ruling year). That 2.1% gap is the model "cheating" by seeing future rulings.

---

## Architecture

```
CourtListener API
      │
      ▼
 src/ingest.py       Fetches 200 landmark SCOTUS cases, extracts opinion text
      │               and verdict via regex, saves to data/processed/
      ▼
 src/embed.py        Embeds case facts using Amazon Titan (Bedrock),
      │               builds FAISS IndexFlatL2 (dim=1536)
      ▼
 src/judge.py        RAG pipeline: embed query → FAISS top-3 (pre-ruling only)
      │               → Claude Haiku prompt → parse JSON verdict → SQLite
      ▼
 src/eval.py         Accuracy metrics by decade + case type, logs to MLflow
      │
      ▼
 app.py              Streamlit dashboard (Browse / Stats / Try It)
```

---

## Stack

- **LLM** — Claude 3.5 Haiku via AWS Bedrock (cross-region inference)
- **Embeddings** — Amazon Titan Text (`amazon.titan-embed-text-v1`, dim=1536)
- **Vector search** — FAISS `IndexFlatL2`
- **Data** — [CourtListener API v4](https://www.courtlistener.com/api/) (precedential SCOTUS opinions, ordered by citation count)
- **Storage** — SQLite
- **Experiment tracking** — MLflow
- **Dashboard** — Streamlit + Plotly

---

## Limitations & Honest Notes

- **Verdict categories are coarse.** SCOTUS decisions aren't always clean affirmed/reversed/vacated — partial affirmations, remands, and per curiam decisions get flattened. This adds label noise.
- **RAG similarity ≠ legal similarity.** Two cases may embed close due to shared keywords while having totally different constitutional logic.
- **Only 46 of 200 cases had parseable ground-truth verdicts.** The rest lacked a clean outcome in the opinion text tail.
- **Model is Claude Haiku, not Sonnet/Opus.** Newer/larger models were marked Legacy on this AWS account. Accuracy would likely improve with a stronger model.

---

## Setup

### Prerequisites
- Python 3.10+
- AWS account with Bedrock access (Claude + Titan models enabled in `us-east-1`)

### Install

```bash
git clone https://github.com/dingjamma/ai_judge.git
cd ai_judge
python -m venv venv
venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### Configure

Create a `.env` file:

```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
COURTLISTENER_BASE_URL=https://www.courtlistener.com/api/rest/v4/
COURTLISTENER_TOKEN=your_token
```

Get a free CourtListener token at [courtlistener.com](https://www.courtlistener.com/sign-in/).

### Run

```bash
# 1. Ingest 200 landmark SCOTUS cases
python src/ingest.py --limit 200

# 2. Build FAISS embeddings index
python src/embed.py

# 3. Run the AI judge
python src/judge.py --all

# 4. Evaluate accuracy
python src/eval.py

# 5. Launch dashboard
streamlit run app.py
```

---

## Project Structure

```
ai_judge/
├── app.py              Streamlit dashboard
├── requirements.txt
├── src/
│   ├── ingest.py       CourtListener → data/processed/
│   ├── embed.py        Titan embeddings → FAISS index
│   ├── judge.py        RAG + Claude → SQLite verdicts
│   ├── eval.py         Metrics + MLflow logging
│   └── db.py           SQLite helpers
└── data/
    └── cases.db        Judged cases (SQLite)
```

---

*Built with [Claude Code](https://claude.ai/code)*
