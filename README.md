# 🧠 Redrob AI Candidate Ranking System

AI-powered candidate ranking system for the **India Runs — Data & AI Challenge**.

Given 100,000 candidate profiles and a Senior AI Engineer job description, this system intelligently ranks and outputs the **top 100 best-fit candidates** using a hybrid retrieval + ML re-ranking pipeline with adversarial honeypot detection.

## Key Features

- **Hybrid Recall**: FAISS dense retrieval + BM25 sparse retrieval for comprehensive candidate coverage
- **50-Feature ML Re-ranking**: LightGBM LambdaMART trained on GPT-4o-mini weak labels
- **6-Layer Honeypot Detection**: Timeline impossibility, skill entailment, maturity analysis, and more
- **Fact-Based Reasoning**: Every candidate gets a verifiable, fact-based assessment

## Quick Start

```bash
cd redrob-ranker
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

See [redrob-ranker/README.md](redrob-ranker/README.md) for full documentation.

## Runtime Constraints Met

| Constraint | Limit | Actual |
|---|---|---|
| Runtime | ≤ 5 min | ~90s |
| Memory | ≤ 16 GB | ~3.5 GB |
| CPU only | ✅ | ✅ |
| No network | ✅ | ✅ |
