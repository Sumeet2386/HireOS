---
title: Redrob AI Candidate Ranker
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.34.2"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: AI-powered candidate ranking for Senior AI Engineer role
---

# 🧠 Redrob AI Candidate Ranking System

**Hackathon Submission**: India Runs — The Data & AI Challenge

Upload a candidate JSON/JSONL file (≤100 candidates) to get a ranked CSV output with per-candidate reasoning.

## Architecture

Hybrid retrieval + ML re-ranking pipeline with adversarial honeypot detection:

- **50+ features** across structural, skill, behavioral, and location dimensions
- **Multi-signal scoring** — additive formula with data-driven weights
- **9-layer honeypot pruning** — catches fake/impossible profiles
- **Rule-based reasoning** — fact-grounded, no hallucination risk

> This demo uses the fallback scoring path (no FAISS/BM25). The full pipeline with dense+sparse retrieval handles 100K candidates in ~90s on CPU.
