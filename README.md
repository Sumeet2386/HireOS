# Redrob AI Candidate Ranking System

**Team Discern** | India Runs — The Data & AI Challenge

A production-grade candidate ranking system that identifies the top 100 best-fit candidates for a Senior AI Engineer role from a pool of 100,000 profiles. The system employs a multi-stage pipeline combining dense and sparse retrieval, learned re-ranking, adversarial profile detection, and deterministic reasoning generation — all within strict compute constraints.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture Overview](#architecture-overview)
- [Pipeline Stages](#pipeline-stages)
- [Key Design Decisions](#key-design-decisions)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Docker Reproduction](#docker-reproduction)
- [Pre-computed Artifacts](#pre-computed-artifacts)
- [Precompute Pipeline](#precompute-pipeline)
- [Evaluation and Validation](#evaluation-and-validation)
- [Runtime Performance](#runtime-performance)
- [AI Tools Declaration](#ai-tools-declaration)
- [Team](#team)

---

## Problem Statement

Given a job description for a Senior AI Engineer (Founding Team) and 100,000 candidate profiles with structured data (career history, skills, certifications, education) and 23 behavioral signals from the Redrob platform, produce a ranked list of the top 100 candidates with per-candidate reasoning.

**Constraints**: The ranking step must execute in under 5 minutes, within 16 GB RAM, on CPU only, with no network access.

---

## Architecture Overview

```
                           100,000 candidates
                                  |
                    +-------------+-------------+
                    |                           |
              FAISS Dense                  BM25 Sparse
           (bge-small-en-v1.5)           (token-level)
              top 5,000                    top 500
                    |                           |
                    +-------------+-------------+
                                  |
                          Union + Dedup
                         ~5,000 candidates
                                  |
                    50-Feature Engineering
                    (structural, skill, behavioral,
                     semantic, location, honeypot)
                                  |
                    Multi-Signal Weighted Scoring
                                  |
                          top 300 candidates
                                  |
                    6-Layer Honeypot Pruning
                    (timeline, entailment, maturity,
                     overlap, keyword-stuffing, junior)
                                  |
                         ~100 clean candidates
                                  |
                    Deterministic Reasoning Generation
                                  |
                          submission.csv
```

**Single-command execution** (from repository root):
```bash
cd redrob-ranker && python rank.py --candidates ../candidates.jsonl --out ../submission.csv
```

---

## Pipeline Stages

### Stage 1 — Hybrid Recall (100K to 5K)

Two complementary retrieval strategies ensure both semantic and lexical coverage:

- **Dense retrieval**: Candidate profiles are embedded using BAAI/bge-small-en-v1.5 (384 dimensions) and indexed with FAISS IndexFlatIP. The pre-computed JD embedding queries the index to retrieve the top 5,000 candidates by cosine similarity.
- **Sparse retrieval**: BM25 token-based scoring retrieves the top 500 candidates. This captures "plain-language" engineers whose profiles lack trendy keywords but whose career history demonstrates relevant experience.
- **Fusion**: Both recall sets are merged with min-max normalized scores. The union typically yields approximately 5,000 unique candidates.

### Stage 2 — Feature Engineering and Scoring (5K to 300)

A 50-dimensional feature vector is constructed per candidate across six categories:

| Category | Count | Representative Features |
|:---|:---:|:---|
| Semantic | 2 | Cosine similarity to JD, BM25 relevance score |
| Structural | 13 | Years of experience in ideal range, title relevance score, average tenure fit, consulting company ratio, career progression consistency |
| Skill | 12 | Core AI skill count (out of 25 tracked), weighted proficiency score, skill-career entailment rate, assessment-proficiency gap |
| Behavioral | 18 | Activity recency decay, recruiter response rate, notice period penalty, GitHub activity score, profile completeness, interview completion rate |
| Location | 5 | India-based flag, Tier-1 city match, relocation willingness, work mode compatibility |
| Adversarial | 2 | Honeypot flag, flag count from detection module |

A multi-signal weighted scoring formula combines these features with tuned weights. The top 300 candidates advance to the pruning stage.

### Stage 3 — Honeypot Pruning (300 to 100)

The dataset contains approximately 80 adversarial "honeypot" candidates with subtly impossible profiles. Six deterministic detection layers identify them:

1. **Timeline impossibility** — Career start dates that precede graduation dates, or claimed tenure exceeding the time since a company was founded.
2. **Skill-text entailment failure** — "Expert" proficiency claims in skills that appear nowhere in career descriptions or project work.
3. **Technology maturity analysis** — Claimed years of experience with a technology exceeding that technology's existence (e.g., 8 years of GPT-4 experience).
4. **Career overlap detection** — Multiple concurrent full-time positions beyond what is realistically plausible.
5. **Keyword stuffer detection** — Profiles listing extensive AI/ML skills alongside exclusively non-technical job titles (e.g., "HR Manager" with "expert" in 10 AI skills).
6. **Suspicious junior profiles** — Candidates with very low years of experience but statistically improbable behavioral signals (e.g., extremely high recruiter save rates).

A candidate is flagged as a honeypot when two or more distinct detection categories trigger. This threshold balances precision (avoiding false positives among legitimate candidates) with recall (catching adversarial profiles).

### Stage 4 — Reasoning Generation (100 candidates)

Each of the final 100 candidates receives a deterministic, fact-based reasoning string constructed from their actual profile data:

- **Sentence 1** highlights positive signals: current title, years of experience, top relevant skills, company type, and engagement metrics.
- **Sentence 2** notes considerations: notice period, location relative to preferred cities, consulting-heavy background, or declining engagement signals.

All facts are extracted directly from the candidate record. No generative model is used during inference, eliminating hallucination risk entirely.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|:---|:---|:---|
| Embedding model | bge-small-en-v1.5 (384d) | Best quality-to-speed ratio for CPU-only inference; consistently ranks in the top tier for short-text retrieval on MTEB |
| Index type | FAISS IndexFlatIP | Exact inner product search; the 100K-scale dataset does not require approximate methods, and exact search is reproducible |
| Sparse retrieval | BM25 (rank-bm25) | Lightweight, no training required; captures keyword matches that dense models may under-weight |
| Scoring approach | Tuned multi-signal formula | A LightGBM LambdaMART model was trained but exhibited heavy feature dominance (79% importance on a single behavioral signal); the hand-tuned formula distributes weight more evenly across signal categories |
| Weak label generation | GPT-4o-mini (2K samples) | Cost-effective teacher model ($1.50 total) for offline knowledge distillation; stratified sampling ensures coverage across relevance tiers |
| Reasoning method | Rule-based templates | Guarantees factual accuracy by construction; each claim maps to a specific field in the candidate record |
| Honeypot threshold | 2 or more categories | Single-flag triggers risk false positives; requiring two independent signals provides high-confidence detection |

---

## Repository Structure

```
HireOS/
|-- README.md                          # This file
|-- submission_metadata.yaml           # Hackathon-required team and methodology metadata
|-- redrob-ranker/
|   |-- rank.py                        # Main inference entry point
|   |-- ranker/                        # Core ranking modules
|   |   |-- recall.py                  #   Hybrid FAISS + BM25 recall
|   |   |-- features.py               #   50-feature engineering
|   |   |-- ltr.py                     #   Scoring (weighted formula + LTR fallback)
|   |   |-- honeypot.py               #   6-layer adversarial detection
|   |   |-- reasoning.py              #   Deterministic reasoning generation
|   |   |-- validator.py              #   Output CSV format validation
|   |   |-- constants.py              #   Skill taxonomies, company lists, city mappings
|   |-- precompute/                    # Offline pre-computation pipeline (8 scripts)
|   |   |-- 01_eda.py                  #   Exploratory data analysis
|   |   |-- 02_text_synthesis.py       #   Candidate text concatenation
|   |   |-- 03_generate_embeddings.py  #   BAAI/bge-small-en-v1.5 encoding
|   |   |-- 04_build_faiss_index.py    #   FAISS IndexFlatIP construction
|   |   |-- 05_build_bm25_index.py     #   BM25 sparse index
|   |   |-- 06_extract_features.py     #   Structural feature extraction
|   |   |-- 07_generate_weak_labels.py #   GPT-4o-mini teacher scoring
|   |   |-- 08_train_ltr_model.py      #   LightGBM LambdaMART training
|   |-- artifacts/                     # Pre-computed outputs (not in git; see setup)
|   |-- download_artifacts.py          # Downloads artifacts from HuggingFace
|   |-- evaluate.py                    # Evaluation suite
|   |-- test_smoke.py                  # Smoke test
|   |-- Dockerfile                     # Sandboxed reproduction container
|   |-- requirements.txt              # Python dependencies
|   |-- final_submission.csv           # Generated submission output
|   |-- huggingface-space/             # HuggingFace Space demo application
```

---

## Quick Start

### Prerequisites

- Python 3.10 or later
- Approximately 600 MB disk space for pre-computed artifacts

### Installation

```bash
cd redrob-ranker
pip install -r requirements.txt
```

### Download Artifacts

The `artifacts/` directory (~589 MB) is hosted on a public HuggingFace model repository. No authentication is required.

```bash
python download_artifacts.py
```

### Run Ranking

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Validate Output

```bash
python -c "from ranker.validator import validate_submission; print(validate_submission('submission.csv'))"
```

---

## Docker Reproduction

The Dockerfile is designed for Stage 3 evaluation. It automatically downloads artifacts during the build phase, then runs ranking offline during execution.

```bash
# Build (downloads artifacts from HuggingFace — untimed setup step)
docker build -t redrob-ranker .

# Run (timed execution — fully offline, no network)
docker run \
  -v /path/to/candidates.jsonl:/data/candidates.jsonl \
  -v /path/to/output:/output \
  redrob-ranker
```

The `docker run` step executes `rank.py` within the sandboxed container. It requires no network access, no GPU, and completes in approximately 16 seconds.

---

## Pre-computed Artifacts

The following artifacts are generated by the precompute pipeline and consumed by `rank.py` at inference time:

| Artifact | Size | Description |
|:---|---:|:---|
| `bm25_index.pkl` | 206 MB | BM25 sparse retrieval index over 100K candidates |
| `index.faiss` | 154 MB | FAISS IndexFlatIP dense retrieval index |
| `embeddings.npy` | 154 MB | 100,000 x 384 candidate embeddings (float32) |
| `candidate_texts.parquet` | 69 MB | Synthesized text documents per candidate |
| `features.parquet` | 3.4 MB | 100,000 x 50 pre-computed feature matrix |
| `id_mapping.json` | 2.5 MB | Row index to candidate_id mapping |
| `weak_labels.json` | 388 KB | GPT-4o-mini relevance labels (2,000 samples) |
| `lgbm_ltr_model.bin` | 18 KB | Trained LightGBM LambdaMART model weights |
| `jd_embedding.npy` | 1.6 KB | JD query embedding (384 dimensions) |
| `eda_stats.json` | 3.2 KB | Dataset statistics from exploratory analysis |
| `feature_columns.json` | 1.2 KB | Ordered feature column names |
| `feature_importance.json` | 2.7 KB | LightGBM feature importance scores |

**Download source**: [huggingface.co/harshal9657/HireOS-artifacts](https://huggingface.co/harshal9657/HireOS-artifacts) (public, no authentication required)

---

## Precompute Pipeline

The 8-step precompute pipeline generates all artifacts from the raw `candidates.jsonl`. Steps 1 through 6 and step 8 are fully offline. Step 7 requires an OpenAI API key for weak label generation.

Total precompute time: approximately 45 minutes on an 8-core CPU.

```bash
export CANDIDATES=./candidates.jsonl

# Step 1: Exploratory data analysis
python precompute/01_eda.py --candidates $CANDIDATES --out artifacts

# Step 2: Text synthesis
python precompute/02_text_synthesis.py --candidates $CANDIDATES --out artifacts

# Step 3: Embedding generation (downloads bge-small-en-v1.5 on first run)
python precompute/03_generate_embeddings.py --texts artifacts/candidate_texts.parquet --out artifacts

# Step 4: FAISS index construction
python precompute/04_build_faiss_index.py --embeddings artifacts/embeddings.npy --out artifacts

# Step 5: BM25 index construction
python precompute/05_build_bm25_index.py --texts artifacts/candidate_texts.parquet --out artifacts

# Step 6: Feature extraction
python precompute/06_extract_features.py --candidates $CANDIDATES --out artifacts

# Step 7: Weak label generation (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python precompute/07_generate_weak_labels.py \
    --features artifacts/features.parquet \
    --candidates $CANDIDATES \
    --out artifacts

# Step 8: LightGBM LambdaMART training
python precompute/08_train_ltr_model.py \
    --features artifacts/features.parquet \
    --labels artifacts/weak_labels.json \
    --out artifacts
```

---

## Evaluation and Validation

### Format Validation

The built-in validator checks all requirements from the submission specification:

- Exactly 100 data rows with a header row
- Columns in the required order: `candidate_id`, `rank`, `score`, `reasoning`
- All candidate IDs match the `CAND_XXXXXXX` format
- Ranks 1 through 100, each appearing exactly once
- Scores are monotonically non-increasing
- Tied scores broken by ascending candidate ID

### Evaluation Suite

```bash
python evaluate.py
```

Computes quality metrics including score distribution analysis, reasoning quality checks, and honeypot detection statistics.

---

## Runtime Performance

Measured on the full 100,000-candidate dataset:

| Metric | Value |
|:---|:---|
| Total inference time | 15.7 seconds |
| Peak memory usage | ~3.5 GB |
| Artifact load time | 7.6 seconds |
| Candidate parsing | 6.2 seconds |
| Recall + feature extraction | 1.8 seconds |
| Scoring + pruning + output | < 0.1 seconds |
| GPU required | No |
| Network required | No |

All measurements taken with the full pipeline executing end-to-end, including file I/O, inside a clean environment using only HuggingFace-downloaded artifacts.

---

## AI Tools Declaration

| Tool | Usage Scope |
|:---|:---|
| Gemini (Antigravity IDE) | Architecture planning, code scaffolding, competitive analysis |
| GPT-4o-mini | Weak label generation during offline pre-computation only (2,000 candidates, ~$1.50 total cost) |

No AI tools or external services are invoked during the ranking inference step. All candidate data processing during inference is performed by deterministic, locally-executed code.

---

## Team

**Team Discern**

| Member | Role |
|:---|:---|
| Harshal Andhale | AI/ML Engineer and Ranking Architect |
| Sumeet Gite | Data Engineer and Backend Developer |
| Prem More | LLM, Evaluation, and DevOps Engineer |
| Soham Ingole | Backend & Documentation |

---

## Links

| Resource | URL |
|:---|:---|
| GitHub Repository | [github.com/Sumeet2386/HireOS](https://github.com/Sumeet2386/HireOS) |
| Live Demo | [huggingface.co/spaces/harshal9657/HireOS](https://huggingface.co/spaces/harshal9657/HireOS) |
| Artifacts Repository | [huggingface.co/harshal9657/HireOS-artifacts](https://huggingface.co/harshal9657/HireOS-artifacts) |

---

*Submission for the Redrob India Runs Data and AI Challenge 2026.*
