# 🧠 Redrob AI Candidate Ranking System

> **Hackathon Submission**: India Runs — The Data & AI Challenge  
> **Task**: Rank 100,000 candidates against a Senior AI Engineer job description and output the top 100 best-fit candidates.

---

## 🏗️ Architecture

A **hybrid retrieval + ML re-ranking pipeline** with adversarial honeypot detection:

```
100,000 candidates
    ↓ Stage 1: FAISS dense retrieval (bge-small-en-v1.5, top 5K)
              + BM25 sparse recall (top 500)
~5,000 candidates
    ↓ Stage 2: 50-feature engineering
              + LightGBM LambdaMART re-ranking
  300 candidates
    ↓ Stage 3: 6-layer honeypot pruning
 ~100 candidates
    ↓ Stage 4: Rule-based reasoning generation
  Final ranked CSV
```

**Runtime**: ~90 seconds | **Memory**: <4 GB | **CPU only, no network**

---

## 📁 Project Structure

```
redrob-ranker/
├── rank.py                    # 🔥 Main inference script (run this)
├── ranker/                    # Core ranking modules
│   ├── recall.py              #   FAISS + BM25 hybrid recall
│   ├── features.py            #   50-feature engineering
│   ├── ltr.py                 #   LightGBM LTR inference
│   ├── honeypot.py            #   6-layer adversarial detection
│   ├── reasoning.py           #   Hybrid rule-based reasoning
│   ├── validator.py           #   Output CSV validation
│   └── constants.py           #   Skill lists, consulting firms, city maps
│
├── precompute/                # Offline pre-computation scripts
│   ├── 01_eda.py              #   Exploratory data analysis
│   ├── 02_text_synthesis.py   #   Candidate text concatenation
│   ├── 03_generate_embeddings.py  # bge-small-en-v1.5 → 384d vectors
│   ├── 04_build_faiss_index.py    # IndexFlatIP construction
│   ├── 05_build_bm25_index.py     # BM25 sparse index
│   ├── 06_extract_features.py     # Structural features → Parquet
│   ├── 07_generate_weak_labels.py # GPT-4o-mini teacher scoring
│   └── 08_train_ltr_model.py      # LightGBM LambdaMART training
│
├── artifacts/                 # Pre-computed outputs (not in git — see setup)
├── evaluate.py                # Comprehensive evaluation suite
├── test_smoke.py              # Quick smoke test
├── Dockerfile                 # Docker build for sandbox reproduction
├── requirements.txt           # Python dependencies
├── submission_metadata.yaml   # Hackathon metadata
└── final_submission.csv       # Output: top 100 ranked candidates
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- ~600 MB of pre-computed artifacts (see below)

### Setup

```bash
cd redrob-ranker

# Install dependencies
pip install -r requirements.txt
```

### Pre-computed Artifacts

The `artifacts/` directory (~580 MB) is excluded from git due to size. You have two options:

**Option A: Regenerate from scratch** (~45 min, requires `candidates.jsonl`)

Run the 8 precompute scripts in order. Steps 1–6 and 8 are fully offline; step 7 requires an OpenAI API key.

```bash
# Set path to your candidates file
export CANDIDATES=../India_runs_data_and_ai_challenge/candidates.jsonl

# Step 1: Exploratory data analysis → artifacts/eda_stats.json
python precompute/01_eda.py --candidates $CANDIDATES --out artifacts

# Step 2: Text synthesis → artifacts/candidate_texts.parquet, artifacts/id_mapping.json
python precompute/02_text_synthesis.py --candidates $CANDIDATES --out artifacts

# Step 3: Generate embeddings → artifacts/embeddings.npy, artifacts/jd_embedding.npy
#   (Downloads BAAI/bge-small-en-v1.5 on first run, ~130 MB)
python precompute/03_generate_embeddings.py --texts artifacts/candidate_texts.parquet --out artifacts

# Step 4: Build FAISS index → artifacts/index.faiss
python precompute/04_build_faiss_index.py --embeddings artifacts/embeddings.npy --out artifacts

# Step 5: Build BM25 index → artifacts/bm25_index.pkl
python precompute/05_build_bm25_index.py --texts artifacts/candidate_texts.parquet --out artifacts

# Step 6: Extract features → artifacts/features.parquet, artifacts/feature_columns.json
python precompute/06_extract_features.py --candidates $CANDIDATES --out artifacts

# Step 7: Generate weak labels → artifacts/weak_labels.json
#   ⚠ Requires OPENAI_API_KEY environment variable (GPT-4o-mini, ~$1.50 for 2K samples)
export OPENAI_API_KEY=sk-...
python precompute/07_generate_weak_labels.py --candidates $CANDIDATES --out artifacts

# Step 8: Train LTR model → artifacts/lgbm_ltr_model.bin, artifacts/feature_importance.json
python precompute/08_train_ltr_model.py --features artifacts/features.parquet --labels artifacts/weak_labels.json --out artifacts
```

**Option B: Use pre-built artifacts** (if available)

If you have a copy of the artifacts bundle, simply extract it into `redrob-ranker/artifacts/`. The expected contents:

| File | Size | Description |
|---|---|---|
| `bm25_index.pkl` | 206 MB | BM25 sparse retrieval index |
| `index.faiss` | 154 MB | FAISS dense retrieval index |
| `embeddings.npy` | 154 MB | 100K × 384 candidate embeddings |
| `candidate_texts.parquet` | 69 MB | Synthesized candidate text documents |
| `features.parquet` | 3.4 MB | 100K × 50 pre-computed features |
| `id_mapping.json` | 2.5 MB | Row index → candidate_id mapping |
| `weak_labels.json` | 388 KB | GPT-4o-mini weak labels (2K samples) |
| `lgbm_ltr_model.bin` | 18 KB | Trained LightGBM LambdaMART model |
| `jd_embedding.npy` | 1.6 KB | JD query embedding (384-dim) |
| `eda_stats.json` | 3.2 KB | Dataset statistics |
| `feature_columns.json` | 1.2 KB | Feature column ordering |
| `feature_importance.json` | 2.7 KB | LightGBM feature importance |

### Run Ranking

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --artifacts artifacts
```

### Validate Output

```bash
python -c "from ranker.validator import validate_submission; print(validate_submission('submission.csv'))"
```

### Run Evaluation Suite

```bash
python evaluate.py
```

### Docker

```bash
docker build -t redrob-ranker .
docker run -v /path/to/data:/data -v /path/to/output:/output redrob-ranker
```

---

## 🔬 Methodology

### Stage 1: Hybrid Recall (100K → ~5K)

- **Dense retrieval**: Pre-computed candidate embeddings (BAAI/bge-small-en-v1.5, 384-dim) indexed with FAISS IndexFlatIP. Query with pre-computed JD embedding, retrieve top 5,000 by cosine similarity.
- **Sparse retrieval**: BM25 token-based scoring, top 500 candidates. Catches "plain-language" candidates that dense embeddings miss.
- **Union + score normalization**: Merge both recall sets with min-max normalized scores.

### Stage 2: Feature Engineering + LTR Re-ranking (~5K → 300)

**50 features** across 6 categories:

| Category | Features | Examples |
|---|---|---|
| Semantic | 2 | Cosine similarity, BM25 score |
| Structural | 13 | YoE ideal range, title relevance, tenure fit, consulting ratio |
| Skill | 12 | Core AI skill count, proficiency score, entailment rate, assessment gap |
| Behavioral | 18 | Activity decay, response rate, notice period multiplier, GitHub score |
| Location | 5 | India flag, Tier-1 city, relocation willingness, work mode |
| Honeypot | 2 | Is honeypot flag, flag count |

**LightGBM LambdaMART** re-ranking model trained on GPT-4o-mini weak labels (2,000 stratified candidates scored on 0-10 relevance scale).

### Stage 3: Honeypot Pruning (300 → ~100)

6-layer deterministic detection:

1. **Timeline impossibility** — career dates before graduation
2. **Skill-text entailment failure** — advanced skills absent from career descriptions
3. **Skill maturity analysis** — claimed experience exceeds technology's lifespan
4. **Heavy career overlap** — concurrent positions beyond realistic
5. **Keyword stuffer detection** — AI skills with all non-technical titles
6. **Suspicious junior profiles** — too-good-to-be-true signals for low YoE

A candidate triggers honeypot status if ≥2 distinct check categories fire.

### Stage 4: Reasoning Generation

**Rule-based template** generating 2-sentence assessments per candidate:
- Sentence 1: Positive signals (title, YoE, top skills, company type, engagement)
- Sentence 2: Considerations (notice period, location, consulting background, engagement)

All facts are directly extracted from the candidate record — no hallucination risk.

---

## 📊 Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | `bge-small-en-v1.5` (384d) | Best quality/speed tradeoff for CPU inference |
| LTR model | LightGBM LambdaMART | Gold standard for CPU-efficient learning-to-rank; directly optimizes NDCG |
| Weak labels | GPT-4o-mini (2K samples) | Cost-effective ($1.50) teacher for distillation |
| Reasoning | Rule-based templates | Guaranteed factual accuracy; no hallucination risk |
| Recall strategy | FAISS + BM25 hybrid | Dense catches semantic matches; sparse catches keyword matches |
| Honeypot threshold | ≥2 categories | Balances precision (avoid false positives) with recall |

---

## 🤖 AI Tools Used

- **Gemini (Antigravity IDE)**: Architecture planning, code scaffolding, competitive analysis
- **GPT-4o-mini**: Weak label generation during offline pre-computation only (2,000 candidates × $0.15/1M tokens ≈ $1.50)

No AI tools or network access used during inference runtime.

---

## 📋 Constraints Met

| Constraint | Limit | Actual |
|---|---|---|
| Runtime | ≤ 5 minutes | ~90 seconds |
| Memory | ≤ 16 GB RAM | ~3.5 GB |
| Compute | CPU only | ✅ |
| Network | Offline | ✅ |

---

## License

This project was created for the Redrob India Runs Data & AI Challenge 2026.
