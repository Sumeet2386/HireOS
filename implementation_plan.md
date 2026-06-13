# 🏗️ Implementation Plan: Redrob AI Candidate Ranking System

## Research Comparison & Design Decisions

Before the plan, here's why Research 2 is the foundation — it patched every critical gap from Research 1:

| Gap in Research 1 | Fixed in Research 2? | Decision |
|---|---|---|
| FAISS-only recall misses plain-language candidates | ✅ Added BM25 hybrid recall (FAISS 5K + BM25 500) | **Use hybrid** |
| Embedding model unresolved (bge-large vs MiniLM) | ✅ Settled on `bge-small-en-v1.5` (384d, fast, small) | **Use bge-small** |
| Recall window too small (2K) | ✅ Expanded to 5,000 | **Use 5K** |
| No feature vector defined | ✅ Enumerated ~30 features across 5 categories | **Use full schema** |
| Weak supervision underspecified | ✅ Added Margin MSE distillation from Teacher LLM | **Use distillation** |
| SLM hallucination risk | ✅ Hybrid: rule-based template + SLM polish + fallback | **Use hybrid reasoning** |
| No location matching logic | ✅ Added `is_tier1_india` with city list | **Use city matching** |
| Consulting detection too simplistic | ✅ Added fuzzy match + "at least one product company" bypass | **Use refined logic** |
| No skill maturity analysis | ✅ Added tech release date validation | **Use skill maturity check** |
| No title-relevance scoring | ✅ Added `current_title_relevance` | **Use semantic title match** |

> [!IMPORTANT]
> **Research 2 is the blueprint. This plan fills in remaining implementation details.**

---

## Proposed Changes

### Project Structure

#### [NEW] Project Layout
```
d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\
├── README.md                          # Setup + reproduce instructions
├── submission_metadata.yaml           # From template, filled in
├── requirements.txt                   # All Python dependencies
├── pyproject.toml                     # Project config
│
├── precompute/                        # Phase 1: Offline scripts
│   ├── 01_eda.py                      # Exploratory data analysis
│   ├── 02_text_synthesis.py           # Concatenate profile text per candidate
│   ├── 03_generate_embeddings.py      # bge-small-en-v1.5 → 384d vectors
│   ├── 04_build_faiss_index.py        # IndexFlatIP construction
│   ├── 05_build_bm25_index.py         # BM25 sparse index construction
│   ├── 06_extract_features.py         # Structural features → Parquet
│   ├── 07_generate_weak_labels.py     # GPT-4o-mini Teacher LLM scoring (~$1.50)
│   └── 08_train_ltr_model.py          # LightGBM LambdaMART training
│
├── artifacts/                         # Pre-computed outputs (committed)
│   ├── index.faiss                    # FAISS vector index (~150 MB)
│   ├── bm25_index.pkl                 # BM25 serialized index
│   ├── features.parquet               # Structural features (~200 MB)
│   ├── candidate_texts.parquet        # Synthesized text per candidate
│   ├── lgbm_ltr_model.bin             # Trained LightGBM model
│   ├── jd_embedding.npy               # Pre-computed JD embedding
│   └── qwen2.5-0.5b-instruct.Q4_K_M.gguf  # Quantized SLM model
│
├── models/                            # ONNX model for online JD embedding
│   └── bge-small-en-v1.5-onnx/       # INT8 ONNX-quantized model
│
├── rank.py                            # 🔥 Main inference script (5-min sandbox)
├── ranker/                            # Core ranking modules
│   ├── __init__.py
│   ├── recall.py                      # FAISS + BM25 hybrid recall
│   ├── features.py                    # Feature engineering (40 features)
│   ├── ltr.py                         # LightGBM LTR inference
│   ├── honeypot.py                    # Adversarial detection heuristics
│   ├── reasoning.py                   # Hybrid rule-based + SLM reasoning
│   ├── validator.py                   # Output CSV validation
│   └── constants.py                   # Skill lists, consulting firms, city maps
│
├── validate_submission.py             # Official validator (from bundle)
└── tests/                             # Unit tests
    ├── test_honeypot.py
    ├── test_features.py
    └── test_reasoning.py
```

---

### Phase 1: Offline Pre-computation

> No time constraints. GPU allowed. Network allowed.

---

#### [NEW] precompute/01_eda.py

Exploratory data analysis on the full 100K dataset:
- Distribution of `years_of_experience`, `notice_period_days`, `recruiter_response_rate`
- Calculate mean/std for each behavioral signal (needed for normalization curves)
- Count of candidates by country, industry, title
- Identify the distribution of education tiers
- Count how many candidates have assessment scores
- Output: Statistical summary JSON used to calibrate decay functions

---

#### [NEW] precompute/02_text_synthesis.py

For each candidate, create a unified text document:
```python
text = f"{profile.headline}. {profile.summary}. " + " ".join(
    f"At {job.company} as {job.title} ({job.duration_months}mo): {job.description}"
    for job in career_history
)
```
- Strip formatting noise, normalize whitespace
- Output: `candidate_texts.parquet` with columns `[candidate_id, synthesized_text]`

---

#### [NEW] precompute/03_generate_embeddings.py

- Model: `BAAI/bge-small-en-v1.5` (384-dim, ~130MB)
- GPU-accelerated batch encoding of 100K texts
- Normalize vectors to unit length for cosine similarity via inner product
- Output: `embeddings.npy` (100K × 384 float32 = ~150 MB)

---

#### [NEW] precompute/04_build_faiss_index.py

- Load `embeddings.npy`
- Build `faiss.IndexFlatIP` (exact inner-product search)
- Map: row index → `candidate_id` (store mapping in Parquet)
- Output: `index.faiss` (~150 MB)

---

#### [NEW] precompute/05_build_bm25_index.py

- Tokenize each candidate's synthesized text (simple whitespace + lowercasing)
- Build BM25 index using `rank_bm25` library or custom implementation
- Serialize to pickle
- Output: `bm25_index.pkl`

---

#### [NEW] precompute/06_extract_features.py

Extract all structural/behavioral features from raw JSONL into a columnar format for fast loading during online phase. For each candidate, compute and store:

**All ~40 features** (defined in Section 6.3 of Research 2):

```
Semantic (computed online): cosine_similarity_jd, bm25_score_jd
Structural: years_of_experience, yoe_in_ideal_range, num_career_entries,
            avg_tenure_months, is_title_chaser, current_title_relevance,
            has_product_company_exp, all_consulting_career
Skill: num_core_ai_skills, skill_proficiency_score, skill_text_entailment_rate,
       assessment_vs_proficiency_gap, has_embedding_skills, has_vector_db_skills,
       has_evaluation_skills
Behavioral: days_since_active, activity_decay_score, recruiter_response_rate,
            response_time_score, notice_period_multiplier,
            interview_completion_rate, saved_by_recruiters_30d
Location: is_india, is_tier1_india, willing_to_relocate, work_mode_compatible
Honeypot: timeline_impossible, skill_text_contradiction (bool flags)
```

- Output: `features.parquet`

---

#### [NEW] precompute/07_generate_weak_labels.py

Teacher LLM distillation using **GPT-4o-mini** (~$0.15/1M input tokens):
1. Sample ~5,000 diverse candidates (stratified by title, YoE, country, honeypot status)
2. For each, construct a compact prompt with the JD summary + candidate profile highlights
3. Ask GPT-4o-mini to rate relevance on a 0-5 scale with brief reasoning
4. Parse scores → create `(candidate_id, relevance_score)` training dataset
5. Output: `weak_labels.csv`
6. **Estimated cost**: ~5K calls × ~1500 input tokens × $0.15/1M + ~100 output tokens × $0.60/1M ≈ **$1.50 total** (well within $2-3 budget)

**Prompt template for GPT-4o-mini**:
```
Rate candidate fit for Senior AI Engineer (founding team, NLP/retrieval/ranking focus,
production embeddings+vector DBs required, 5-9 YoE ideal, product companies preferred).
Scale: 0=irrelevant/honeypot 1=tangential 2=some-fit 3=decent 4=strong 5=exceptional

Candidate: {title}, {yoe}y exp, skills: {top_skills}, companies: {companies}
Summary: {summary_truncated_to_200_chars}

JSON only: {"score":N,"reason":"..."}
```

> [!TIP]
> The prompt is deliberately compact (~300-500 tokens input) to minimize cost. We truncate the candidate profile to essential fields only — the Teacher LLM doesn't need the full JSONL record, just enough to make a relevance judgment.

---

#### [NEW] precompute/08_train_ltr_model.py

- Load `features.parquet` + `weak_labels.csv`
- Join on candidate_id
- Configure LightGBM with:
  ```python
  params = {
      'objective': 'lambdarank',
      'metric': 'ndcg',
      'ndcg_eval_at': [10, 50],
      'learning_rate': 0.05,
      'num_leaves': 31,
      'min_data_in_leaf': 10,
      'feature_fraction': 0.8,
      'verbose': -1
  }
  ```
- Train with `qid` groups (all candidates share the same JD query, so `qid` is constant; but we can create synthetic queries by perturbing the JD to improve generalization)
- **Alternative if LTR training proves unreliable**: Fall back to a hand-tuned weighted scoring formula using the same features, calibrated against the GPT-4o-mini weak labels as a validation set
- Output: `lgbm_ltr_model.bin`

---

### Phase 2: Online Inference Engine (rank.py)

> **THE critical file.** Must run in ≤5 min, ≤16GB RAM, CPU only, no network.

---

#### [NEW] rank.py

Main entry point:
```
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Orchestrates the 4-stage pipeline:
1. Load artifacts → 2. Hybrid recall → 3. Feature engineering + LTR re-rank → 4. Honeypot pruning → 5. Reasoning generation → 6. CSV output + validation

---

#### [NEW] ranker/recall.py — Hybrid Candidate Recall

**Stage 1: 100K → ~5,000 unique candidates**

```python
def hybrid_recall(jd_embedding, faiss_index, bm25_index, jd_text, k_dense=5000, k_sparse=500):
    # Dense retrieval
    dense_scores, dense_ids = faiss_index.search(jd_embedding, k_dense)
    
    # Sparse retrieval (BM25)
    bm25_scores = bm25_index.get_scores(tokenize(jd_text))
    sparse_ids = np.argsort(bm25_scores)[-k_sparse:]
    
    # Union with score normalization
    all_ids = set(dense_ids) | set(sparse_ids)
    # Normalize both score distributions to [0,1] and combine
    return merged_candidates  # ~5,000-5,300 unique
```

---

#### [NEW] ranker/features.py — Feature Engineering

**Stage 2: Generate ~40 features for recalled candidates**

Key feature formulas:

```python
# YoE ideal range (5-9 years is 1.0, outside decays)
yoe_in_ideal_range = max(0, 1.0 - 0.15 * abs(yoe - 7.0))  # peaks at 7

# Title chaser detection
avg_tenure = sum(durations) / num_jobs
is_title_chaser = avg_tenure < 18  # months

# Consulting career detection (fuzzy matching)
CONSULTING_FIRMS = {"tcs", "infosys", "wipro", "accenture", "cognizant", 
                    "capgemini", "hcl", "tech mahindra", "mindtree", "mphasis",
                    "tata consultancy", "l&t infotech", "lt infotech"}
all_consulting = all(fuzzy_match(company, CONSULTING_FIRMS) for company in companies)
has_product_exp = not all_consulting

# Notice period multiplier (logistic decay)
notice_multiplier = 1.0 / (1.0 + math.exp(0.05 * (notice_days - 45)))
# 0 days → ~1.0, 30 days → ~0.68, 60 days → ~0.32, 90 days → ~0.09

# Activity decay (exponential)
days_inactive = (current_date - last_active_date).days
activity_decay = math.exp(-0.005 * days_inactive)
# 0 days → 1.0, 30 days → 0.86, 90 days → 0.64, 180 days → 0.41

# Assessment vs proficiency gap
for skill in candidate.skills:
    if skill.name in assessment_scores:
        claimed = {"beginner": 25, "intermediate": 50, "advanced": 75, "expert": 95}
        gap = claimed[skill.proficiency] - assessment_scores[skill.name]
        if gap > 30:  # Claims advanced but scores <45
            trust_penalty += 0.1

# Skill-text entailment rate
entailed = sum(1 for s in advanced_skills if s.lower() in career_text.lower())
entailment_rate = entailed / max(1, len(advanced_skills))

# Location matching
TIER1_INDIA_CITIES = {"pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", 
                       "gurugram", "bangalore", "bengaluru", "chennai", "kolkata",
                       "new delhi", "greater noida", "navi mumbai"}
is_tier1_india = any(city in location.lower() for city in TIER1_INDIA_CITIES)
```

---

#### [NEW] ranker/honeypot.py — Adversarial Detection

**Stage 3: Top 300 → Top 100 (purge honeypots)**

Three deterministic checks:

```python
def detect_honeypot(candidate):
    flags = []
    
    # 1. Timeline Impossibility
    grad_year = max(edu.end_year for edu in candidate.education) if candidate.education else None
    if grad_year:
        for job in candidate.career_history:
            job_start_year = parse_date(job.start_date).year
            if job_start_year < grad_year and job.duration_months > 12:
                flags.append("TIMELINE_IMPOSSIBLE")
    
    # 2. Skill-Text Entailment Failure
    career_text = " ".join(job.description for job in candidate.career_history).lower()
    advanced_skills = [s for s in candidate.skills 
                       if s.proficiency in ("advanced", "expert")]
    for skill in advanced_skills:
        if not skill_mentioned_in_text(skill.name, career_text):
            flags.append(f"SKILL_NOT_ENTAILED:{skill.name}")
    
    # Flag if >50% of advanced skills are unentailed
    if len([f for f in flags if "SKILL_NOT_ENTAILED" in f]) > len(advanced_skills) * 0.5:
        flags.append("SEMANTIC_CONTRADICTION")
    
    # 3. Skill Maturity Analysis
    TECH_RELEASE_YEARS = {
        "langchain": 2022, "chatgpt": 2022, "gpt-4": 2023,
        "lora": 2021, "qlora": 2023, "llama": 2023, ...
    }
    for skill in candidate.skills:
        if skill.name.lower() in TECH_RELEASE_YEARS:
            max_possible_months = (2026 - TECH_RELEASE_YEARS[skill.name.lower()]) * 12
            if skill.duration_months > max_possible_months + 6:
                flags.append(f"MATURITY_IMPOSSIBLE:{skill.name}")
    
    return len(flags) >= 2  # Honeypot if 2+ flags triggered
```

---

#### [NEW] ranker/reasoning.py — Hybrid Reasoning Generation

**Stage 4: Top 100 → Generate per-candidate reasoning**

```python
def generate_reasoning(candidate, rank, score, jd_summary):
    # Step 1: Rule-based template (always factually correct)
    template = build_template(candidate)
    
    # Step 2: SLM polish (if time permits)
    try:
        prompt = f"""You are an AI recruiter. Rewrite this candidate assessment 
        into exactly 2 fluent sentences. Do NOT add any information not present 
        in the original. Original: {template}"""
        
        polished = slm.generate(prompt, max_tokens=80, temperature=0.3)
        
        # Step 3: Hallucination check
        if hallucination_detected(polished, candidate):
            return template  # Fallback to safe template
        return polished
    except TimeoutError:
        return template

def build_template(candidate):
    """Deterministic, fact-based template."""
    parts = []
    parts.append(f"{candidate.profile.current_title} with "
                 f"{candidate.profile.years_of_experience} years of experience")
    
    # Positive signals
    if has_relevant_skills(candidate):
        skills = get_top_relevant_skills(candidate, n=3)
        parts.append(f"demonstrating expertise in {', '.join(skills)}")
    
    if candidate.redrob_signals.recruiter_response_rate > 0.7:
        parts.append(f"strong engagement (response rate: "
                     f"{candidate.redrob_signals.recruiter_response_rate:.0%})")
    
    # Concerns
    concerns = []
    if candidate.redrob_signals.notice_period_days > 30:
        concerns.append(f"{candidate.redrob_signals.notice_period_days}-day notice period")
    if not is_tier1_india(candidate):
        concerns.append(f"located in {candidate.profile.location}")
    
    sentence1 = ". ".join(parts) + "."
    sentence2 = f"Concerns: {'; '.join(concerns)}." if concerns else "No major concerns noted."
    return f"{sentence1} {sentence2}"
```

---

#### [NEW] ranker/constants.py — Reference Data

Contains all lookup tables:
- `CORE_AI_SKILLS`: List of skills relevant to the JD (embeddings, retrieval, NLP, Python, etc.)
- `CONSULTING_FIRMS`: Set of consulting/IT services company names with variants
- `TIER1_INDIA_CITIES`: Set of Indian tier-1 city names with common variants
- `TECH_RELEASE_YEARS`: Dictionary of technology → first available year
- `SKILL_SYNONYMS`: Mapping of skill names to their synonyms for entailment checking
- `NEGATIVE_TITLE_PATTERNS`: Titles that indicate non-engineering roles (Marketing Manager, Accountant, HR Manager, etc.)

---

### Phase 3: Submission & Sandbox

---

#### [NEW] Dockerfile

```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "rank.py", "--candidates", "./candidates.jsonl", "--out", "./submission.csv"]
```

---

#### [MODIFY] submission_metadata.yaml

Fill in from template with actual values after implementation.

---

### Phase 4: Presentation Deck (PDF)

Create a 10-15 slide deck covering:
1. Problem understanding
2. Architecture diagram (two-stage pipeline)
3. Feature engineering approach
4. Honeypot detection strategy
5. Behavioral signal integration
6. Reasoning generation approach
7. Time budget breakdown
8. Results/sample outputs

---

## Resolved Design Decisions

| Question | Decision |
|---|---|
| **LTR Training Budget** | Use **GPT-4o-mini** for weak labeling. ~5,000 calls ≈ **$1.50** (within $2-3 budget) |
| **Reasoning Strategy** | **Hybrid**: rule-based template + Qwen2.5-0.5B SLM polish + hallucination fallback. Best quality with safety net. |
| **Embedding Model** | **`BAAI/bge-small-en-v1.5`** (384d, 33M params). Sweet spot of quality, speed, and FAISS index size (~150MB). |
| **Git Strategy** | **Fresh GitHub repo** with incremental commits to show authentic iteration history. |

---

## Verification Plan

### Automated Tests
```bash
# Unit tests for each module
python -m pytest tests/ -v

# Run on sample_candidates.json (5 candidates) — fast sanity check
python rank.py --candidates ./sample_candidates.json --out ./test_submission.csv

# Validate submission format
python validate_submission.py ./test_submission.csv

# Full run on complete dataset — must finish in <5 min
time python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Validate final submission
python validate_submission.py ./submission.csv
```

### Manual Verification
- Review top-10 reasoning strings for factual accuracy
- Spot-check that no sample honeypots (CAND_0000001, CAND_0000002) appear in output
- Verify scores are monotonically non-increasing
- Verify all candidate_ids exist in the source dataset
- Test the sandbox demo link (HuggingFace Space or Docker) with a 100-candidate subset
