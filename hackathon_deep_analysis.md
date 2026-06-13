# 🏆 Redrob Hackathon — Deep Analysis

## 1. The Big Picture

**Challenge**: Build an AI system that ranks **100,000 candidates** against a specific job description and outputs the **top 100 best-fit candidates**, ranked from best (#1) to worst (#100).

**Organizer**: Redrob AI — a Series A AI-native talent intelligence platform.

**Deadline**: July 2, 2026 (IST)

**Key Constraint**: This is NOT a Kaggle-style "maximize metric" challenge. It's a **multi-stage hiring-pipeline simulation** where your ranking, code, reasoning, and ability to defend your work are all judged.

---

## 2. The Job Description — Deconstructed

The role is: **Senior AI Engineer — Founding Team** at Redrob AI (Pune/Noida, India, Hybrid).

### 2.1 Hard Requirements (Must-Have)
| Requirement | What It Really Means |
|---|---|
| Production embeddings-based retrieval | sentence-transformers, BGE, E5 etc. — in PRODUCTION, not just experiments |
| Vector DB / hybrid search infra | Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch — operational experience |
| Strong Python | Code quality matters, not just "knows Python" |
| Ranking evaluation frameworks | NDCG, MRR, MAP, A/B testing — must have DESIGNED these, not just heard of them |

### 2.2 Nice-to-Have
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost, neural)
- HR-tech / recruiting / marketplace experience
- Distributed systems / large-scale inference
- Open-source AI/ML contributions

### 2.3 Explicit Disqualifiers (Negative Signals)
| Disqualifier | Why It Matters for Ranking |
|---|---|
| Pure research, no production deployment | JD says "tried twice, didn't work" |
| Only recent LangChain/OpenAI experience (<12mo) | Must have pre-LLM-era ML production experience |
| No code in last 18 months (pure architect) | "This role writes code" |
| Title-chasers (1.5yr job hops) | Need 3+ year commitment |
| Framework enthusiasts (LangChain tutorials) | Need systems thinkers |
| Entire career at consulting firms (TCS, Infosys, Wipro, etc.) | Explicit bad fit |
| CV/speech/robotics only (no NLP/IR) | Would need to relearn fundamentals |
| Only proprietary systems, no external validation | Need to see how they think |

### 2.4 The "Ideal Candidate" Profile
- **6–8 years** total experience, **4–5 in applied ML/AI** at **product companies**
- Has **shipped end-to-end ranking/search/recommendation** to real users
- Strong opinions on retrieval, evaluation, LLM integration — backed by real systems
- Located in or willing to relocate to **Noida or Pune**
- **Active on platform** (clear signal of being in job market)
- Tilts toward **"shipper" over "researcher"**

### 2.5 Location & Logistics
- **Preferred**: Pune/Noida, India (Hybrid)
- **Acceptable**: Hyderabad, Mumbai, Delhi NCR
- **Notice period**: Sub-30 days ideal, can buy out up to 30 days. 30+ days = higher bar
- **No visa sponsorship** for outside India

---

## 3. Dataset Anatomy

### 3.1 Scale
- **100,000 candidates** in `candidates.jsonl` (~487 MB)
- **5 sample candidates** in `sample_candidates.json`

### 3.2 Candidate Profile Structure

Each candidate has 7 top-level sections:

```
candidate_id          → "CAND_XXXXXXX" (7-digit)
├── profile           → Name, headline, summary, location, country, YoE, title, company, industry
├── career_history[]  → Up to 10 positions: company, title, dates, duration, industry, company_size, description
├── education[]       → Up to 5: institution, degree, field, years, grade, tier (1-4 + unknown)
├── skills[]          → name, proficiency (beginner/intermediate/advanced/expert), endorsements, duration_months
├── certifications[]  → name, issuer, year
├── languages[]       → language, proficiency
└── redrob_signals    → 23 behavioral signals (see below)
```

### 3.3 The 23 Redrob Behavioral Signals

| # | Signal | Range | Interpretation |
|---|---|---|---|
| 1 | `profile_completeness_score` | 0–100 | Higher = more serious/engaged |
| 2 | `signup_date` | date | Recency of platform join |
| 3 | `last_active_date` | date | **CRITICAL** — inactive = unavailable |
| 4 | `open_to_work_flag` | bool | Are they actually looking? |
| 5 | `profile_views_received_30d` | int | Market signal — are recruiters interested? |
| 6 | `applications_submitted_30d` | int | Active job seeking behavior |
| 7 | `recruiter_response_rate` | 0.0–1.0 | **CRITICAL** — do they actually respond? |
| 8 | `avg_response_time_hours` | num | Lower = more responsive |
| 9 | `skill_assessment_scores` | dict | Platform-verified skill levels |
| 10 | `connection_count` | int | Network size |
| 11 | `endorsements_received` | int | Social validation |
| 12 | `notice_period_days` | 0–180 | **CRITICAL** — JD wants <30 days |
| 13 | `expected_salary_range_inr_lpa` | {min,max} | Compensation expectations |
| 14 | `preferred_work_mode` | enum | onsite/hybrid/remote/flexible |
| 15 | `willing_to_relocate` | bool | Important since role is Pune/Noida |
| 16 | `github_activity_score` | -1 to 100 | Technical engagement indicator |
| 17 | `search_appearance_30d` | int | How visible they are to recruiters |
| 18 | `saved_by_recruiters_30d` | int | Market validation signal |
| 19 | `interview_completion_rate` | 0.0–1.0 | Reliability/seriousness |
| 20 | `offer_acceptance_rate` | -1 to 1.0 | Historical hiring completion |
| 21 | `verified_email` | bool | Identity verification |
| 22 | `verified_phone` | bool | Identity verification |
| 23 | `linkedin_connected` | bool | External profile validation |

---

## 4. Traps & Honeypots — Critical

> [!CAUTION]
> The dataset is deliberately adversarial. Naïve keyword-matching WILL fail.

### 4.1 Keyword Stuffers
Candidates who have all the right AI keywords in their `skills[]` but whose actual career history is completely unrelated (e.g., Marketing Manager, Accountant, HR Manager with "NLP", "Fine-tuning LLMs", "Milvus" listed as skills). The sample data already shows this pattern clearly.

### 4.2 Honeypots (~80 candidates)
Subtly impossible profiles:
- 8 years of experience at a company founded 3 years ago
- "Expert" proficiency in 10 skills with 0 months duration
- Other logical impossibilities

**Honeypot rate >10% in top 100 = DISQUALIFICATION at Stage 3.**

### 4.3 "Plain-Language Tier 5s"
Strong candidates who don't use trendy keywords (won't say "RAG" or "Pinecone") but whose career history shows they built recommendation systems at product companies. A great ranking system should find these.

### 4.4 Behavioral Twins
Two candidates with similar profiles but vastly different behavioral signals — one is active and responsive, the other is ghost-mode. The active one should rank higher.

---

## 5. What the Sample Data Reveals

Looking at the 5 sample candidates:

| ID | Title | YoE | Country | Red Flags |
|---|---|---|---|---|
| CAND_0000001 | Backend Engineer | 6.9 | Canada | Summary says "data/backend hybrid" but skills list tons of unrelated AI skills (GANs, TTS, Speech Recognition). Assessment scores are LOW for claimed advanced skills. Location: Canada. |
| CAND_0000002 | Operations Manager | 12.5 | India | Career is marketing → operations. Skills include React, Kafka, Feature Engineering. No AI skills. Profile summary is generic template. |
| CAND_0000003 | Customer Support | 1.1 | USA | Only 1.1 years experience. Career: business analyst at consulting. Skills: Angular, SEO, Excel. Not open to work. |
| CAND_0000004 | Marketing Manager | 3.8 | Australia | Career: mechanical engineering → content writing → operations. Skills are scattered. Location: Australia. |
| CAND_0000005 | Accountant | 11.0 | India | Career: business analyst → HR → accounting. Has Image Classification as "advanced" skill but is an accountant. |

> [!IMPORTANT]
> **Key Observation**: The sample data is full of **mismatches between titles/career and listed skills**. These are the trap candidates the JD warns about. A good system must verify skills against career history, not just trust the skills array.

---

## 6. Evaluation Pipeline — 5 Stages

### Stage 1: Format Validation (Auto)
- Exactly 100 rows + 1 header
- Correct column names and order
- Valid candidate_ids, unique ranks 1-100
- Scores non-increasing with rank
- Tie-breaking: candidate_id ascending

### Stage 2: Scoring (Hidden Ground Truth)
**Formula**: `0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10`

- **NDCG@10 dominates** (50% weight) — your top 10 picks matter enormously
- Hidden ground truth with relevance tiers
- No live leaderboard — blind evaluation
- Tiebreakers: P@5 → P@10 → submission timestamp

### Stage 3: Code Reproduction + Honeypot Check
- Full code repo, reproduced in Docker sandbox
- **5 min, 16GB RAM, CPU only, no network**
- Honeypot rate >10% → disqualified
- Cannot reproduce → disqualified

### Stage 4: Manual Review
- Reasoning quality (6 checks: specific facts, JD connection, honest concerns, no hallucination, variation, rank consistency)
- Git history authenticity (must show real iteration, not single dump)
- Code quality review

### Stage 5: Interview (Top X)
- 30-minute video call
- Walk through architecture, defend design choices
- Must demonstrate familiarity with your own code

---

## 7. Submission Requirements

### 7.1 CSV File
```csv
candidate_id,rank,score,reasoning
CAND_XXXXXXX,1,0.987,"Specific, fact-based reasoning..."
...
CAND_XXXXXXX,100,0.412,"Honest assessment with concerns..."
```

### 7.2 Code Repository (GitHub)
- Clean README with reproduce command
- Full source code (no hidden steps)
- Pre-computed artifacts or scripts to generate them
- `requirements.txt` / `pyproject.toml`
- `submission_metadata.yaml`

### 7.3 Sandbox / Demo Link (Required)
- HuggingFace Spaces, Streamlit Cloud, Replit, Colab, Docker, or Binder
- Must run end-to-end on small sample (≤100 candidates)
- Within compute budget

### 7.4 Constraints
| Constraint | Limit |
|---|---|
| Runtime | ≤ 5 minutes wall-clock |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only — no GPU |
| Network | Offline — no API calls |
| Disk | ≤ 5 GB intermediate state |
| Submissions | Max 3 total |

---

## 8. Strategic Implications for Solution Design

### 8.1 What Will WIN
1. **Deep JD understanding** — parse the gap between what the JD says and what it means
2. **Career trajectory analysis** — not just current title, but progression and context
3. **Skill verification** — cross-reference skills against actual career descriptions
4. **Honeypot detection** — logical consistency checks
5. **Behavioral signal integration** — as a multiplicative modifier, not an additive one
6. **Specific, honest reasoning** — per-candidate, referencing actual profile facts

### 8.2 What Will LOSE
1. Keyword/embedding similarity on skills alone
2. Trusting self-reported proficiency without career verification
3. Ignoring behavioral signals
4. Template-based reasoning
5. Solutions requiring GPU or API calls at ranking time
6. Over-engineering (must run in 5 min on CPU)

### 8.3 Architecture Implications
- **Pre-computation is allowed** (embeddings, indexes) — just the final ranking step must be <5 min
- **No LLM API calls at ranking time** — but local small models (e.g., sentence-transformers) are fine if they fit in 16GB
- **Hybrid approach likely wins**: rule-based filtering + semantic matching + behavioral modifier
- Need both **precision** (don't rank bad candidates high) and **recall** (don't miss hidden gems)

---

## 9. Summary of Key Files

| File | Purpose |
|---|---|
| [job_description.docx](file:///d:/Hackathons/Data%20Challenge%20IndiaRuns/%5BPUB%5D%20India_runs_data_and_ai_challenge/%5BPUB%5D%20India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.docx) | The JD to rank candidates against |
| [submission_spec.docx](file:///d:/Hackathons/Data%20Challenge%20IndiaRuns/%5BPUB%5D%20India_runs_data_and_ai_challenge/%5BPUB%5D%20India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/submission_spec.docx) | Rules, format, evaluation stages |
| [redrob_signals_doc.docx](file:///d:/Hackathons/Data%20Challenge%20IndiaRuns/%5BPUB%5D%20India_runs_data_and_ai_challenge/%5BPUB%5D%20India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/redrob_signals_doc.docx) | Behavioral signals reference |
| [candidate_schema.json](file:///d:/Hackathons/Data%20Challenge%20IndiaRuns/%5BPUB%5D%20India_runs_data_and_ai_challenge/%5BPUB%5D%20India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidate_schema.json) | Data structure definition |
| [candidates.jsonl](file:///d:/Hackathons/Data%20Challenge%20IndiaRuns/%5BPUB%5D%20India_runs_data_and_ai_challenge/%5BPUB%5D%20India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl) | Full 100K candidate dataset (487 MB) |
| [sample_candidates.json](file:///d:/Hackathons/Data%20Challenge%20IndiaRuns/%5BPUB%5D%20India_runs_data_and_ai_challenge/%5BPUB%5D%20India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json) | 5 sample candidates |
| [sample_submission.csv](file:///d:/Hackathons/Data%20Challenge%20IndiaRuns/%5BPUB%5D%20India_runs_data_and_ai_challenge/%5BPUB%5D%20India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_submission.csv) | Format reference (NOT quality reference) |
| [validate_submission.py](file:///d:/Hackathons/Data%20Challenge%20IndiaRuns/%5BPUB%5D%20India_runs_data_and_ai_challenge/%5BPUB%5D%20India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py) | Local CSV validator |
| [submission_metadata_template.yaml](file:///d:/Hackathons/Data%20Challenge%20IndiaRuns/%5BPUB%5D%20India_runs_data_and_ai_challenge/%5BPUB%5D%20India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/submission_metadata_template.yaml) | Metadata template for repo root |

---

*Ready for your research materials — send them over and I'll integrate them into the solution design.*
