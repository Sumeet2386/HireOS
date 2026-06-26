# 📚 HACKATHON COMPLETE REFERENCE
# Redrob India Runs — The Data & AI Challenge
# =============================================================================
# This file consolidates ALL official hackathon documents into a single
# searchable reference. Created from the files in:
#   [PUB] India_runs_data_and_ai_challenge/
#
# Contents:
#   1. README (Participant Bundle Overview)
#   2. Job Description (Full Text)
#   3. Submission Specification (Full Rules)
#   4. Redrob Behavioral Signals Reference
#   5. Candidate Schema (JSON Schema)
#   6. Submission Metadata Template
#   7. Official Validator Script
#   8. Sample Submission Format
#   9. Sample Candidates (Schema Examples)
#  10. Dataset Info (candidates.jsonl)
# =============================================================================

---

# 1. README — Participant Bundle Overview

**Source**: `README.docx`

## What's in the bundle

| File | What it is |
|---|---|
| `candidates.jsonl.gz` | The 100,000-candidate pool you'll rank. Gzipped JSONL (~52 MB compressed, ~465 MB uncompressed). |
| `sample_candidates.json` | First 50 candidates as pretty-printed JSON. Use to inspect the schema quickly. |
| `job_description.md` | The job description you're ranking candidates against. Read carefully — including the section at the end specifically for hackathon participants. |
| `submission_spec.md` | Read this in full before starting. Submission format, rules, compute constraints, evaluation stages. |
| `submission_metadata_template.yaml` | Template for the metadata you'll provide alongside your submission. |
| `candidate_schema.json` | JSON Schema describing every field in a candidate record. |
| `redrob_signals_doc.md` | Reference for the 23 behavioral signals in each candidate's `redrob_signals` object. |
| `sample_submission.csv` | A format reference. Not a high-quality ranking — just an example of the CSV structure your submission should match. |
| `validate_submission.py` | Format validator. Run this on your submission before uploading. |

## Getting started

**1. Read the docs (~30 minutes)** in this order:
1. `job_description.md` — understand what role you're ranking candidates for
2. `submission_spec.md` — understand the rules and evaluation pipeline
3. `redrob_signals_doc.md` — understand the trap candidates and signal envelopes
4. `candidate_schema.json` — understand the candidate data structure
5. Open `sample_candidates.json` and skim a few candidates to see what real data looks like

**2. Unpack the candidate pool**
```bash
gunzip -k candidates.jsonl.gz   # -k keeps the .gz; you get both files
wc -l candidates.jsonl           # should print 100000
```
Or load the gzipped file directly in Python:
```python
import gzip, json
with gzip.open("candidates.jsonl.gz", "rt") as f:
    candidates = [json.loads(line) for line in f if line.strip()]
print(len(candidates))  # 100000
```

**3. Build your ranker**
Your job: produce a CSV with the top 100 candidates for the JD, ranked best-fit first, with a 1-2 sentence reasoning for each.

The format is described in `submission_spec.md` Section 2-3. The compute constraints are in Section 3 (5 min, 16 GB, CPU only, no network during ranking).

**4. Validate before submitting**
```bash
python validate_submission.py your_submission.csv
```

**5. Submit**
Submit via the portal. You'll be asked for:
- The CSV file
- All the metadata from `submission_metadata_template.yaml` (team name, GitHub repo, sandbox link, AI tools declaration, etc.)
- See `submission_spec.md` Section 10 for the full list

**Sandbox link is required** — a working hosted environment (HuggingFace Spaces, Streamlit Cloud, Replit, Colab, Docker, or Binder) where your ranker can be run on a small sample. See Section 10.5 for what counts as a valid sandbox.

## Key things to know

- **No live leaderboard.** Scores are revealed only after submissions close. There is no feedback during the competition.
- **Three submissions max.** Your last valid submission counts.
- **AI tools are allowed.** Declare them honestly. The evaluation is designed so that AI-assisted work where you did real engineering succeeds, while AI-only submissions fail at Stages 3-5.
- **The dataset contains traps.** Keyword stuffers, plain-language Tier 5s, behavioral twins, and **~80 honeypots** with subtly impossible profiles. Submissions with honeypot rate > 10% in top 100 are disqualified. See `redrob_signals_doc.md`.
- **You will be interviewed if you reach the top X.** Be prepared to walk through your architecture and defend your design choices.

---

# 2. Job Description — Senior AI Engineer (Founding Team)

**Source**: `job_description.docx`

## Header
- **Title**: Senior AI Engineer — Founding Team
- **Company**: Redrob AI (Series A AI-native talent intelligence platform)
- **Location**: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
- **Employment Type**: Full-time
- **Experience Required**: 5–9 years (see "what we mean by this" below)

## About the Role

> "We're going to write this JD differently from most. We're a Series A company that just raised our round and we're building a new AI Engineering org from scratch."

They need someone who is simultaneously comfortable with:
1. **Deep technical depth** in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning
2. **Scrappy product-engineering attitude** — willing to ship a working ranker in a week even if the underlying ML is "obviously suboptimal"

> "We'd rather you tilt slightly toward shipper than toward researcher."

## What you'd actually be doing

Own the intelligence layer of Redrob's product: ranking, retrieval, and matching systems.

**First 90 days:**
- Weeks 1-3: Audit current system (mostly BM25 + rule-based scoring). Identify the 3-4 highest-leverage things to fix.
- Weeks 4-8: Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics. Will involve embeddings, hybrid retrieval, probably some LLM-based re-ranking.
- Weeks 9-12: Set up evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops.

## What "5-9 years" means

> "Some people hit 'senior engineer' judgment at 4 years; some never hit it after 15."

### Disqualifiers (actually enforced):
1. **Pure research without production deployment** — "We've tried it twice and it didn't work for either side."
2. **AI experience is primarily recent (<12 months) LangChain/OpenAI projects** — Unless substantial pre-LLM-era ML production experience exists. "We're looking for people who understood retrieval and ranking before it became fashionable."
3. **Haven't written production code in last 18 months** — "This role writes code."

## Skills Inventory

### Things you ABSOLUTELY need:
1. **Production experience with embeddings-based retrieval** (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users
2. **Production experience with vector databases or hybrid search** (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS)
3. **Strong Python** — they care about code quality
4. **Hands-on experience designing evaluation frameworks for ranking systems** — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation

### Nice to have:
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost-based or neural)
- HR-tech / recruiting tech / marketplace products experience
- Distributed systems / large-scale inference optimization
- Open-source contributions in AI/ML

### Explicitly DO NOT want:
1. **Title-chasers** — "Senior" → "Staff" → "Principal" switching every 1.5 years
2. **Framework enthusiasts** — LangChain tutorials, blog posts about hot frameworks
3. **Only consulting firms career** (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) — "We've had bad fit experiences." Prior product-company experience with current consulting is fine
4. **Primary expertise in CV/Speech/Robotics** without NLP/IR exposure
5. **Entirely closed-source proprietary work** for 5+ years without external validation

## Location, Comp, Logistics
- **Location**: Pune/Noida-preferred but flexible. Offices in Noida and Pune (mostly Tue/Thu). Quarterly travel for offsites. Hyderabad, Pune, Mumbai, Delhi NCR welcome. Outside India: case-by-case, no visa sponsorship.
- **Notice period**: Sub-30-day preferred. Can buy out up to 30 days. 30+ day candidates still in scope but higher bar.

## The "Ideal Candidate" (reading between the lines)
- 6-8 years total experience, 4-5 in applied ML/AI roles at **product companies** (not pure services)
- Has shipped at least one end-to-end ranking/search/recommendation system to real users at meaningful scale
- Strong opinions about retrieval (hybrid vs dense), evaluation (offline vs online), LLM integration (when to fine-tune vs prompt) — can defend with real systems built
- Located in or willing to relocate to Noida or Pune
- Active on Redrob platform (or has clear signal of being in the job market)

> "We're explicitly OK with finding only 10 great matches in a 100K candidate pool."

## ⚠️ CRITICAL: Note for Hackathon Participants

> "The 'right answer' to this JD is NOT 'find candidates whose skills section contains the most AI keywords.' **That's a trap we've explicitly built into the dataset.**"

The right answer involves reasoning about the **gap between what the JD says and what the JD means**:
- A Tier 5 candidate may not use "RAG" or "Pinecone" but if their career history shows they built a recommendation system at a product company, they're a fit
- A candidate with all AI keywords but title "Marketing Manager" is NOT a fit, no matter how perfect their skill list looks

**Your ranking system should also weigh behavioral signals:**
> "A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately."

---

# 3. Submission Specification (v4)

**Source**: `submission_spec.docx`

## 3.1 What you're submitting

A CSV file ranking the **top 100 candidates** from `candidates.jsonl` for the released job description.
- Rank 1 = best fit; Rank 100 = 100th best fit
- You do NOT rank candidates 101 onward

## 3.2 File Format

### Filename
Your team's registered participant ID, with `.csv` extension. E.g., `team_xxx.csv`

### Encoding
UTF-8

### Required columns (in this order)
```
candidate_id,rank,score,reasoning
```

| Column | Type | Required? | Description |
|---|---|---|---|
| `candidate_id` | string | ✅ Yes | The `CAND_XXXXXXX` ID from `candidates.jsonl` |
| `rank` | int (1-100) | ✅ Yes | The rank position. Must use each integer 1 through 100 exactly once. |
| `score` | float | ✅ Yes | Your model's score. Must be monotonically non-increasing as rank increases. |
| `reasoning` | string | ⚠ Optional but strongly recommended | 1-2 sentence justification. Used at Stage 4 (manual review). |

### Example
```csv
candidate_id,rank,score,reasoning
CAND_0042871,1,0.987,"Senior AI Engineer with 7 years building RAG systems at product companies; strong recent engagement and Bangalore-based."
CAND_0019884,2,0.973,"6 years applied ML; previously shipped vector search at scale; matches the 'product over research' profile in the JD."
CAND_0091235,3,0.962,"Strong NLP + retrieval background; some concern on notice period (120 days) but otherwise strong fit."
...
CAND_0007729,100,0.412,"Adjacent skills only — likely below cutoff but included as final filler given experience and engagement signals."
```

## 3.3 Rules

### Format rules:
- Exactly 100 rows of data (plus 1 header row)
- Each rank (1 through 100) appears exactly once
- Each `candidate_id` appears exactly once
- Every `candidate_id` must exist in the released `candidates.jsonl`
- Score is non-increasing with rank: `score[rank 1] ≥ score[rank 2] ≥ ... ≥ score[rank 100]`
- If two candidates have the same score: assign unique ranks, break ties deterministically using a secondary signal or by `candidate_id` ascending

### Compute constraints:

| Constraint | Limit |
|---|---|
| Total runtime | ≤ 5 minutes wall-clock |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only — no GPU during ranking |
| Network | Off — no external API calls (no OpenAI, Anthropic, Cohere, Gemini, or any hosted LLM service) |
| Disk | ≤ 5 GB intermediate state |

**Why these constraints?** This is a real-world recruiting system, not a benchmark. A system that calls GPT-4 per candidate cannot scale to 200K candidates in production.

**You CANNOT during the ranking step:**
- Call hosted LLM APIs
- Use GPUs
- Exceed the runtime/memory limits

**Enforcement**: At Stage 3, top-N submissions must provide their full code repository. Your ranking step will be reproduced inside a **sandboxed Docker container** matching these constraints exactly.

### Three-submission cap
- At most 3 submissions total during the competition window
- Your final entry is your **last valid submission**
- Earlier submissions are NOT preserved

### Reasoning column (Stage 4 evaluation)

At Stage 4, judges sample 10 random rows and check each reasoning entry against:

| Check | What they're looking for |
|---|---|
| **Specific facts** | Does the reasoning reference specific facts from the candidate's profile (YoE, current title, named skills, signal values)? |
| **JD connection** | Does it connect to specific JD requirements, not just generic praise? |
| **Honest concerns** | Where the candidate has gaps/concerns, does the reasoning acknowledge them? |
| **No hallucination** | Does every claim correspond to something actually in the candidate's profile? |
| **Variation** | Are the 10 sampled reasonings substantively different from each other (not templated)? |
| **Rank consistency** | Does the reasoning's tone match the rank? Rank-5 with critical reasoning = bad. Rank-95 with glowing reasoning = bad. |

**What's penalized:**
- Empty reasoning
- All-identical reasoning strings
- Templated reasoning that just inserts the candidate's name
- Reasoning that mentions skills NOT in the candidate's profile (hallucination)
- Reasoning that contradicts the rank

> "Plain-language reasoning that demonstrates you actually understood the candidate's profile will rank highly here. Don't try to be impressive; try to be specific and honest."

## 3.4 How Submissions Are Scored

| Metric | Weight | What it measures |
|---|---|---|
| **NDCG@10** | **0.50** | Quality of your top-10 picks |
| NDCG@50 | 0.30 | Quality of your top-50 picks |
| MAP | 0.15 | Precision across all relevance levels |
| P@10 | 0.05 | Fraction of top-10 that are "relevant" (tier 3+) |

### Final Composite Formula
```
Final = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

- Scoring happens **once, after submissions close**
- No public partition, no live leaderboard, no per-submission feedback
- Score computed against the **full hidden ground truth**

### Tiebreaks (if two submissions have identical composites):
1. Higher P@5 wins
2. Higher P@10 wins
3. Earlier submission timestamp wins

## 3.5 Evaluation Pipeline (5 Stages)

| Stage | What happens | What gets you eliminated |
|---|---|---|
| **1. Format validation** | Auto-validator runs on every submission | Any spec violation in Section 3 |
| **2. Scoring** | Composite computed once on full hidden ground truth | Score below cutoff for Stage 3 |
| **3. Code reproduction + honeypot check** | Top-N: full code repo requested. Ranking reproduced in sandbox (5min, 16GB, no GPU, no network). Honeypot rate computed. | Cannot reproduce; honeypot rate >10% in top 100; missing/fabricated repo |
| **4. Manual review** | Reasoning quality (6 checks). Methodology coherence. **Git history authenticity** (real iteration vs single dump). Code quality. | Failed reasoning checks; flat git history; codebase is entirely LLM API calls |
| **5. Defend-your-work interview** | Top X finalists: 30-min video call with Redrob engineering. Walk through architecture, defend design choices, demonstrate familiarity with own code. | Cannot explain architecture; contradicts submitted code; clearly didn't build it |

> **Note on AI tool usage**: You are allowed to use AI tools as part of your workflow. The evaluation filters for genuine engineering, not for absence of AI use.

## 3.6 Common Rejections

- 99 rows or 101 rows instead of exactly 100
- Ranks starting at 0 instead of 1
- Duplicate `candidate_id`s
- `candidate_id` typos that don't exist in `candidates.jsonl`
- All scores set to the same value (model isn't differentiating)
- Scores increasing as rank increases (rank 1 has lowest score)
- Submission file submitted as `.xlsx` or `.json` instead of `.csv`

## 3.7 Honeypot Warning

> "The dataset contains a small number (~80) of honeypot candidates with subtly impossible profiles."

Examples:
- 8 years of experience at a company founded 3 years ago
- "Expert" proficiency in 10 skills with 0 years used

**Honeypots are forced to relevance tier 0 in the ground truth.**

> "If your submission ranks honeypots in the top 10, this is a strong signal that your system isn't reading profiles — it's just doing keyword embedding."

**Disqualification threshold**: Honeypot rate > 10% in top 100 at Stage 3.

> "You can identify honeypots through careful profile inspection. We expect a good ranking system to naturally avoid them; you don't need to special-case them."

## 3.8 What You Submit (Full Picture)

### 10.1 The CSV File
Top-100 ranking per Sections 2-3.

### 10.2 Portal Metadata

| Field | Required? | Notes |
|---|---|---|
| Team name | ✅ Yes | Used in leaderboard and announcements |
| Primary contact name | ✅ Yes | One person as point of contact |
| Primary contact email | ✅ Yes | All organizer communication |
| Primary contact phone | ✅ Yes | Top-N / top-X communication |
| GitHub repository URL | ✅ Yes | Must be reachable. Private repos OK with organizer access at Stage 3 |
| Sandbox / demo link | ✅ Yes | Working hosted environment. See Section 10.5 |
| AI tools declared | ✅ Yes | Multi-select: Claude / ChatGPT / Copilot / Cursor / Gemini / Other / None |
| Compute environment summary | ✅ Yes | One line, e.g. "MacBook Pro M2, 16GB RAM, Python 3.11" |
| Team member list | ✅ Yes | Name + email for each member |
| Methodology summary | Optional | ≤200 words. Strongly recommended — helps at Stage 4 |

### 10.3 Code Repository Requirements

Must include:
- A clear `README.md` with setup instructions and exact commands to reproduce
- Full source code that produced the CSV (no hidden steps, no manual edits)
- Pre-computed artifacts (embeddings, indexes, model weights), or a script that produces them
- `requirements.txt`, `pyproject.toml`, or equivalent
- `submission_metadata.yaml` at repo root (mirroring portal metadata)
- **Single command** that produces submission CSV from candidates file:
  ```bash
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
  ```
- If pre-computation required: document clearly. Pre-computation may exceed 5-min window, but the ranking step must complete within it.

### 10.4 AI Tools Declaration
- Permitted. Designed so AI-assisted + real engineering succeeds. AI-only (paste-and-pray) fails at Stages 3-5.
- Declaration is for transparency, not filtering. Be honest.
- If interview answers contradict declaration → stronger negative signal than AI use itself.

### 10.5 Sandbox / Demo Link Requirement

Acceptable platforms:
- **HuggingFace Spaces** (free tier fine)
- **Streamlit Cloud** (free tier fine)
- **Replit** (public repl)
- **Google Colab** (link to notebook that runs end-to-end)
- **Docker pull + docker run** link to public registry image
- **Binder link** for runnable Jupyter notebook

Requirements:
- Accept a small candidate sample (≤100 candidates)
- Run ranking system end-to-end and produce ranked CSV
- Complete within compute budget (≤5 min on CPU)
- Does NOT need to handle full 100K pool

> **Why mandatory**: "The sandbox is a faster, lower-stakes sanity check that lets us verify the code runs at all before we invest in full reproduction. Submissions without a working sandbox link are flagged at Stage 1."

Alternative: Self-contained `docker run` recipe in GitHub README — Dockerfile must build and run unmodified.

---

# 4. Redrob Behavioral Signals Reference

**Source**: `redrob_signals_doc.docx`

## What are Redrob signals?

Observable candidate behavior beyond their profile:
- Do they actually respond to recruiter messages?
- Have they logged in recently?
- Did they complete assessments?
- Are recruiters saving their profile?
- Have they completed previous interview cycles?

> "These behavioral signals are often more predictive of whether a candidate can actually be hired than their static profile."

> "This dataset includes these signals so that ranking systems can incorporate them as a **multiplier or modifier** on top of skill-match scoring."

## The 23 Signals

| # | Signal | Range / Type | What it measures |
|---|---|---|---|
| 1 | `profile_completeness_score` | 0-100 | How much of the profile they've filled in |
| 2 | `signup_date` | date string | When they signed up on Redrob |
| 3 | `last_active_date` | date string | When they last logged in |
| 4 | `open_to_work_flag` | bool | Have they marked themselves available |
| 5 | `profile_views_received_30d` | integer ≥ 0 | How often recruiters viewed their profile in last 30 days |
| 6 | `applications_submitted_30d` | integer ≥ 0 | How many roles they've applied to recently |
| 7 | `recruiter_response_rate` | 0.0–1.0 | Fraction of recruiter messages they reply to |
| 8 | `avg_response_time_hours` | number ≥ 0 | Median time to respond to a recruiter message |
| 9 | `skill_assessment_scores` | dict[str, 0-100] | Per-skill Redrob assessment scores |
| 10 | `connection_count` | integer ≥ 0 | Number of Redrob connections |
| 11 | `endorsements_received` | integer ≥ 0 | Total skill endorsements received |
| 12 | `notice_period_days` | 0-180 | Their stated notice period |
| 13 | `expected_salary_range_inr_lpa` | {min, max} | Salary expectations in INR lakhs per annum |
| 14 | `preferred_work_mode` | onsite/hybrid/remote/flexible | Their stated work-mode preference |
| 15 | `willing_to_relocate` | bool | Will they relocate if needed |
| 16 | `github_activity_score` | -1 to 100 | GitHub score (-1 if no GitHub linked) |
| 17 | `search_appearance_30d` | integer ≥ 0 | How often they show up in recruiter searches |
| 18 | `saved_by_recruiters_30d` | integer ≥ 0 | How many recruiters bookmarked them in last 30 days |
| 19 | `interview_completion_rate` | 0.0–1.0 | Fraction of interviews actually attended |
| 20 | `offer_acceptance_rate` | -1 to 1.0 | Fraction of offers accepted (-1 if no prior offers) |
| 21 | `verified_email` | bool | Whether email is verified |
| 22 | `verified_phone` | bool | Whether phone is verified |
| 23 | `linkedin_connected` | bool | Whether LinkedIn is connected |

---

# 5. Candidate Schema (JSON Schema)

**Source**: `candidate_schema.json`

## Top-level structure
```json
{
  "candidate_id": "CAND_XXXXXXX",    // 7 digits, unique
  "profile": { ... },
  "career_history": [ ... ],          // 1-10 entries
  "education": [ ... ],               // 0-5 entries
  "skills": [ ... ],                  // 0+ entries
  "certifications": [ ... ],          // 0+ entries
  "languages": [ ... ],               // 0+ entries
  "redrob_signals": { ... }           // 23 behavioral signals
}
```

## `profile` object
| Field | Type | Description |
|---|---|---|
| `anonymized_name` | string | Anonymized full name |
| `headline` | string | One-line professional headline |
| `summary` | string | Multi-sentence professional summary |
| `location` | string | City, region/state |
| `country` | string | Country |
| `years_of_experience` | number (0-50) | Total YoE |
| `current_title` | string | Current job title |
| `current_company` | string | Current employer |
| `current_company_size` | enum | "1-10", "11-50", ..., "10001+" |
| `current_industry` | string | Current industry |

## `career_history` array (1-10 items)
| Field | Type | Description |
|---|---|---|
| `company` | string | Employer name |
| `title` | string | Job title |
| `start_date` | date string | Start date (YYYY-MM-DD) |
| `end_date` | date string or null | End date (null if current) |
| `duration_months` | integer ≥ 0 | Duration in months |
| `is_current` | boolean | Whether this is the current role |
| `industry` | string | Industry |
| `company_size` | enum | Same as profile company_size |
| `description` | string | Role responsibilities and achievements |

## `education` array (0-5 items)
| Field | Type | Description |
|---|---|---|
| `institution` | string | School/university name |
| `degree` | string | Degree type (B.Tech, M.Sc, Ph.D, etc.) |
| `field_of_study` | string | Major/field |
| `start_year` | integer (1970-2030) | |
| `end_year` | integer (1970-2035) | |
| `grade` | string or null | GPA / percentage / class |
| `tier` | enum | "tier_1", "tier_2", "tier_3", "tier_4", "unknown" |

## `skills` array
| Field | Type | Description |
|---|---|---|
| `name` | string | Skill name |
| `proficiency` | enum | "beginner", "intermediate", "advanced", "expert" |
| `endorsements` | integer ≥ 0 | Number of endorsements |
| `duration_months` | integer ≥ 0 | Months of experience with this skill |

## `certifications` array
| Field | Type | Description |
|---|---|---|
| `name` | string | Certification name |
| `issuer` | string | Issuing organization |
| `year` | integer | Year obtained |

## `languages` array
| Field | Type | Description |
|---|---|---|
| `language` | string | Language name |
| `proficiency` | enum | "basic", "conversational", "professional", "native" |

## `redrob_signals` object
All 23 signals as documented in Section 4 above.

---

# 6. Submission Metadata Template

**Source**: `submission_metadata_template.yaml`

```yaml
# Copy to your repo root as `submission_metadata.yaml` and fill it in.
# Stage 3 review uses this file to verify your portal metadata.

# Team identity
team_name: "your-team-name-here"

primary_contact:
  name: "Full Name"
  email: "primary@example.com"
  phone: "+91-XXXXXXXXXX"

team_members:
  - name: "Member 1 Full Name"
    email: "member1@example.com"
    role: "ML Engineer"         # Optional
  - name: "Member 2 Full Name"
    email: "member2@example.com"
    role: "Data Engineer"

# Code and reproducibility
github_repo: "https://github.com/YOUR_USERNAME/YOUR_REPO"
sandbox_link: "https://huggingface.co/spaces/YOUR_USERNAME/redrob-ranker"
reproduce_command: "python rank.py --candidates ./candidates.jsonl --out ./submission.csv"

# Compute environment
compute:
  platform: "MacBook Pro M2"
  cpu_cores: 8
  ram_gb: 16
  python_version: "3.11.4"
  os: "macOS 14.2"
  uses_gpu_for_inference: false      # Must be false
  has_network_during_ranking: false  # Must be false
  pre_computation_required: false    # true if you pre-compute embeddings/train models offline
  pre_computation_time_minutes: 0

# AI tools declaration (transparency only — NOT penalized)
ai_tools_used:
  - "Claude"
  - "GitHub Copilot"
  # Options: "Claude", "ChatGPT", "Copilot", "Cursor", "Gemini", "Codeium", "Other", "None"

ai_usage_summary: |
  Briefly describe how AI tools were used.

# Approach summary (optional but recommended)
methodology_summary: |
  ≤200 word summary of your approach.

# Declarations
declarations:
  read_submission_spec: true
  code_is_original_work: true
  no_collusion: true
  honeypot_check_done: false          # Set true if you explicitly checked
  reproduction_tested: true
```

---

# 7. Official Validator Script

**Source**: `validate_submission.py` (166 lines)

### Key validation rules:
1. File extension must be `.csv`
2. UTF-8 encoded
3. Row 1 = header: `candidate_id,rank,score,reasoning`
4. Exactly 100 data rows (rows 2-101)
5. Each row has exactly 4 columns
6. `candidate_id` matches `CAND_XXXXXXX` (7 digits) — regex: `^CAND_[0-9]{7}$`
7. No duplicate `candidate_id`s
8. `rank` must be integer, 1-100
9. Each rank 1-100 appears exactly once
10. `score` must be valid float
11. Scores are **non-increasing** by rank: `score[rank n] >= score[rank n+1]`
12. Tie-breaking: if `score[rank n] == score[rank n+1]`, then `candidate_id[rank n] < candidate_id[rank n+1]` (ascending)

### Usage:
```bash
python validate_submission.py your_submission.csv
```
Exits with code 0 if valid, code 1 if invalid (with error list).

### Important detail from validator:
```python
# Line 98: str(rank) != rank_s — rank must be a clean integer string
# e.g., "1" is OK, "1.0" is NOT
```

---

# 8. Sample Submission Format

**Source**: `sample_submission.csv`

> **WARNING**: This is NOT a high-quality ranking. It's only a format reference. It intentionally includes honeypot-style candidates (HR Managers, Accountants, Graphic Designers ranked as top AI Engineer fits).

```csv
candidate_id,rank,score,reasoning
CAND_0004989,1,0.9920,HR Manager with 6.1 yrs; 9 AI core skills; response rate 0.76.
CAND_0001195,2,0.9840,HR Manager with 8.7 yrs; 9 AI core skills; response rate 0.20.
CAND_0003114,3,0.9760,ML Engineer with 6.4 yrs; 4 AI core skills; response rate 0.88.
...
CAND_0002689,100,0.2000,Content Writer with 14.7 yrs; 7 AI core skills; response rate 0.57.
```

Notice: Scores go from 0.9920 (rank 1) to 0.2000 (rank 100), decreasing by 0.008 each rank. This is clearly a dummy linear scoring — NOT a real model output.

---

# 9. Sample Candidates (Schema Examples)

**Source**: `sample_candidates.json` (50 candidates, 8242 lines, 300 KB)

### Key observations from examining the sample data:

**Example Candidate 1** — `CAND_0000001` (Ira Vora):
- Backend Engineer at Mindtree, 6.9 YoE
- Located in Toronto, Canada
- Skills include NLP, Fine-tuning LLMs, GANs, Milvus — BUT career descriptions are all about Spark/Airflow/data pipelines
- Has advanced/expert skills in NLP, Speech Recognition, TTS, GANs, Milvus — but assessment scores are mediocre (NLP: 38.8, Fine-tuning LLMs: 41.6)
- **Observation**: This is likely a honeypot — skills don't match career text, assessment scores contradict proficiency claims

**Example Candidate 2** — `CAND_0000002` (Saanvi Sethi):
- Operations Manager at Wipro, 12.5 YoE
- Located in Chennai, India
- Skills: mostly non-AI (Project Management, React, Photoshop, etc.)
- Career history: All operations/marketing/customer support roles
- **Observation**: Clear non-fit — no AI/ML background, non-technical career

**Example Candidate 4** — `CAND_0000004` (Anil Bose):
- Marketing Manager at Dunder Mifflin, 3.8 YoE
- Located in Sydney, Australia
- Has a Ph.D in Electronics AND B.Tech in Machine Learning — BUT career is all marketing/operations
- **Observation**: Education seems planted, career doesn't match. Possible honeypot.

---

# 10. Dataset Info

**Source**: `candidates.jsonl`

- **Total candidates**: 100,000
- **Format**: JSONL (one JSON object per line)
- **File size**: ~465 MB uncompressed
- **ID range**: `CAND_0000001` to `CAND_0100000` (7-digit zero-padded)
- **Structure**: Each line matches the schema in Section 5
- **Top-level keys**: `candidate_id`, `profile`, `career_history`, `education`, `skills`, `certifications`, `languages`, `redrob_signals`

### Known data characteristics:
- ~80 honeypot candidates with subtly impossible profiles
- Keyword stuffers: AI skills on non-technical people
- Plain-language Tier 5s: Real engineers who don't list trendy keywords
- Behavioral twins: Similar profiles with different engagement signals
- Fictional company names used (Dunder Mifflin, Stark Industries, Globex Inc, Initech, Acme Corp)
- Real company names also used (TCS, Wipro, Infosys, Mindtree)

---

# Quick Reference Card

## Scoring Formula
```
Final = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

## Compute Limits
| Limit | Value |
|---|---|
| Runtime | ≤ 5 min |
| RAM | ≤ 16 GB |
| GPU | ❌ None |
| Network | ❌ Offline |
| Disk | ≤ 5 GB |

## Disqualification Triggers
- Format validation failures
- Honeypot rate > 10% in top 100
- Cannot reproduce within compute limits
- Missing/fabricated code repo
- Failed reasoning checks
- Flat git history with no iteration
- Cannot defend architecture in interview

## Submission Checklist
- [ ] CSV with exactly 100 rows + header
- [ ] Scores non-increasing
- [ ] Tied scores have ascending candidate_ids
- [ ] All candidate_ids exist in candidates.jsonl
- [ ] Reasoning is specific, fact-based, varied
- [ ] GitHub repo with README, requirements.txt, submission_metadata.yaml
- [ ] Working sandbox/demo link
- [ ] AI tools declared honestly
- [ ] Compute environment described
- [ ] Team info filled in
